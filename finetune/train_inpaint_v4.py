#!/home/jingyue/miniconda3/envs/internimage/bin/python
"""
EarthSynth V4 — 真正的 ControlNet Inpainting 训练
==================================================
训练时:
- UNet 输入从 4→9 通道 (noisy_latent + masked_latent + mask)
- mask=1 区域加噪→模型学习重绘, mask=0 区域不加噪→模型学会保留
- ControlNet 仍用 SAM mask 做空间引导

推理时:
- 原图目标保留、背景加噪 → 模型只重绘背景
- 目标像素完全不变, 零贴图感
"""
import os, json, math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from diffusers import (
    ControlNetModel, UniPCMultistepScheduler,
    AutoencoderKL, UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# ── Config ──
SD15 = "/home/jingyue/models/sd15_modelscope/AI-ModelScope/stable-diffusion-v1-5"
EARTHSYNTH = "/home/jingyue/models/earthsynth_flat"
TRAIN_DATA = "/home/jingyue/EarthSynth/mar20_sam_data/train"
OUTPUT_DIR = "/home/jingyue/EarthSynth/finetune_sam_v4_inpaint"

BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_STEPS = 5000
LR = 1e-5
WARMUP_STEPS = 200
SAVE_EVERY = 1000
RESOLUTION = 512
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Distributed ──
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
device = torch.device(f"cuda:{local_rank}")
if world_size > 1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


class InpaintDataset(Dataset):
    def __init__(self, data_dir, resolution=512):
        with open(f"{data_dir}/metadata.jsonl") as f:
            self.entries = [json.loads(l) for l in f]
        self.data_dir = data_dir
        self.resolution = resolution

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(f"{self.data_dir}/{e['image']}").convert("RGB").resize(
            (self.resolution, self.resolution))
        mask = Image.open(f"{self.data_dir}/{e['conditioning_image']}").convert("RGB").resize(
            (self.resolution, self.resolution))
        # SAM mask: object pixels > 0
        sam_arr = np.array(mask)[:, :, 0]
        obj_mask = (sam_arr > 0).astype(np.float32)
        # 膨胀目标区域留过渡带
        obj_dilated = binary_dilation(obj_mask, iterations=3).astype(np.float32)
        # Inpainting mask 反向: 背景=1(重绘), 目标=0(保留)
        inpaint_mask = 1.0 - obj_dilated
        # ControlNet 条件: 保持原始 SAM mask (3通道, 归一化到 [-1,1])
        cond_np = np.array(mask).astype(np.float32) / 127.5 - 1.0
        return {
            "pixel_values": torch.tensor(np.array(img) / 127.5 - 1.0).permute(2, 0, 1).float(),
            "mask_values": torch.tensor(inpaint_mask).float(),  # 1=重绘背景, 0=保留目标
            "conditioning_values": torch.tensor(cond_np).permute(2, 0, 1).float(),
            "caption": e["text"],
        }


def expand_unet_input(unet):
    """将 UNet 第一层卷积从 4→9 通道, 新权重零初始化"""
    old_conv = unet.conv_in
    new_conv = nn.Conv2d(9, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=old_conv.bias is not None)
    # 前4通道复制原权重, 后5通道零初始化
    new_conv.weight.data[:, :4] = old_conv.weight.data
    new_conv.weight.data[:, 4:] = 0
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data
    unet.conv_in = new_conv
    return unet


if local_rank == 0:
    print(f"Loading base models on {world_size} GPU(s)...")

# ── Load ControlNet in FP32 (GradScaler needs fp32 master weights) ──
controlnet = ControlNetModel.from_pretrained(EARTHSYNTH).to(device)

# ── Load SD components in fp16 for memory ──
vae = AutoencoderKL.from_pretrained(SD15, subfolder="vae", torch_dtype=torch.float16).to(device)
tokenizer = CLIPTokenizer.from_pretrained(SD15, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(SD15, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

# UNet in fp16 (frozen, no grads needed)
unet = UNet2DConditionModel.from_pretrained(SD15, subfolder="unet")
unet = expand_unet_input(unet)  # 4→9 channels
unet.to(device, dtype=torch.float16)

from diffusers import DDPMScheduler
noise_scheduler = DDPMScheduler.from_pretrained(SD15, subfolder="scheduler")

# Freeze
unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Train ControlNet only (fp32)
controlnet.train()
controlnet.requires_grad_(True)

# ── DDP ──
if world_size > 1:
    controlnet = DDP(controlnet, device_ids=[local_rank], output_device=local_rank)

controlnet_raw = controlnet.module if world_size > 1 else controlnet

# Optimizer + Scheduler
optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    return 1.0
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Dataset
dataset = InpaintDataset(TRAIN_DATA)
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if world_size > 1 else None
loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(sampler is None),
                    num_workers=4, pin_memory=True)

scaler = GradScaler()

if local_rank == 0:
    trainable = sum(p.numel() for p in controlnet_raw.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Training {MAX_STEPS} steps, batch={BATCH_SIZE}×{world_size}×{GRAD_ACCUM}={BATCH_SIZE*world_size*GRAD_ACCUM}")

# ── Training ──
optimizer.zero_grad()
data_iter = iter(loader)

for step in range(1, MAX_STEPS + 1):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        batch = next(data_iter)

    with autocast():
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
        mask_values = batch["mask_values"].to(device).unsqueeze(1)  # (B,1,H,W)
        cond_values = batch["conditioning_values"].to(device, dtype=torch.float16)

        # Encode original to latents
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        bsz, _, h, w = latents.shape

        # Resize mask to latent resolution
        mask_latent = F.interpolate(mask_values, size=(h, w), mode="nearest")

        # ── Inpainting setup ──
        # masked_latent: original latents with mask=1 areas zeroed
        masked_latent = latents * (1 - mask_latent)

        # Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                  device=device, dtype=torch.long)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # ── UNet input: concat(noisy_latent, masked_latent, mask) = 9 channels ──
        unet_input = torch.cat([noisy_latents, masked_latent, mask_latent], dim=1)

        # Text encoding
        text_inputs = tokenizer(batch["caption"], padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt")
        encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]

        # ControlNet
        down_block_res_samples, mid_block_res_sample = controlnet_raw(
            noisy_latents, timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=cond_values,
            return_dict=False,
        )

        # UNet (frozen weights, but gradients flow through to ControlNet)
        noise_pred = unet(
            unet_input, timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )[0]

        # Loss: standard MSE
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        loss = loss / GRAD_ACCUM

    scaler.scale(loss).backward()

    if step % GRAD_ACCUM == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()

    if local_rank == 0:
        if step % 100 == 0 or step <= 10:
            print(f"Step {step}/{MAX_STEPS} loss={loss.item() * GRAD_ACCUM:.4f} lr={lr_scheduler.get_last_lr()[0]:.2e}")
        elif step % 10 == 0:
            print(f"\rStep {step}/{MAX_STEPS} loss={loss.item() * GRAD_ACCUM:.4f}", end="")

        if step % SAVE_EVERY == 0:
            save_dir = f"{OUTPUT_DIR}/checkpoint-{step}"
            controlnet_raw.save_pretrained(save_dir)
            print(f"\nSaved to {save_dir}")

if local_rank == 0:
    controlnet_raw.save_pretrained(OUTPUT_DIR)
    print(f"\nDone! → {OUTPUT_DIR}")

if world_size > 1:
    dist.destroy_process_group()
