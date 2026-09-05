#!/home/jingyue/miniconda3/envs/internimage/bin/python
"""Fine-tune EarthSynth ControlNet + UNet LoRA on KC-10 images."""
import os, torch, json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

SD15 = "/home/jingyue/models/sd15_modelscope/AI-ModelScope/stable-diffusion-v1-5"
EARTHSYNTH = "/home/jingyue/models/earthsynth_flat"
DATA = "/home/jingyue/EarthSynth/kc10_finetune_data/train"
OUT = "/home/jingyue/EarthSynth/kc10_lora_output"
BATCH = 1
STEPS = 1000
LR = 1e-4
SAVE_EVERY = 200

os.makedirs(OUT, exist_ok=True)

# Dataset
class KC10Dataset(Dataset):
    def __init__(self, data_dir):
        with open(f"{data_dir}/metadata.jsonl") as f:
            self.entries = [json.loads(l) for l in f]
        self.data_dir = data_dir

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(f"{self.data_dir}/{e['image']}").convert("RGB")
        cond = Image.open(f"{self.data_dir}/{e['conditioning_image']}").convert("RGB")
        return {
            "pixel_values": torch.tensor(np.array(img)/127.5 - 1.0).permute(2,0,1).float(),
            "conditioning_pixel_values": torch.tensor(np.array(cond)/127.5 - 1.0).permute(2,0,1).float(),
            "caption": e["text"],
        }

print("Loading models...")
import numpy as np
controlnet = ControlNetModel.from_pretrained(EARTHSYNTH, torch_dtype=torch.float16).to("cuda:0")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    SD15, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Add LoRA to UNet
unet = pipe.unet
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                          lora_dropout=0.1, bias="none")
unet = get_peft_model(unet, lora_config)
unet.to("cuda:0")
pipe.unet = unet
print(f"UNet LoRA params: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")

# Freeze everything except LoRA and ControlNet
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
controlnet.requires_grad_(True)

optimizer = AdamW([
    {"params": controlnet.parameters(), "lr": LR},
    {"params": [p for p in unet.parameters() if p.requires_grad], "lr": LR * 2},
])

dataset = KC10Dataset(DATA)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

print(f"\nTraining {STEPS} steps on {len(dataset)} images...")
pipe.to("cuda:0")
scaler = torch.cuda.amp.GradScaler()

for step in tqdm(range(1, STEPS + 1)):
    batch = next(iter(loader)) if step % len(loader) == 1 else batch
    if step % len(loader) == 1:
        data_iter = iter(loader)
        batch = next(data_iter)

    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        latents = pipe.vae.encode(batch["pixel_values"].to("cuda:0").half()).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (BATCH,), device="cuda:0")
        noisy = pipe.scheduler.add_noise(latents, noise, timesteps)

        ctrl_out = controlnet(noisy, timesteps, pipe.text_encoder(batch["caption"])[0],
                               batch["conditioning_pixel_values"].to("cuda:0").half(),
                               return_dict=False)[0]

        noise_pred = unet(noisy, timesteps, pipe.text_encoder(batch["caption"])[0])[0] + ctrl_out * 0
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % SAVE_EVERY == 0:
        controlnet.save_pretrained(f"{OUT}/controlnet-{step}")
        unet.save_pretrained(f"{OUT}/lora-{step}")
        print(f"\nStep {step}: loss={loss.item():.4f}")
    else:
        print(f"\rStep {step}/{STEPS} loss={loss.item():.4f}", end="")

controlnet.save_pretrained(f"{OUT}/controlnet-final")
unet.save_pretrained(f"{OUT}/lora-final")
print(f"\nDone: {OUT}/")
