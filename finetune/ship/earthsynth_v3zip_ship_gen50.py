#!/usr/bin/env python3
"""用新微调模型 (finetune_ship_1000) 对现有 v3zip 舰船图片生成 50 张新图。
复用 earthsynth_1000_finaldatav2_airplane.py 的 Phase 2 生成逻辑:
  conditioning 掩码 + prompt → StableDiffusionControlNetPipeline (20 步, gs=7.5)
数据: v3zip_ship_sam_data/train 前 50 张 (images + conditioning + Qwen text 均就绪)
输出: /home/jingyue/ship_gen1000_output/
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("EARTHSYNTH_GPU", "1")
import sys, json, random
from pathlib import Path
from PIL import Image
import torch

# ── Config ─────────────────────────────────────────
N_SAMPLES = 50
SHIP_CKPT = "/home/jingyue/EarthSynth/finetune_ship_1000/checkpoint-1000/controlnet"
SD15 = "/home/jingyue/models/sd15_modelscope/AI-ModelScope/stable-diffusion-v1-5"
SAM_DATA = "/home/jingyue/EarthSynth/v3zip_ship_sam_data/train"
OUT_DIR = Path("/home/jingyue/ship_gen1000_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 数据: train 前 N 张 (image / conditioning / text) ──
meta = [json.loads(l) for l in open(f"{SAM_DATA}/metadata.jsonl")]
samples = meta[:N_SAMPLES]
print(f"Samples: {len(samples)}")

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
print("Loading ControlNet...", flush=True)
cn = ControlNetModel.from_pretrained(SHIP_CKPT, torch_dtype=torch.float16)
pipe_es = StableDiffusionControlNetPipeline.from_pretrained(
    SD15, controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
pipe_es.scheduler = UniPCMultistepScheduler.from_config(pipe_es.scheduler.config)
pipe_es.enable_model_cpu_offload()
print("Loaded.", flush=True)

for e in samples:
    stem = Path(e["image"]).stem
    out = OUT_DIR / f"{stem}.png"
    if out.exists():
        continue
    img = Image.open(f"{SAM_DATA}/{e['image']}").convert("RGB")
    cond = Image.open(f"{SAM_DATA}/{e['conditioning_image']}").convert("RGB")
    text = e["text"]
    seed = random.randint(0, 2**32 - 1)
    gen = pipe_es(text, image=cond, num_inference_steps=20,
                  guidance_scale=7.5,
                  generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    gen.save(out)
    print(f"  {stem}: {text[:60]}...", flush=True)

print(f"\nDone! {len(samples)} images → {OUT_DIR}/", flush=True)
