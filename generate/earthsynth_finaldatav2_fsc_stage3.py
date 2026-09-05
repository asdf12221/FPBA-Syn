#!/usr/bin/env python3
"""
finaldatav2 发射车(FSC)扩增 - 全流程:Phase 1 背景 + Stage 3 Flux 成图
流程与飞机版(earthsynth_finaldatav2_airplane.py + _stage3.py)完全一致:

源图:  finaldatav2 train 中类别 FSC (COCO id 24) 的全部 485 张
       (与 v6 回退后的发射车 train 划分一致,不碰 val 118 张)
掩码:  直接复用 v5_fsc_sam_data 的 conditioning/*.png (SAM vit_b 已生成,603 张全覆盖 485 源图)
prompt:直接复用 v5_fsc_sam_data 的 metadata.jsonl text (模板 caption)
Phase1:微调后的 EarthSynth ControlNet (finetune_fsc_800) + LaMa 擦除 -> bg/<stem>/rank<N>.png
Stage3:Flux Redux + Fill 重绘目标 -> final/<stem>_rank<N>.png

分片:EARTHSYNTH_GPU=<gpu> PART_ID=<id> N_PART=<n> 确定性取模;跳过已存在文件,可断点续跑。
用法:
  EARTHSYNTH_GPU=1 PART_ID=0 N_PART=2 python earthsynth_finaldatav2_fsc_stage3.py
  PHASE1_ONLY=1 ...  只跑 Phase 1 背景
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("EARTHSYNTH_GPU", "1")
import sys, json, random, time
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch

# ── Config ─────────────────────────────────────────
PART_ID = int(os.environ.get("PART_ID", "0"))
N_PART = int(os.environ.get("N_PART", "1"))
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))
PHASE1_ONLY = os.environ.get("PHASE1_ONLY", "0") == "1"
N_VARIANTS = int(os.environ.get("N_VARIANTS", "3"))
OUT_DIR = Path("/home/jingyue/finaldatav2_fsc_3x_output")
COCO_ANN = "/home/jingyue/datasets/finaldatav2/annotations_train.json"
IMG_DIR = "/home/jingyue/datasets/finaldatav2/images/all"
SAM_DATA = Path("/home/jingyue/EarthSynth/v5_fsc_sam_data")   # 掩码 + prompt 复用
SD15 = "/home/jingyue/models/sd15_modelscope/AI-ModelScope/stable-diffusion-v1-5"
EARTHSYNTH_CKPT = os.environ.get("EARTHSYNTH_CKPT",
    "/home/jingyue/EarthSynth/finetune_fsc_ckpts/checkpoint-800/controlnet")
LAMA_MODEL = "/home/jingyue/Domain-RAG/model/simple_lama"
FLUX_DIR = "/home/jingyue/Domain-RAG/model"
STRENGTH, GUIDANCE = 0.3, 30.0
MAX_SIDE = 896   # Stage3 最大边长(1024 时 Flux attention 峰值 ~23.5GB 会 OOM,24GB 卡;896 峰值 ~18GB)
STAGE3_PROMPT = "Aerial remote sensing satellite image of empty military depot yard background, no vehicles, birds-eye perspective"
(OUT_DIR / "bg").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "final").mkdir(parents=True, exist_ok=True)

# ── 数据:finaldatav2 train FSC (COCO id 24) ────────
coco = json.load(open(COCO_ANN))
imgid2info = {im['id']: im for im in coco['images']}
ann_by_img = defaultdict(list)
for a in coco['annotations']:
    if a['category_id'] == 24:
        ann_by_img[a['image_id']].append(a)
tasks = []
for img_id, anns in ann_by_img.items():
    fname = imgid2info[img_id]['file_name']
    tasks.append((Path(fname).stem, f"{IMG_DIR}/{fname}",
                  [a['bbox'] for a in anns], [a['category_id'] for a in anns]))
tasks.sort(key=lambda t: t[0])
print(f"FSC source images: {len(tasks)}", flush=True)
STEM = os.environ.get("STEM", "")
if STEM:
    want = set(s for s in STEM.split(",") if s)
    tasks = [t for t in tasks if t[0] in want]
    print(f"STEM override: {len(tasks)} source", flush=True)
tasks = [t for i, t in enumerate(tasks) if i % N_PART == PART_ID]
if MAX_TASKS > 0: tasks = tasks[:MAX_TASKS]
print(f"shard {PART_ID}/{N_PART}: {len(tasks)} source -> {len(tasks)*N_VARIANTS} outputs", flush=True)

# ── 掩码 + prompt 复用 (v5_fsc_sam_data, train+val 合并按 stem 查) ──
mask_by_stem, text_by_stem = {}, {}
for split in ('train', 'val'):
    for fp in (SAM_DATA / split / 'conditioning').glob('*.png'):
        mask_by_stem[fp.stem] = fp
    meta = SAM_DATA / split / 'metadata.jsonl'
    if meta.exists():
        for line in open(meta):
            e = json.loads(line)
            text_by_stem[Path(e['image']).stem] = e['text']
missing = [t[0] for t in tasks if t[0] not in mask_by_stem or t[0] not in text_by_stem]
if missing:
    print(f"WARNING: {len(missing)} source missing mask/prompt: {missing[:5]}", flush=True)

# ── Phase 1: 微调 EarthSynth + LaMa 生成背景 ────────
print(f"[1/2] Loading EarthSynth (finetuned: {EARTHSYNTH_CKPT}) + LaMa...", flush=True)
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
cn = ControlNetModel.from_pretrained(EARTHSYNTH_CKPT, torch_dtype=torch.float16)
pipe_es = StableDiffusionControlNetPipeline.from_pretrained(
    SD15, controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
pipe_es.scheduler = UniPCMultistepScheduler.from_config(pipe_es.scheduler.config)
pipe_es.enable_model_cpu_offload()
from simple_lama_inpainting import SimpleLama
lama = SimpleLama(device="cuda")

t0 = time.time()
done = 0
for stem, orig_path, bboxes, cat_ids in tqdm(tasks, desc="Phase1 背景"):
    mask_path = mask_by_stem.get(stem)
    text = text_by_stem.get(stem, "A satellite image of a military depot with a missile launcher vehicle")
    mask_img = Image.open(mask_path).convert('RGB').resize((512, 512)) if mask_path else Image.new('RGB', (512, 512), (0, 0, 0))
    original = Image.open(orig_path).convert('RGB')
    for rank in range(N_VARIANTS):
        bg_path = OUT_DIR / "bg" / f"{stem}" / f"rank{rank+1}.png"
        if bg_path.exists():
            continue
        seed = random.randint(0, 2**32 - 1)
        gen = pipe_es(text, image=mask_img, num_inference_steps=20,
                      guidance_scale=7.5, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        lm = Image.new('L', gen.size, 0); d = ImageDraw.Draw(lm)
        sw, sh = 512 / original.width, 512 / (original.height or 1)
        for b in bboxes:
            d.rectangle([max(0, int(b[0]*sw)), max(0, int(b[1]*sh)),
                        min(511, int((b[0]+b[2])*sw)), min(511, int((b[1]+b[3])*sh))], fill=255)
        clean = lama(gen, lm)
        bg_path.parent.mkdir(parents=True, exist_ok=True); clean.save(bg_path)
        done += 1
        if done % 20 == 0:
            rate = done / (time.time() - t0) if t0 else 0
            print(f"[phase1] {done} bg, {rate:.2f} img/s", flush=True)

del pipe_es, cn, lama; torch.cuda.empty_cache()
print(f"\nPhase1 背景完成 {done} 张 -> {OUT_DIR}/bg/", flush=True)
if PHASE1_ONLY:
    sys.exit(0)

# ── Stage 3: Flux Redux + Fill 成图 ────────────────
print(f"\n[2/2] Loading Flux Redux + Fill...", flush=True)
from diffusers import FluxPriorReduxPipeline, FluxFillPipeline
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
dt = torch.bfloat16
te = CLIPTextModel.from_pretrained(f"{FLUX_DIR}/FLUX.1-dev", subfolder="text_encoder", torch_dtype=dt)
te2 = T5EncoderModel.from_pretrained(f"{FLUX_DIR}/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=dt)
tok = CLIPTokenizer.from_pretrained(f"{FLUX_DIR}/FLUX.1-dev", subfolder="tokenizer")
tok2 = T5TokenizerFast.from_pretrained(f"{FLUX_DIR}/FLUX.1-dev", subfolder="tokenizer_2")
pr = FluxPriorReduxPipeline.from_pretrained(f"{FLUX_DIR}/FLUX.1-Redux-dev",
    text_encoder=te, text_encoder_2=te2, tokenizer=tok, tokenizer_2=tok2, torch_dtype=dt)
pr.enable_model_cpu_offload()
pf = FluxFillPipeline.from_pretrained(f"{FLUX_DIR}/FLUX.1-Fill-dev", torch_dtype=dt)
pf.enable_model_cpu_offload()
print("flux ready", flush=True)

manifest = {}
total = 0
t0 = time.time()
for stem, orig_path, bboxes, cat_ids in tqdm(tasks, desc="Stage3"):
    for rank in range(N_VARIANTS):
        out_name = f"{stem}_rank{rank+1}"
        fp = OUT_DIR / "final" / f"{out_name}.png"
        if fp.exists():
            total += 1; manifest[out_name] = "FSC"; continue
        bg_path = OUT_DIR / "bg" / f"{stem}" / f"rank{rank+1}.png"
        if not bg_path.exists():
            continue
        org = Image.open(orig_path).convert('RGB'); w, h = org.size
        s = min(MAX_SIDE / w, MAX_SIDE / h)
        nw, nh = int(w * s) // 16 * 16, int(h * s) // 16 * 16
        proc = org.resize((nw, nh), Image.BICUBIC)
        sx, sy = nw / w, nh / h
        pb = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in bboxes]
        m = Image.new('L', proc.size, 255); d3 = ImageDraw.Draw(m)
        for b in pb:
            d3.rectangle([max(0, int(b[0])), max(0, int(b[1])),
                         min(proc.size[0], int(b[0]+b[2])), min(proc.size[1], int(b[1]+b[3]))], fill=0)
        bg = Image.open(bg_path).convert('RGB').resize(proc.size, Image.BICUBIC)
        re = pr([bg], prompt=STAGE3_PROMPT, prompt_2="", prompt_embeds_scale=[1.0], pooled_prompt_embeds_scale=[1.0])
        seed = random.randint(0, 2**32 - 1)
        res = pf(image=proc, mask_image=m, height=proc.height, width=proc.width,
                 guidance_scale=GUIDANCE, num_inference_steps=50,
                 prompt_embeds=re.prompt_embeds, pooled_prompt_embeds=re.pooled_prompt_embeds,
                 generator=torch.Generator("cpu").manual_seed(seed), strength=STRENGTH).images[0]
        res.save(fp); total += 1
        manifest[out_name] = "FSC"
        if total % 20 == 0:
            rate = total / (time.time() - t0)
            left = (len(tasks)*N_VARIANTS - total) / rate if rate > 0 else 0
            print(f"[stage3] {total} {rate:.2f} img/s ETA {left/3600:.1f}h", flush=True)

with open(OUT_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"\nDone! {total} images -> {OUT_DIR}/final/, manifest -> {OUT_DIR}/manifest.json", flush=True)
