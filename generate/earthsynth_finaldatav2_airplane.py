#!/usr/bin/env python3
"""
finaldatav2 train 全部飞机类图片 ×3 扩增生成
复用 earthsynth_scarce 管线:EarthSynth-5000 + LaMa → Flux Redux + Fill(1024 尺寸修复版)

源图:finaldatav2 train 中类别 A1_SU-35 ~ A20_SU-24(COCO id 4-23)的全部 7,579 张
每张生成 N_VARIANTS=3,总量 ≈ 22,737。

掩码:
  - MAR20_<num>.jpg 且 mar20_sam_data 有 conditioning 掩码(≈2,213 张)→ 原样复用
  - 其余(≈5,366 张)→ SAM vit_b(COCO 飞机 bbox 作 box prompt)现生成,缓存 OUT_DIR/masks/
提示词:
  - 原生 MAR20 图用 metadata 原文
  - 其余按类别从 mar20 metadata 池"同类型复用":①类别集完全一致 → ②包含 → ③固定兜底

分片:PART_ID / N_PART 环境变量,确定性取模;跳过已存在文件,可断点续跑。
运行:EARTHSYNTH_GPU=1 PART_ID=0 N_PART=2 python earthsynth_finaldatav2_airplane.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("EARTHSYNTH_GPU", "1")
import sys, json, random, re, xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import torch
try:
    import psutil
    def mem_log(tag):
        vm = psutil.virtual_memory()
        print(f"[mem:{tag}] used {vm.used/1e9:.1f}G / avail {vm.available/1e9:.1f}G", flush=True)
except ImportError:
    def mem_log(tag):
        with open('/proc/meminfo') as f:
            d = dict(l.strip().split(':') for l in f)
        print(f"[mem:{tag}] avail {int(d['MemAvailable'].split()[0])/1e6:.1f}G", flush=True)

# ── Config ─────────────────────────────────────────
N_VARIANTS = 3                       # 每张源图生成几张
PART_ID = int(os.environ.get("PART_ID", "0"))
N_PART = int(os.environ.get("N_PART", "1"))
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))   # 冒烟测试:>0 只跑本分片前 N 张
PHASE1_ONLY = os.environ.get("PHASE1_ONLY", "0") == "1"   # 只做背景图(Phase 1),跳过 FLUX(Phase 2)
EARTHSYNTH_CKPT = "/home/jingyue/EarthSynth/finetune_sam_5000/checkpoint-5000/controlnet"
SD15 = "/home/jingyue/models/sd15_modelscope/AI-ModelScope/stable-diffusion-v1-5"
SAM_DATA = "/home/jingyue/EarthSynth/mar20_sam_data/train"
LAMA_MODEL = "/home/jingyue/Domain-RAG/model/simple_lama"
OUT_DIR = Path("/home/jingyue/finaldatav2_airplane_3x_output")
STRENGTH, GUIDANCE = 0.4, 30.0
STAGE3_PROMPT = "Aerial remote sensing satellite image of empty airfield background, no aircraft, birds-eye perspective"
COCO_ANN = "/home/jingyue/datasets/finaldatav2/annotations_train.json"
IMG_DIR = "/home/jingyue/datasets/finaldatav2/images/all"
SAM_CKPT = "/home/jingyue/models/sam_vit_b.pth"
FALLBACK_PROMPT = ("A satellite image of a military airfield with several aircraft parked "
                   "on the concrete apron, runways and taxiways visible, birds-eye view")
(OUT_DIR / "bg").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "final").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

# ── 数据:finaldatav2 train 飞机类(COCO)────────────
AIR_IDS = set(range(4, 24))          # A1_SU-35 ~ A20_SU-24
coco = json.load(open(COCO_ANN))
imgid2info = {im['id']: im for im in coco['images']}
catid2name = {c['id']: c['name'] for c in coco['categories']}

# v5 验证集排除:finaldatav2 train 源图里属于 finaldatav5 val 的飞机图不能用(泄漏)
V5_VAL_EXCLUDE = set()
cv = json.load(open('/home/jingyue/datasets/finaldatav5/annotations_val.json'))
imgid2info_v = {im['id']: im for im in cv['images']}
for a in cv['annotations']:
    if a['category_id'] in AIR_IDS:
        V5_VAL_EXCLUDE.add(Path(imgid2info_v[a['image_id']]['file_name']).stem)
print(f"v5-val 排除 {len(V5_VAL_EXCLUDE)} 张飞机源图", flush=True)

ann_by_img = defaultdict(list)
for a in coco['annotations']:
    if a['category_id'] in AIR_IDS:
        ann_by_img[a['image_id']].append(a)

tasks = []  # (stem, orig_path, bboxes[x,y,w,h], cat_ids, class_names)
for img_id, anns in ann_by_img.items():
    if Path(imgid2info[img_id]['file_name']).stem in V5_VAL_EXCLUDE:
        continue
    stem = Path(imgid2info[img_id]['file_name']).stem
    orig = f"{IMG_DIR}/{imgid2info[img_id]['file_name']}"
    bboxes = [a['bbox'] for a in anns]
    cat_ids = [a['category_id'] for a in anns]
    classes = sorted({catid2name[a['category_id']] for a in anns})
    tasks.append((stem, orig, bboxes, cat_ids, classes))
tasks.sort(key=lambda t: t[0])
print(f"airplane images total: {len(tasks)}", flush=True)
tasks = [t for i, t in enumerate(tasks) if i % N_PART == PART_ID]
if MAX_TASKS > 0: tasks = tasks[:MAX_TASKS]
print(f"shard {PART_ID}/{N_PART}: {len(tasks)} source images -> {len(tasks)*N_VARIANTS} outputs", flush=True)

# ── 掩码/提示词分配 ────────────────────────────────
with open(f"{SAM_DATA}/metadata.jsonl") as f:
    meta_entries = [json.loads(l) for l in f]
meta_num2text = {e['image'].replace('images/', '').replace('.jpg', ''): e['text'] for e in meta_entries}
MAR20_RE = re.compile(r'^MAR20_(\d+)$')

def short_name(coco_name):
    return coco_name.split('_')[0]   # 'A10_B-1B' -> 'A10'

# 提示词复用池:metadata 图 → MAR20 XML 类别短名集 → 文本列表
xml_cache = {}
def xml_classes(num_id):
    if num_id in xml_cache: return xml_cache[num_id]
    xf = f"/home/jingyue/MAR20/Annotations/Horizontal Bounding Boxes/{num_id}.xml"
    out = set()
    if os.path.exists(xf):
        tree = ET.parse(xf)
        out = {o.find('name').text for o in tree.findall('object')}
    xml_cache[num_id] = out
    return out

pool_by_cls = defaultdict(list)
for e in meta_entries:
    num_id = e['image'].replace('images/', '').replace('.jpg', '')
    cls = xml_classes(num_id)
    if not cls: continue
    pool_by_cls[frozenset(cls)].append(e['text'])
pool_items = sorted(pool_by_cls.items(), key=lambda kv: sorted(kv[0]))

def pick_text(classes):
    ts = frozenset(short_name(c) for c in classes)
    if not ts: return FALLBACK_PROMPT
    for cs, texts in pool_items:            # ① 类别集完全一致
        if cs == ts: return texts[0]
    for cs, texts in pool_items:            # ② 池类别集包含目标
        if cs >= ts: return texts[0]
    return FALLBACK_PROMPT                  # ③ 兜底

def gen_sam_mask(predictor, orig_path, bboxes, cat_ids):
    """生成与 mar20_sam_data 一致的类别语义掩码(像素值 = cid+1,背景 0),512x512 NEAREST。

    算法与 EarthSynth/prepare_sam_data.py 相同:
    1. SAM box prompt(multimask_output=True → 3 个候选)
    2. 候选交集 → tight consensus,排除阴影
    3. 只保留最大连通分量(去掉阴影尾巴)
    4. 掩码值 = COCO category_id - 3(A1=1 ... A20=20,与原生 conditioning 一致)
    """
    from scipy import ndimage
    arr = np.array(Image.open(orig_path).convert('RGB'))
    H, W = arr.shape[:2]
    predictor.set_image(arr)
    combined = np.zeros((H, W), dtype=np.uint8)
    for (x, y, w, h), cid in zip(bboxes, cat_ids):
        box = np.array([[x, y, x + w, y + h]], dtype=float)
        masks, _, _ = predictor.predict(box=box, multimask_output=True)
        if masks is None or len(masks) == 0: continue
        ms = masks[:, 0] if masks.ndim == 4 else masks   # (3, H, W)
        mask = np.ones((H, W), dtype=bool)
        for m in ms:
            mask &= (m > 0.0)                            # 交集:排除阴影
        labeled, num = ndimage.label(mask)
        if num > 0:                                      # 最大连通分量
            sizes = ndimage.sum(mask, labeled, range(1, num + 1))
            mask = labeled == (np.argmax(sizes) + 1)
        combined[mask] = cid - 3
    return Image.fromarray(combined).resize((512, 512), Image.NEAREST).convert('RGB')

def resolve_mask_and_text(predictor, stem, orig_path, bboxes, cat_ids, classes):
    """返回 (mask_path, text)。MAR20 原生子集复用 mar20_sam_data,其余 SAM 兜底。"""
    m = MAR20_RE.match(stem)
    if m and m.group(1) in meta_num2text:
        mp = f"{SAM_DATA}/conditioning/{m.group(1)}.png"
        if os.path.exists(mp):
            return mp, meta_num2text[m.group(1)]
    cache = OUT_DIR / "masks" / f"{stem}.png"
    if not cache.exists():
        gen_sam_mask(predictor, orig_path, bboxes, cat_ids).save(cache)
    return str(cache), pick_text(classes)

# ── Phase 1: EarthSynth + LaMa ─────────────────────
mem_log("start")
print("[1/2] Loading EarthSynth + LaMa...", flush=True)
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
cn = ControlNetModel.from_pretrained(EARTHSYNTH_CKPT, torch_dtype=torch.float16)
pipe_es = StableDiffusionControlNetPipeline.from_pretrained(
    SD15, controlnet=cn, torch_dtype=torch.float16, safety_checker=None)
pipe_es.scheduler = UniPCMultistepScheduler.from_config(pipe_es.scheduler.config)
pipe_es.enable_model_cpu_offload()
from simple_lama_inpainting import SimpleLama
lama = SimpleLama(device="cuda")

from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to("cuda")
sam_predictor = SamPredictor(sam)
print("SAM vit_b ready", flush=True)

gen_tasks = []  # (stem, rank, orig_path, bboxes, classes)
for stem, orig_path, bboxes, cat_ids, classes in tqdm(tasks, desc="Generate+LaMa"):
    mask_path, text = resolve_mask_and_text(sam_predictor, stem, orig_path, bboxes, cat_ids, classes)
    mask_img = Image.open(mask_path).convert('RGB').resize((512, 512))
    original = Image.open(orig_path).convert('RGB')
    for rank in range(N_VARIANTS):
        bg_path = OUT_DIR / "bg" / f"{stem}" / f"rank{rank+1}.png"
        if bg_path.exists():
            gen_tasks.append((stem, rank + 1, orig_path, bboxes, classes)); continue
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
        gen_tasks.append((stem, rank + 1, orig_path, bboxes, classes))

del pipe_es, cn, lama, sam, sam_predictor; torch.cuda.empty_cache()
mem_log("after phase1")
if PHASE1_ONLY:
    print(f"\nPHASE1_ONLY: 背景图完成 {len(gen_tasks)}/{len(tasks)*N_VARIANTS},跳过 Phase 2", flush=True)
    sys.exit(0)

# ── Phase 2: Flux Redux + Fill(尺寸修复版)──────────
print(f"\n[2/2] Loading Flux Redux + Fill ({len(gen_tasks)} tasks)...", flush=True)
from diffusers import FluxPriorReduxPipeline, FluxFillPipeline
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
md = "/home/jingyue/Domain-RAG/model"; dt = torch.bfloat16
te = CLIPTextModel.from_pretrained(f"{md}/FLUX.1-dev", subfolder="text_encoder", torch_dtype=dt)
te2 = T5EncoderModel.from_pretrained(f"{md}/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=dt)
tok = CLIPTokenizer.from_pretrained(f"{md}/FLUX.1-dev", subfolder="tokenizer")
tok2 = T5TokenizerFast.from_pretrained(f"{md}/FLUX.1-dev", subfolder="tokenizer_2")
pr = FluxPriorReduxPipeline.from_pretrained(f"{md}/FLUX.1-Redux-dev",
    text_encoder=te, text_encoder_2=te2, tokenizer=tok, tokenizer_2=tok2, torch_dtype=dt)
pr.enable_model_cpu_offload()
pf = FluxFillPipeline.from_pretrained(f"{md}/FLUX.1-Fill-dev", torch_dtype=dt)
pf.enable_model_cpu_offload()
mem_log("after flux load")

total = 0
manifest = {}
for stem, rank, orig_path, bboxes, classes in tqdm(gen_tasks, desc="Stage3"):
    out_name = f"{stem}_rank{rank}"
    fp = OUT_DIR / "final" / f"{out_name}.png"
    if fp.exists(): total += 1; manifest[out_name] = classes; continue
    bg_path = OUT_DIR / "bg" / f"{stem}" / f"rank{rank}.png"
    if not bg_path.exists(): continue
    org = Image.open(orig_path).convert('RGB'); w, h = org.size
    # 统一放进 1024x1024(长边压到 1024,16 对齐)——scarce 跑通的修复版逻辑
    s = min(1024.0 / w, 1024.0 / h)
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
    manifest[out_name] = classes

with open(OUT_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
mem_log("done")

print(f"\nDone! {total} images → {OUT_DIR}/final/, manifest → {OUT_DIR}/manifest.json", flush=True)
