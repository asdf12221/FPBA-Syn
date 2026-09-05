"""
Generate SAM ship masks for ALL v3zip ship images (bbox prompt).
复用 prepare_sam_data.py (MAR20 版) 的完全相同逻辑,仅数据源改为 finaldatav3_zip:
  - 类别: 舰船组 HM(0) LQS(1) QHS(2) MS(3)
  - 数据: v3zip train + val 标注中的全部舰船图 (1336 张), 90/10 random split (seed 42)
  - SAM vit_b, COCO bbox 作 box prompt, 多 mask 交集 + 最大连通域 (抗阴影)
  - 512×512, 模板 caption (舰船版) + metadata.jsonl
输出: /home/jingyue/EarthSynth/v3zip_ship_sam_data/{train,val}/{images,conditioning,metadata.jsonl}
GPU 1, ~2s/张 × 1336 ≈ 45 分钟
"""

import os, json, random
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ── Config ─────────────────────────────────────────
ANN_TRAIN = "/home/jingyue/datasets/finaldatav3_zip/annotations_train.json"
ANN_VAL = "/home/jingyue/datasets/finaldatav3_zip/annotations_val.json"
IMG_DIR = "/home/jingyue/datasets/finaldatav3_zip/images/all"
SAM_CKPT = "/home/jingyue/models/sam_vit_b.pth"
OUTPUT_DIR = "/home/jingyue/EarthSynth/v3zip_ship_sam_data"
TRAIN_RATIO = 0.9
DEVICE = "cuda:1"          # GPU 1 空闲

# 舰船组 + caption 用英文描述 (不猜具体舰型, 通用外观)
SHIP_IDS = {0, 1, 2, 3}
SHIP_NAMES = {
    0: "aircraft carrier",
    1: "large warship",
    2: "light warship",
    3: "civilian cargo ship",
}


def build_caption(box_ids):
    """与 MAR20 版 build_caption 同构: 按类别计数 → 模板句."""
    counts = {}
    for cid in box_ids:
        name = SHIP_NAMES[cid]
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name, count in counts.items():
        parts.append(f"a {name}" if count == 1 else f"{count} {name}s")
    return f"A satellite image of the sea surface with {', '.join(parts)}"


def load_ship_pairs():
    """合并 v3zip train + val 标注, 取舰船类."""
    pairs = []  # (stem, img_path, [(cid, x1, y1, x2, y2), ...])
    for ann_path in [ANN_TRAIN, ANN_VAL]:
        coco = json.load(open(ann_path))
        imgid2info = {im['id']: im for im in coco['images']}
        ann_by_img = defaultdict(list)
        for a in coco['annotations']:
            if a['category_id'] in SHIP_IDS:
                ann_by_img[a['image_id']].append(a)
        for img_id, anns in ann_by_img.items():
            info = imgid2info[img_id]
            stem = Path(info['file_name']).stem
            img_path = os.path.join(IMG_DIR, info['file_name'])
            if not os.path.exists(img_path):
                continue
            boxes = []
            for a in anns:
                x, y, w, h = a['bbox']
                boxes.append((a['category_id'], int(x), int(y),
                              int(x + w), int(y + h)))
            pairs.append((stem, img_path, boxes))
    # 按 stem 去重 (防 train/val 图名重叠)
    seen, dedup = set(), []
    for p in pairs:
        if p[0] in seen:
            continue
        seen.add(p[0])
        dedup.append(p)
    return dedup


def main():
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    pairs = load_ship_pairs()
    print(f"Ship images (train+val merged): {len(pairs)}")

    # ── SAM (与 MAR20 版完全一致: SamPredictor vit_b) ──
    from segment_anything import sam_model_registry, SamPredictor
    print(f"Loading SAM on {DEVICE}...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(DEVICE)
    predictor = SamPredictor(sam)

    # ── 90/10 split, seed 42 (同 MAR20) ──
    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * TRAIN_RATIO)

    for split_name, split_pairs in [("train", pairs[:split]), ("val", pairs[split:])]:
        os.makedirs(f"{OUTPUT_DIR}/{split_name}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split_name}/conditioning", exist_ok=True)
        print(f"\n{split_name.upper()}: {len(split_pairs)}")

        metadata = []
        for stem, img_path, boxes in tqdm(split_pairs):
            # Save target image (512×512)
            img = Image.open(img_path).convert("RGB")
            orig_size = img.size
            img.resize((512, 512)).save(f"{OUTPUT_DIR}/{split_name}/images/{stem}.jpg")

            # SAM ship mask with shadow prevention (同 MAR20)
            predictor.set_image(np.array(img))
            combined = np.zeros((orig_size[1], orig_size[0]), dtype=np.uint8)
            for cid, x1, y1, x2, y2 in boxes:
                masks, _, _ = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=np.array([x1, y1, x2, y2]), multimask_output=True,
                )
                if masks is not None and len(masks) > 0:
                    # Intersection of all candidates → tight consensus, excludes shadow
                    mask = np.ones((orig_size[1], orig_size[0]), dtype=bool)
                    for m in masks:
                        mask &= m
                    # Keep only largest connected component (removes shadow tails)
                    if mask.sum() > 0:
                        from scipy.ndimage import label
                        labeled, n = label(mask)
                        if n > 1:
                            sizes = [(labeled == j).sum() for j in range(1, n + 1)]
                            mask = labeled == (np.argmax(sizes) + 1)
                    combined[mask] = cid + 1

            # Resize and save mask (RGB, NEAREST — 同 MAR20)
            mask_img = Image.fromarray(combined).resize((512, 512), Image.NEAREST)
            mask_rgb = np.stack([np.array(mask_img)] * 3, axis=-1).astype(np.uint8)
            Image.fromarray(mask_rgb).save(f"{OUTPUT_DIR}/{split_name}/conditioning/{stem}.png")

            # Caption
            caption = build_caption([b[0] for b in boxes])
            metadata.append({
                "image": f"images/{stem}.jpg",
                "conditioning_image": f"conditioning/{stem}.png",
                "text": caption,
                "file_name": f"images/{stem}.jpg",
            })

        with open(f"{OUTPUT_DIR}/{split_name}/metadata.jsonl", "w") as f:
            for e in metadata:
                f.write(json.dumps(e) + "\n")

    print(f"\nDone! {OUTPUT_DIR}/")
    print(f"  Train: {split}, Val: {len(pairs) - split}")
    print(f"  Sample: {json.dumps(metadata[0], indent=2)}")


if __name__ == "__main__":
    main()
