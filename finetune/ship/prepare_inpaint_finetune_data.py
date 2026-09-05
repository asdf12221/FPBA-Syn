#!/home/jingyue/miniconda3/envs/internimage/bin/python
"""
准备 Inpainting 微调数据: conditioning = 原图目标保留 + 背景灰色
=============================================================
让 ControlNet 学会"看到目标 → 补全真实背景"。
"""
import os, json, random
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# ── Paths ──
SRC_DATA = "/home/jingyue/EarthSynth/mar20_sam_data"      # 现有数据
OUTPUT_DIR = "/home/jingyue/EarthSynth/mar20_inpaint_data" # 新数据
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# 目标膨胀像素 (给边缘留过渡带)
DILATE_PX = 3


def sam_mask_to_binary(mask_img):
    arr = np.array(mask_img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0)


def create_inpaint_conditioning(original_img, mask_img):
    """
    条件图: 目标区域保留原图, 背景设为灰色 (128,128,128).
    模型学习: 在灰色背景上看到飞机 → 生成真实机场背景.
    """
    obj_mask = sam_mask_to_binary(mask_img)

    # 膨胀目标区域, 给边缘留过渡
    obj_dilated = binary_dilation(obj_mask, iterations=DILATE_PX)

    orig_arr = np.array(original_img).astype(np.float32)

    # 背景填灰色
    cond_arr = orig_arr.copy()
    cond_arr[~obj_dilated] = [128, 128, 128]

    return Image.fromarray(cond_arr.astype(np.uint8))


def main():
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/conditioning", exist_ok=True)

        # 读取现有 metadata
        src_jsonl = f"{SRC_DATA}/{split}/metadata.jsonl"
        if not os.path.exists(src_jsonl):
            print(f"SKIP {split}: no metadata.jsonl")
            continue

        with open(src_jsonl) as f:
            entries = [json.loads(l) for l in f]

        new_metadata = []
        for entry in tqdm(entries, desc=f"Preparing {split}"):
            # 原图
            img_name = os.path.splitext(entry["image"].split("/")[-1])[0]
            orig_path = f"{SRC_DATA}/{split}/{entry['image']}"
            orig_img = Image.open(orig_path).convert("RGB").resize((512, 512))

            # mask
            mask_path = f"{SRC_DATA}/{split}/{entry['conditioning_image']}"
            mask_img = Image.open(mask_path).convert("RGB").resize((512, 512))

            # 创建 inpainting conditioning: 目标保留 + 背景灰色
            cond_img = create_inpaint_conditioning(orig_img, mask_img)

            # 保存
            orig_img.save(f"{OUTPUT_DIR}/{split}/images/{img_name}.jpg")
            cond_img.save(f"{OUTPUT_DIR}/{split}/conditioning/{img_name}.png")

            img_rel = f"images/{img_name}.jpg"
            cond_rel = f"conditioning/{img_name}.png"
            new_metadata.append({
                "file_name": img_rel,
                "image": img_rel,
                "conditioning_image": cond_rel,
                "text": entry["text"],
            })

        with open(f"{OUTPUT_DIR}/{split}/metadata.jsonl", "w") as f:
            for e in new_metadata:
                f.write(json.dumps(e) + "\n")
        print(f"  {split}: {len(new_metadata)} samples")

    print(f"\nDone! Data saved to {OUTPUT_DIR}/")
    print(f"Train: {OUTPUT_DIR}/train/")
    print(f"Val:   {OUTPUT_DIR}/val/")


if __name__ == "__main__":
    main()
