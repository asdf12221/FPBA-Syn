"""
Prepare ALL MAR20 images for EarthSynth ControlNet fine-tuning.
Uses layout files (fast) where available, falls back to XML parsing.
Output: metadata.jsonl format for diffusers training.
"""

import os, json, random, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

MAR20_ROOT = "/home/jingyue/AeroGen/datasets/MAR20-VOC"
SAM_CKPT = "/home/jingyue/models/sam_vit_b.pth"
OUTPUT_DIR = "/home/jingyue/EarthSynth/mar20_finetune_data"
TRAIN_RATIO = 0.9


def load_classes():
    """Load class names + build XML code mapping (A1=class0, A2=class1, ...)"""
    with open(os.path.join(MAR20_ROOT, "classes.txt")) as f:
        classes = [l.strip() for l in f]
    code_to_id = {f"A{i+1}": i for i in range(len(classes))}
    return classes, code_to_id


def load_sam(device="cuda:5"):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    print(f"Loading SAM on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
    sam.to(device)
    return SamAutomaticMaskGenerator(
        model=sam, points_per_side=32,
        pred_iou_thresh=0.86, stability_score_thresh=0.90,
        min_mask_region_area=100,
    )


def sam_to_mask(mask_generator, image_path, target_size=512):
    img = Image.open(image_path).convert("RGB")
    result = mask_generator.generate(np.array(img))
    if not result:
        return Image.fromarray(np.zeros((target_size, target_size, 3), dtype=np.uint8))
    result = sorted(result, key=lambda x: x["area"], reverse=True)
    combined = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
    for idx, r in enumerate(result[:25]):
        combined[r["segmentation"]] = idx + 1
    combined_img = Image.fromarray(combined).resize((target_size, target_size), Image.NEAREST)
    return Image.fromarray(np.stack([np.array(combined_img)] * 3, axis=-1).astype(np.uint8))


def get_bboxes(img_id, code_to_id):
    """Get bboxes for an image. Try layout file first, then XML."""
    # Try layout file
    layout_path = os.path.join(MAR20_ROOT, "layouts", f"{img_id}.txt")
    if os.path.exists(layout_path):
        boxes = []
        with open(layout_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append(int(parts[0]))  # class_id is 0-indexed
        return boxes

    # Fall back to XML
    xml_path = os.path.join(MAR20_ROOT, "Annotations", "Oriented_Bounding_Boxes", f"{img_id}.xml")
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        boxes = []
        for obj in tree.findall("object"):
            code = obj.find("name").text
            if code in code_to_id:
                boxes.append(code_to_id[code])
        return boxes

    return []


def build_caption(box_ids, classes):
    """Build prompt from class IDs."""
    if not box_ids:
        return "A satellite image of an airfield"
    counts = {}
    for cid in box_ids:
        name = classes[cid].split()[0]  # e.g. "F-16 Fighting Falcon" → "F-16"
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name, count in counts.items():
        if count == 1:
            parts.append(f"a {name}")
        else:
            parts.append(f"{count} {name}s")
    return f"A satellite image of an airfield with {', '.join(parts)}"


def main():
    # Clean up old data
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for sub in ["train/images", "train/conditioning", "val/images", "val/conditioning"]:
        os.makedirs(f"{OUTPUT_DIR}/{sub}", exist_ok=True)

    classes, code_to_id = load_classes()
    print(f"Classes: {len(classes)}")
    sam = load_sam()

    # Collect ALL annotated images
    image_dir = os.path.join(MAR20_ROOT, "VOC2007", "JPEGImages")
    all_imgs = sorted([os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")])
    print(f"Total images: {len(all_imgs)}")

    # Build pairs: (img_id, bbox_class_ids)
    pairs = []
    layout_hits = 0
    for img_id in all_imgs:
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            box_ids = get_bboxes(img_id, code_to_id)
            if box_ids:  # Only include annotated images
                pairs.append((img_id, img_path, box_ids))
                if os.path.exists(os.path.join(MAR20_ROOT, "layouts", f"{img_id}.txt")):
                    layout_hits += 1

    print(f"Annotated: {len(pairs)} ({layout_hits} from layouts, {len(pairs)-layout_hits} from XML)")

    # Shuffle and split
    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Process
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        metadata = []
        print(f"\n=== {split_name.upper()} ({len(split_pairs)} images) ===")
        for img_id, img_path, box_ids in tqdm(split_pairs):
            # Resize + save image
            img = Image.open(img_path).convert("RGB").resize((512, 512))
            img.save(f"{OUTPUT_DIR}/{split_name}/images/{img_id}.jpg")

            # SAM mask
            mask = sam_to_mask(sam, img_path)
            mask.save(f"{OUTPUT_DIR}/{split_name}/conditioning/{img_id}.png")

            # Caption
            caption = build_caption(box_ids, classes)
            metadata.append({
                "image": f"images/{img_id}.jpg",
                "conditioning_image": f"conditioning/{img_id}.png",
                "text": caption,
            })

        with open(f"{OUTPUT_DIR}/{split_name}/metadata.jsonl", "w") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")
        print(f"  Saved {len(metadata)} entries")

    print(f"\n=== Dataset ready at {OUTPUT_DIR}/ ===")
    print(f"  Total: {len(pairs)} images ({len(train_pairs)} train / {len(val_pairs)} val)")
    print(f"  Sample: {json.dumps(metadata[0], indent=2)}")


if __name__ == "__main__":
    main()
