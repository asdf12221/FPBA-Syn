"""
Generate SAM aircraft masks for ALL MAR20 images (bbox prompt).
~2s per image × 3842 ≈ 2 hours. Uses GPU 1.
"""

import os, json, random, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

MAR20_ROOT = "/home/jingyue/AeroGen/datasets/MAR20-VOC"
SAM_CKPT = "/home/jingyue/models/sam_vit_b.pth"
OUTPUT_DIR = "/home/jingyue/EarthSynth/mar20_sam_data"
TRAIN_RATIO = 0.9


def load_classes():
    with open(os.path.join(MAR20_ROOT, "classes.txt")) as f:
        classes = [l.strip() for l in f]
    return classes, {f"A{i+1}": i for i in range(len(classes))}


def get_aircraft_bboxes(img_id, code_to_id):
    """Get aircraft bboxes from XML annotation."""
    xml_path = os.path.join(MAR20_ROOT, "Annotations", "Oriented_Bounding_Boxes", f"{img_id}.xml")
    tree = ET.parse(xml_path)
    boxes = []
    for obj in tree.findall("object"):
        code = obj.find("name").text
        bb = obj.find("bndbox")
        x1, y1 = int(bb.find("xmin").text), int(bb.find("ymin").text)
        x2, y2 = int(bb.find("xmax").text), int(bb.find("ymax").text)
        if code in code_to_id:
            boxes.append((code_to_id[code], x1, y1, x2, y2))
    return boxes


def build_caption(box_ids, classes):
    if not box_ids: return "A satellite image of an airfield"
    counts = {}
    for cid in box_ids:
        name = classes[cid].split()[0]
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name, count in counts.items():
        parts.append(f"a {name}" if count == 1 else f"{count} {name}s")
    return f"A satellite image of an airfield with {', '.join(parts)}"


def main():
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    classes, code_to_id = load_classes()

    # Load SAM
    from segment_anything import sam_model_registry, SamPredictor
    print("Loading SAM...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to("cuda:1")
    predictor = SamPredictor(sam)

    # Collect all images
    image_dir = os.path.join(MAR20_ROOT, "VOC2007", "JPEGImages")
    all_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")])

    pairs = []
    for img_id in all_ids:
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path): continue
        boxes = get_aircraft_bboxes(img_id, code_to_id)
        if boxes:
            pairs.append((img_id, img_path, boxes))

    print(f"Images: {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * TRAIN_RATIO)

    for split_name, split_pairs in [("train", pairs[:split]), ("val", pairs[split:])]:
        os.makedirs(f"{OUTPUT_DIR}/{split_name}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split_name}/conditioning", exist_ok=True)
        print(f"\n{split_name.upper()}: {len(split_pairs)}")

        metadata = []
        for img_id, img_path, boxes in tqdm(split_pairs):
            # Save target image
            img = Image.open(img_path).convert("RGB")
            orig_size = img.size
            img.resize((512, 512)).save(f"{OUTPUT_DIR}/{split_name}/images/{img_id}.jpg")

            # Generate SAM aircraft mask with shadow prevention
            predictor.set_image(np.array(img))
            combined = np.zeros((orig_size[1], orig_size[0]), dtype=np.uint8)
            for i, (cid, x1, y1, x2, y2) in enumerate(boxes):
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

            # Resize and save mask
            mask_img = Image.fromarray(combined).resize((512, 512), Image.NEAREST)
            mask_rgb = np.stack([np.array(mask_img)] * 3, axis=-1).astype(np.uint8)
            Image.fromarray(mask_rgb).save(f"{OUTPUT_DIR}/{split_name}/conditioning/{img_id}.png")

            # Caption
            caption = build_caption([b[0] for b in boxes], classes)
            metadata.append({
                "image": f"images/{img_id}.jpg",
                "conditioning_image": f"conditioning/{img_id}.png",
                "text": caption,
                "file_name": f"images/{img_id}.jpg",
            })

        with open(f"{OUTPUT_DIR}/{split_name}/metadata.jsonl", "w") as f:
            for e in metadata:
                f.write(json.dumps(e) + "\n")

    print(f"\nDone! {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
