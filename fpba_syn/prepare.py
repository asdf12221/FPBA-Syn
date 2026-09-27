"""Prepare generic SAM conditioning data from a COCO detection dataset."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _caption(names: list[str], template: str) -> str:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    parts = [f"{count} {name}s" if count > 1 else f"a {name}" for name, count in counts.items()]
    return template.format(objects=", ".join(parts))


def prepare_conditioning(
    annotations: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    sam_checkpoint: str | Path,
    category_ids: list[int],
    train_ratio: float = 0.9,
    seed: int = 42,
    device: str = "cuda",
    caption_template: str = "Aerial remote sensing image with {objects}",
) -> None:
    """Create ``train``/``val`` images, masks and metadata.jsonl.

    Masks use the category id as the pixel value and are resized to 512x512,
    which matches the ControlNet training format used by this repository.
    """
    from segment_anything import SamPredictor, sam_model_registry
    from scipy import ndimage

    annotations = Path(annotations)
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in coco.get("images", [])}
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    label_values = {category_id: index + 1 for index, category_id in enumerate(category_ids)}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("category_id") in category_ids:
            by_image[ann["image_id"]].append(ann)
    pairs = []
    for image_id, anns in by_image.items():
        info = images[image_id]
        image_path = image_root / info["file_name"]
        if image_path.is_file():
            pairs.append((Path(info["file_name"]).stem, image_path, anns))
    random.Random(seed).shuffle(pairs)
    split = int(len(pairs) * train_ratio)

    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint)).to(device)
    predictor = SamPredictor(sam)
    for split_name, split_pairs in (("train", pairs[:split]), ("val", pairs[split:])):
        image_out = output_dir / split_name / "images"
        mask_out = output_dir / split_name / "conditioning"
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        metadata = []
        for stem, image_path, anns in tqdm(split_pairs, desc=f"SAM {split_name}"):
            image = Image.open(image_path).convert("RGB")
            array = np.asarray(image)
            predictor.set_image(array)
            combined = np.zeros(array.shape[:2], dtype=np.uint8)
            names = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                category_id = int(ann["category_id"])
                names.append(categories.get(category_id, str(category_id)))
                masks, _, _ = predictor.predict(
                    box=np.asarray([x, y, x + w, y + h]),
                    multimask_output=True,
                )
                if masks is None or len(masks) == 0:
                    continue
                candidate = np.all(masks, axis=0)
                labeled, count = ndimage.label(candidate)
                if count:
                    sizes = ndimage.sum(candidate, labeled, range(1, count + 1))
                    candidate = labeled == (int(np.argmax(sizes)) + 1)
                # Conditioning images are uint8; use compact local labels so
                # arbitrary COCO ids (for example 1001) remain representable.
                combined[candidate] = label_values[category_id]
            image.resize((512, 512), Image.Resampling.BICUBIC).save(image_out / f"{stem}.jpg")
            mask = Image.fromarray(combined).resize((512, 512), Image.Resampling.NEAREST)
            Image.fromarray(np.repeat(np.asarray(mask)[..., None], 3, axis=2)).save(mask_out / f"{stem}.png")
            metadata.append({
                "image": f"images/{stem}.jpg",
                "conditioning_image": f"conditioning/{stem}.png",
                "text": _caption(names, caption_template),
            })
        (output_dir / split_name / "metadata.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metadata),
            encoding="utf-8",
        )
