"""Merge generated images back into a COCO annotation file."""

from __future__ import annotations

import copy
import json
from pathlib import Path


def augment_coco(source: str | Path, generated_dir: str | Path, output: str | Path) -> None:
    source = Path(source)
    generated_dir = Path(generated_dir)
    output = Path(output)
    coco = json.loads(source.read_text(encoding="utf-8"))
    result = copy.deepcopy(coco)
    images = {item["id"]: item for item in coco.get("images", [])}
    by_stem = {Path(item["file_name"]).stem: item for item in coco.get("images", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    next_image = max(images, default=-1) + 1
    next_ann = max((a["id"] for a in coco.get("annotations", [])), default=-1) + 1
    for generated in sorted(generated_dir.glob("*.png")):
        stem = generated.stem.rsplit("_rank", 1)[0]
        source_image = by_stem.get(stem)
        if source_image is None:
            continue
        new_image = copy.deepcopy(source_image)
        new_image["id"] = next_image
        new_image["file_name"] = str(generated)
        with generated.open("rb") as handle:
            from PIL import Image
            with Image.open(handle) as image:
                new_image["width"], new_image["height"] = image.size
        sx = new_image["width"] / float(source_image.get("width", new_image["width"]))
        sy = new_image["height"] / float(source_image.get("height", new_image["height"]))
        result["images"].append(new_image)
        for ann in anns_by_image.get(source_image["id"], []):
            new_ann = copy.deepcopy(ann)
            new_ann["id"] = next_ann
            new_ann["image_id"] = next_image
            x, y, width, height = ann["bbox"]
            new_ann["bbox"] = [x * sx, y * sy, width * sx, height * sy]
            new_ann["area"] = float(ann.get("area", width * height)) * sx * sy
            result["annotations"].append(new_ann)
            next_ann += 1
        next_image += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(result['images'])} images and {len(result['annotations'])} annotations to {output}")
