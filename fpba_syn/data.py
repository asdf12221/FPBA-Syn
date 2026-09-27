"""COCO task and conditioning-data adapters."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import DatasetConfig


@dataclass(frozen=True)
class Task:
    stem: str
    image_path: Path
    boxes: tuple[tuple[float, float, float, float], ...]


class CocoTaskSource:
    def __init__(self, config: DatasetConfig):
        self.config = config

    @staticmethod
    def _load(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _excluded_stems(self) -> set[str]:
        if not self.config.exclude_annotations:
            return set()
        coco = self._load(self.config.exclude_annotations)
        images = {x["id"]: x for x in coco.get("images", [])}
        return {
            Path(images[a["image_id"]]["file_name"]).stem
            for a in coco.get("annotations", [])
            if a.get("category_id") in self.config.category_ids
        }

    def load(self) -> list[Task]:
        coco = self._load(self.config.annotations)
        images = {x["id"]: x for x in coco.get("images", [])}
        boxes_by_image: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            if ann.get("category_id") in self.config.category_ids:
                boxes_by_image[ann["image_id"]].append(tuple(float(v) for v in ann["bbox"]))
        excluded = self._excluded_stems()
        tasks = []
        for image_id, boxes in boxes_by_image.items():
            info = images[image_id]
            stem = Path(info["file_name"]).stem
            path = self.config.image_root / info["file_name"]
            if stem in excluded:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Image listed by COCO is missing: {path}")
            tasks.append(Task(stem, path, tuple(boxes)))
        return sorted(tasks, key=lambda x: x.stem)


class ConditioningStore:
    """Find conditioning PNGs and prompts in train/val style directories.

    The accepted layout is either ``root/conditioning`` plus ``metadata.jsonl``
    or ``root/{train,val}/conditioning`` plus a metadata file in each split.
    """

    def __init__(self, root: Path, fallback_prompt: str):
        self.root = root
        self.fallback_prompt = fallback_prompt
        self.masks: dict[str, Path] = {}
        self.prompts: dict[str, str] = {}
        self._read()

    def _read(self) -> None:
        dirs = [self.root]
        dirs.extend(p for p in (self.root / "train", self.root / "val") if p.is_dir())
        for base in dirs:
            for path in (base / "conditioning").glob("*.png"):
                self.masks[path.stem] = path
            for meta in (base / "metadata.jsonl",):
                if not meta.is_file():
                    continue
                for line in meta.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    name = Path(item.get("image", item.get("file_name", ""))).stem
                    if name:
                        self.prompts[name] = str(item.get("text", self.fallback_prompt))

    def resolve(self, task: Task) -> tuple[Path, str]:
        mask = self.masks.get(task.stem)
        if mask is None:
            raise FileNotFoundError(
                f"No conditioning mask for '{task.stem}' under {self.root}. "
                "Run the conditioning-preparation step or provide a matching PNG."
            )
        return mask, self.prompts.get(task.stem, self.fallback_prompt)


def shard(tasks: Iterable[Task], part_id: int, num_parts: int) -> list[Task]:
    return [task for index, task in enumerate(tasks) if index % num_parts == part_id]


def verify_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size

