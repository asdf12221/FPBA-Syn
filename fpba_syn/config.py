"""Configuration loading and validation for the generic generation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

def _path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


@dataclass
class DatasetConfig:
    annotations: Path
    image_root: Path
    category_ids: list[int]
    conditioning_root: Path | None = None
    exclude_annotations: Path | None = None
    caption_fallback: str = "Aerial remote sensing image"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetConfig":
        return cls(
            annotations=_path(raw["annotations"]),
            image_root=_path(raw["image_root"]),
            category_ids=[int(x) for x in raw.get("category_ids", [])],
            conditioning_root=_path(raw.get("conditioning_root")),
            exclude_annotations=_path(raw.get("exclude_annotations")),
            caption_fallback=str(raw.get("caption_fallback", "Aerial remote sensing image")),
        )


@dataclass
class ModelConfig:
    sd15: str
    controlnet: str
    flux_root: str
    device: str = "cuda"
    dtype: str = "float16"
    offload: bool = True
    lama_device: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        return cls(
            sd15=os.path.expandvars(os.path.expanduser(str(raw["sd15"]))),
            controlnet=os.path.expandvars(os.path.expanduser(str(raw["controlnet"]))),
            flux_root=os.path.expandvars(os.path.expanduser(str(raw["flux_root"]))),
            device=str(raw.get("device", "cuda")),
            dtype=str(raw.get("dtype", "float16")),
            offload=bool(raw.get("offload", True)),
            lama_device=raw.get("lama_device"),
        )


@dataclass
class GenerationConfig:
    output_dir: Path
    variants: int = 3
    max_side: int = 896
    phase1_steps: int = 20
    phase1_guidance: float = 7.5
    stage3_steps: int = 50
    stage3_guidance: float = 30.0
    stage3_strength: float = 0.4
    stage3_prompt: str = "Aerial remote sensing image of an empty background"
    seed: int | None = None
    part_id: int = 0
    num_parts: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GenerationConfig":
        return cls(
            output_dir=_path(raw["output_dir"]),
            variants=int(raw.get("variants", 3)),
            max_side=int(raw.get("max_side", 896)),
            phase1_steps=int(raw.get("phase1_steps", 20)),
            phase1_guidance=float(raw.get("phase1_guidance", 7.5)),
            stage3_steps=int(raw.get("stage3_steps", 50)),
            stage3_guidance=float(raw.get("stage3_guidance", 30.0)),
            stage3_strength=float(raw.get("stage3_strength", 0.4)),
            stage3_prompt=str(raw.get("stage3_prompt", "Aerial remote sensing image of an empty background")),
            seed=None if raw.get("seed") is None else int(raw["seed"]),
            part_id=int(raw.get("part_id", 0)),
            num_parts=int(raw.get("num_parts", 1)),
        )


@dataclass
class AppConfig:
    dataset: DatasetConfig
    models: ModelConfig
    generation: GenerationConfig
    name: str = "dataset"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        return cls(
            name=str(raw.get("name", "dataset")),
            dataset=DatasetConfig.from_dict(raw["dataset"]),
            models=ModelConfig.from_dict(raw["models"]),
            generation=GenerationConfig.from_dict(raw["generation"]),
            metadata=dict(raw.get("metadata", {})),
        )


def load_config(path: str | Path) -> AppConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configuration files. Install the project dependencies first.") from exc
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return AppConfig.from_dict(raw)


def validate_config(config: AppConfig, check_models: bool = True) -> list[str]:
    """Return actionable validation errors without importing GPU libraries."""
    errors: list[str] = []
    ds = config.dataset
    if not ds.annotations.is_file():
        errors.append(f"dataset.annotations does not exist: {ds.annotations}")
    if not ds.image_root.is_dir():
        errors.append(f"dataset.image_root does not exist: {ds.image_root}")
    if not ds.category_ids:
        errors.append("dataset.category_ids must contain at least one COCO category id")
    if ds.conditioning_root is None:
        errors.append("dataset.conditioning_root is required for generation")
    elif not ds.conditioning_root.is_dir():
        errors.append(f"dataset.conditioning_root does not exist: {ds.conditioning_root}")
    if ds.exclude_annotations and not ds.exclude_annotations.is_file():
        errors.append(f"dataset.exclude_annotations does not exist: {ds.exclude_annotations}")
    if check_models:
        for label, value in (("models.sd15", config.models.sd15), ("models.controlnet", config.models.controlnet)):
            if not Path(value).exists():
                errors.append(f"{label} does not exist: {value}")
        for name in ("FLUX.1-dev", "FLUX.1-Redux-dev", "FLUX.1-Fill-dev"):
            if not (Path(config.models.flux_root) / name).exists():
                errors.append(f"missing Flux model directory: {Path(config.models.flux_root) / name}")
    gen = config.generation
    if gen.variants < 1:
        errors.append("generation.variants must be >= 1")
    if gen.max_side < 16 or gen.max_side % 16:
        errors.append("generation.max_side must be a positive multiple of 16")
    if gen.num_parts < 1 or not 0 <= gen.part_id < gen.num_parts:
        errors.append("generation.part_id must satisfy 0 <= part_id < num_parts")
    return errors
