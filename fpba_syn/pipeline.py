"""Two-stage EarthSynth + LaMa + Flux generation pipeline."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, TYPE_CHECKING

from PIL import Image, ImageDraw
from tqdm import tqdm

from .config import AppConfig
from .data import ConditioningStore, CocoTaskSource, Task, shard

if TYPE_CHECKING:
    import torch


def _dtype(name: str):
    import torch
    values = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported models.dtype: {name}") from exc


def _bbox_mask(size: tuple[int, int], boxes: tuple[tuple[float, float, float, float], ...]) -> Image.Image:
    mask = Image.new("L", size, 255)
    draw = ImageDraw.Draw(mask)
    for x, y, w, h in boxes:
        draw.rectangle((max(0, int(x)), max(0, int(y)), min(size[0], int(x + w)), min(size[1], int(y + h))), fill=0)
    return mask


class SyntheticGenerator:
    def __init__(self, config: AppConfig, phase1_only: bool = False, max_tasks: int | None = None):
        self.config = config
        self.phase1_only = phase1_only
        self.max_tasks = max_tasks
        self.device = config.models.device
        self.dtype = _dtype(config.models.dtype)
        self.out = config.generation.output_dir
        self.bg_out = self.out / "bg"
        self.final_out = self.out / "final"
        self.bg_out.mkdir(parents=True, exist_ok=True)
        self.final_out.mkdir(parents=True, exist_ok=True)

    def _tasks(self) -> tuple[list[Task], ConditioningStore]:
        all_tasks = CocoTaskSource(self.config.dataset).load()
        tasks = shard(all_tasks, self.config.generation.part_id, self.config.generation.num_parts)
        if self.max_tasks is not None:
            tasks = tasks[: self.max_tasks]
        if self.config.dataset.conditioning_root is None:
            raise ValueError("dataset.conditioning_root is required")
        return tasks, ConditioningStore(self.config.dataset.conditioning_root, self.config.dataset.caption_fallback)

    def _load_phase1(self):
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

        controlnet = ControlNetModel.from_pretrained(self.config.models.controlnet, torch_dtype=self.dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.models.sd15,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if self.config.models.offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        from simple_lama_inpainting import SimpleLama

        lama = SimpleLama(device=self.config.models.lama_device or self.device)
        return pipe, controlnet, lama

    def _load_flux(self):
        import torch
        from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        root = Path(self.config.models.flux_root)
        text_dtype = torch.bfloat16
        dev = root / "FLUX.1-dev"
        text_encoder = CLIPTextModel.from_pretrained(dev, subfolder="text_encoder", torch_dtype=text_dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(dev, subfolder="text_encoder_2", torch_dtype=text_dtype)
        tokenizer = CLIPTokenizer.from_pretrained(dev, subfolder="tokenizer")
        tokenizer_2 = T5TokenizerFast.from_pretrained(dev, subfolder="tokenizer_2")
        redux = FluxPriorReduxPipeline.from_pretrained(
            root / "FLUX.1-Redux-dev",
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            torch_dtype=text_dtype,
        )
        fill = FluxFillPipeline.from_pretrained(root / "FLUX.1-Fill-dev", torch_dtype=text_dtype)
        if self.config.models.offload:
            redux.enable_model_cpu_offload()
            fill.enable_model_cpu_offload()
        else:
            redux.to(self.device)
            fill.to(self.device)
        return redux, fill

    def _seed(self, task_index: int, rank: int) -> int:
        base = self.config.generation.seed
        if base is None:
            return random.randint(0, 2**32 - 1)
        return base + task_index * self.config.generation.variants + rank

    def run(self) -> dict[str, Any]:
        import torch
        tasks, conditions = self._tasks()
        print(f"[{self.config.name}] {len(tasks)} source images, {len(tasks) * self.config.generation.variants} variants")
        pipe, controlnet, lama = self._load_phase1()
        generated: list[tuple[Task, int]] = []
        for index, task in enumerate(tqdm(tasks, desc="Phase 1 / background")):
            mask_path, prompt = conditions.resolve(task)
            original = Image.open(task.image_path).convert("RGB")
            condition = Image.open(mask_path).convert("RGB").resize((512, 512))
            for rank in range(self.config.generation.variants):
                bg_path = self.bg_out / task.stem / f"rank{rank + 1}.png"
                generated.append((task, rank + 1))
                if bg_path.exists():
                    continue
                seed = self._seed(index, rank)
                result = pipe(
                    prompt,
                    image=condition,
                    num_inference_steps=self.config.generation.phase1_steps,
                    guidance_scale=self.config.generation.phase1_guidance,
                    generator=torch.Generator("cpu").manual_seed(seed),
                ).images[0]
                scale_x = result.width / original.width
                scale_y = result.height / original.height
                erase = _bbox_mask(
                    result.size,
                    tuple((x * scale_x, y * scale_y, w * scale_x, h * scale_y) for x, y, w, h in task.boxes),
                )
                clean = lama(result, Image.eval(erase, lambda value: 255 - value))
                bg_path.parent.mkdir(parents=True, exist_ok=True)
                clean.save(bg_path)
        del pipe, controlnet, lama
        if self.phase1_only:
            return {"phase": "background", "sources": len(tasks), "variants": len(generated)}

        redux, fill = self._load_flux()
        manifest: dict[str, Any] = {}
        for task, rank in tqdm(generated, desc="Stage 3 / redraw"):
            output_name = f"{task.stem}_rank{rank}"
            output_path = self.final_out / f"{output_name}.png"
            if output_path.exists():
                manifest[output_name] = {"category": self.config.name, "source": task.stem}
                continue
            bg_path = self.bg_out / task.stem / f"rank{rank}.png"
            if not bg_path.exists():
                continue
            original = Image.open(task.image_path).convert("RGB")
            width, height = original.size
            scale = min(self.config.generation.max_side / width, self.config.generation.max_side / height)
            size = (max(16, int(width * scale) // 16 * 16), max(16, int(height * scale) // 16 * 16))
            image = original.resize(size, Image.Resampling.BICUBIC)
            sx, sy = size[0] / width, size[1] / height
            boxes = tuple((x * sx, y * sy, w * sx, h * sy) for x, y, w, h in task.boxes)
            mask = _bbox_mask(size, boxes)
            bg = Image.open(bg_path).convert("RGB").resize(size, Image.Resampling.BICUBIC)
            embeds = redux([bg], prompt=self.config.generation.stage3_prompt, prompt_2="", prompt_embeds_scale=[1.0], pooled_prompt_embeds_scale=[1.0])
            result = fill(
                image=image,
                mask_image=mask,
                height=size[1],
                width=size[0],
                guidance_scale=self.config.generation.stage3_guidance,
                num_inference_steps=self.config.generation.stage3_steps,
                prompt_embeds=embeds.prompt_embeds,
                pooled_prompt_embeds=embeds.pooled_prompt_embeds,
                generator=torch.Generator("cpu").manual_seed(self._seed(len(manifest), rank - 1)),
                strength=self.config.generation.stage3_strength,
            ).images[0]
            result.save(output_path)
            manifest[output_name] = {"category": self.config.name, "source": task.stem}
        with (self.out / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return {"phase": "complete", "sources": len(tasks), "outputs": len(manifest), "manifest": str(self.out / "manifest.json")}
