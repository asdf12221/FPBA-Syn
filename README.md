<div align="center">

# FPBA-Syn

### Foreground-Preserving · Background-Adaptive · Remote-Sensing Synthesis

Generate detection-ready aerial and satellite imagery while preserving the
target location and adapting the surrounding scene.

<p>
  <a href="https://github.com/asdf12221/FPBA-Syn"><img src="https://img.shields.io/badge/status-source--only-4c8bf5" alt="Source-only project"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/Annotations-COCO-5965A8" alt="COCO annotations">
  <img src="https://img.shields.io/badge/License-Apache--2.0-256A65" alt="Apache 2.0 license">
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#pipeline">Pipeline</a> ·
  <a href="#reference-results">Results</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#project-layout">Layout</a>
</p>

</div>

<p align="center">
  <img src="assets/FPBA-Syn.png" alt="FPBA-Syn generation pipeline" width="940">
</p>

> **In one sentence:** FPBA-Syn adapts the background around a known object,
> redraws the object in place, and exports images plus COCO annotations for
> downstream remote-sensing detection.

## Overview

FPBA-Syn is a configurable, class-agnostic synthesis framework for remote
sensing. It starts from COCO images and bounding boxes, creates a compatible
scene, preserves the target location, and produces an augmented COCO dataset.

### Why use it?

| Capability | What it provides |
| --- | --- |
| **Foreground preserving** | Keeps the source box as the spatial anchor for synthesis. |
| **Background adaptive** | Generates the surrounding scene instead of pasting an object onto a fixed background. |
| **Detection ready** | Writes generated images and COCO annotations for training pipelines. |
| **Class agnostic** | Categories, paths, prompts and generation settings live in YAML. |
| **Scalable** | Resumable jobs with `part_id` / `num_parts` for multi-GPU or multi-machine runs. |
| **Auditable** | Separates background and final outputs and records manifests for inspection. |

## Pipeline

<table>
<tr>
<td width="33%" valign="top"><b>01 · Prepare</b><br><br>
Build masks and conditioning captions from COCO boxes with SAM.<br><br>
<code>COCO → masks + captions</code>
</td>
<td width="33%" valign="top"><b>02 · Adapt</b><br><br>
Synthesize a compatible scene with EarthSynth/ControlNet and remove the source
object with LaMa.<br><br>
<code>conditioning → clean background</code>
</td>
<td width="33%" valign="top"><b>03 · Compose</b><br><br>
Redraw the target in place with Flux Redux + Fill and export augmented labels.<br><br>
<code>background → final image + COCO JSON</code>
</td>
</tr>
</table>

The generated filename convention is `<source_stem>_rank<N>.png`. Existing
outputs are skipped, so interrupted jobs can be resumed safely.

## Milestones

- **2026-9-5**：🎉 Congratulations！FPBA-Syn 项目在 **挑战杯“揭榜挂帅”专项赛 XH-202625** 初赛阶段取得 **前 20%** 的成绩 🚀

## Reference results

The recorded reference run generated **24,732 final images** across three
remote-sensing target categories:

| Category | Final images | Detector precision | Detector recall |
| :--- | ---: | ---: | ---: |
| Airplane | 20,472 | 92.3% | 93.1% |
| FSC / launch vehicle | 1,455 | 83.5% | 83.4% |
| Ship | 2,805 | 75.4% | 83.9% |
| **Total** | **24,732** | **89.9%** | **91.5%** |

The total precision and recall are image-count-weighted averages across the
three categories.

In one recorded downstream setup:

| Training setup | bbox mAP | mAP50 | mAP75 |
| --- | ---: | ---: | ---: |
| Filtered synthetic pretraining | 0.683 | 0.932 | 0.835 |
| Fine-tuned on real data | **0.739** | **0.948** | **0.889** |

The detector audit uses class-matched greedy matching at IoU ≥ 0.5. These are
reference-run numbers for pipeline validation, not a turnkey benchmark or a
claim of human perceptual quality. See [`docs/experiments.md`](docs/experiments.md)
for settings and limitations.

## Quick start

### 1. Install

Set up a CUDA-compatible PyTorch build first, then install the package:

```bash
pip install -e .
```

The optional `prepare` command also requires SAM:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Configure and validate

Copy [`configs/example.yaml`](configs/example.yaml), set your dataset/model
paths, and validate the configuration before starting a long run:

```bash
fpba-syn check --config configs/example.yaml --skip-models
fpba-syn check --config configs/example.yaml
```

### 3. Prepare conditioning data

```bash
fpba-syn prepare \
  --annotations /data/train/annotations.json \
  --image-root /data/train/images \
  --output /data/conditioning \
  --sam-checkpoint /models/sam_vit_b.pth \
  --category-ids 4,5,6 \
  --device cuda
```

### 4. Generate and annotate

Start with one task, then scale out after the output looks correct:

```bash
fpba-syn generate --config configs/example.yaml --max-tasks 1
```

To stop after background synthesis, add `--phase1-only`. Build the augmented
COCO file from completed images with:

```bash
fpba-syn annotate \
  --source /data/train/annotations.json \
  --generated-dir /outputs/example/final \
  --output /data/train/annotations_synthetic.json
```

If the redraw changes object geometry, run a detector or relabelling pass
before using the generated labels for a final benchmark.

## Models and configuration

<details>
<summary><b>Required model components</b></summary>

Full generation requires local, licensed copies of:

- Stable Diffusion 1.5
- a fine-tuned EarthSynth ControlNet
- `FLUX.1-dev`, `FLUX.1-Redux-dev`, and `FLUX.1-Fill-dev`
- LaMa / `simple-lama-inpainting`
- SAM ViT-B when creating conditioning data

Check every third-party model and dataset license before redistribution.

</details>

<details>
<summary><b>Minimal YAML example</b></summary>

```yaml
dataset:
  annotations: ${DATA_ROOT}/annotations_train.json
  image_root: ${DATA_ROOT}/images
  category_ids: [1]
  conditioning_root: ${DATA_ROOT}/conditioning

models:
  sd15: ${MODEL_ROOT}/stable-diffusion-v1-5
  controlnet: ${MODEL_ROOT}/earthsynth_controlnet
  flux_root: ${MODEL_ROOT}
  device: cuda

generation:
  output_dir: ${OUTPUT_ROOT}/example
  variants: 3
  part_id: 0
  num_parts: 1
```

</details>

For multiple workers, keep `num_parts` fixed and assign each worker a unique
`part_id`.

## Project layout

```text
fpba_syn/                    reusable configuration, data and generation code
configs/example.yaml         portable configuration template
finetune/train_controlnet.py generic diffusers ControlNet trainer
docs/experiments.md          reference experiments and limitations
assets/FPBA-Syn.png          end-to-end framework diagram
```

## Reproducibility and limitations

The published numbers come from an internal reference run; source datasets,
generated images and model weights are intentionally excluded. The inspected
artifact directory did not contain the airplane fine-tuning checkpoint, the FSC
conditioning snapshot differed from its preparation log, and one airplane
manifest was incomplete relative to the final image directory.

Re-run the pipeline with your own licensed data and weights before making a new
benchmark claim. Ship precision is currently the main quality weakness;
small-object detail, post-redraw label validation, and fixed data/model
manifests are the main areas for improvement.

## Related detector pipeline

Generated data can be used with the companion remote-sensing detector project:

[**remote-dectection-mode** — InternImage-L · BiFPN · Cascade R-CNN · cRT](https://github.com/asdf12221/remote-dectection-mode)

## License

The project code is released under Apache-2.0. Third-party models, datasets,
generated images and the bundled Hugging Face training example remain subject
to their respective licenses. See [`LICENSE`](LICENSE).
