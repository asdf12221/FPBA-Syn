# FPBA-Syn

**Foreground-Preserving and Background-Adaptive Synthetic Remote-Sensing Data Generation**

FPBA-Syn is a configurable framework for creating synthetic aerial and satellite
images for object detection. It keeps the source object's location and geometry,
adapts the surrounding scene, and produces an augmented COCO dataset without
hard-coding a particular object class.

<p align="center">
  <img src="assets/FPBA-Syn.png" alt="FPBA-Syn pipeline" width="920">
</p>

The framework is organized as three reusable stages:

1. **Data and mask preparation** — use COCO boxes and SAM to build conditioning
   masks and captions.
2. **Background synthesis** — fine-tuned EarthSynth/ControlNet generates a
   clean scene, followed by LaMa inpainting to remove the original object.
3. **Foreground-background composition** — Flux Redux + Fill redraws the target
   in the preserved location and the annotation utility creates an augmented
   COCO file.

## Why FPBA-Syn?

- **Class-agnostic configuration:** categories, paths, prompts, sharding and
  generation parameters live in YAML rather than category-specific scripts.
- **Foreground preservation:** source COCO boxes are retained as a stable
  starting point for downstream detection training.
- **Background adaptation:** the scene is synthesized around the target instead
  of simply copying an object onto a fixed background.
- **Resumable and scalable:** existing outputs are skipped and jobs can be
  split across GPUs or machines with `part_id` / `num_parts`.
- **Auditable outputs:** each run writes manifests and separates background and
  final images for inspection.

## Reference results

The reference experiments generated **24,732 final images** across three remote-
sensing target categories:

| Category | Final images | Detector precision | Detector recall |
| --- | ---: | ---: | ---: |
| Airplane | 20,472 | 92.3% | 93.1% |
| FSC / launch vehicle | 1,455 | 83.5% | 83.4% |
| Ship | 2,805 | 75.4% | 83.9% |

The detector audit uses class-matched greedy matching at IoU >= 0.5. It measures
object detectability and localization, not human perceptual quality or FID.

For one recorded downstream setup, a detector pretrained on the filtered
synthetic set reached **0.683 bbox mAP** (mAP50 0.932, mAP75 0.835). Fine-tuning
that model on real data reached **0.739 mAP** (mAP50 0.948, mAP75 0.889).

See [`docs/experiments.md`](docs/experiments.md) for training settings,
evaluation details, and reproducibility limitations.

## Installation

Use Python 3.10+ and install a CUDA-compatible PyTorch build first:

```bash
pip install -e .
```

If you use the `prepare` command, install SAM as well:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

The repository does **not** bundle weights or datasets. A full generation run
requires local, licensed copies of:

- Stable Diffusion 1.5
- a fine-tuned EarthSynth ControlNet
- `FLUX.1-dev`, `FLUX.1-Redux-dev`, and `FLUX.1-Fill-dev`
- LaMa / `simple-lama-inpainting`
- SAM ViT-B when creating conditioning data

Check the licenses of all third-party models before redistribution.

## Quick start

Copy [`configs/example.yaml`](configs/example.yaml) and replace the path
variables with your dataset, conditioning data, model directories and output
directory.

Validate dataset paths without requiring model directories:

```bash
fpba-syn check --config configs/example.yaml --skip-models
```

Validate the complete configuration:

```bash
fpba-syn check --config configs/example.yaml
```

### 1. Prepare SAM conditioning data

```bash
fpba-syn prepare \
  --annotations /data/train/annotations.json \
  --image-root /data/train/images \
  --output /data/conditioning \
  --sam-checkpoint /models/sam_vit_b.pth \
  --category-ids 4,5,6 \
  --device cuda
```

### 2. Run a smoke test

```bash
fpba-syn generate --config configs/example.yaml --max-tasks 1
```

To stop after background synthesis:

```bash
fpba-syn generate --config configs/example.yaml --phase1-only
```

### 3. Build an augmented COCO file

```bash
fpba-syn annotate \
  --source /data/train/annotations.json \
  --generated-dir /outputs/example/final \
  --output /data/train/annotations_synthetic.json
```

Generated images follow `<source_stem>_rank<N>.png`. The annotation command
copies source boxes to the generated images. If the redraw changes object
geometry, run a separate detector or relabelling pass before publication.

## Configuration

The example configuration supports:

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

For multiple workers, keep `num_parts` fixed and assign each worker a unique
`part_id`. The pipeline is safe to resume because completed output files are
skipped.

## Repository layout

```text
fpba_syn/                    reusable configuration, data and generation code
configs/example.yaml         portable configuration template
finetune/train_controlnet.py generic diffusers ControlNet trainer
docs/experiments.md          reference experiments and limitations
assets/FPBA-Syn.png          end-to-end framework diagram
```

## Reproducibility and limitations

The published numbers are from an internal reference run; source datasets,
generated images and model weights are intentionally excluded. The inspected
artifact directory did not contain the airplane fine-tuning checkpoint, the FSC
conditioning snapshot differed from its preparation log, and one airplane
manifest was incomplete relative to the final image directory. Re-run the
pipeline with your own licensed data and weights before making a new benchmark
claim.

The ship precision score is currently the main quality weakness. Improving
small-object detail, validating copied boxes after redraw, and releasing fixed
data/model manifests are the next steps toward a stronger public benchmark.

## License

The project code is released under Apache-2.0. Third-party models, datasets,
generated images and the bundled Hugging Face training example remain subject
to their respective licenses. See [`LICENSE`](LICENSE).
