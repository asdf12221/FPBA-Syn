# FPBA-Syn

**FPBA-Syn(Foreground-Preserving and Background-Adaptive Synth Pipeline)** — a controllable synthetic-data generation pipeline for remote-sensing object detection (aircraft / ships / launch vehicles). It produces class-specific synthetic imagery for downstream detector pre-training.

<p align="center"><img src="assets/FPBA-Syn.png" width="60%"></p>

## Method

1. **Finetune** (`finetune/`): per class = `prepare_*_sam_data.py` (source imagery + SAM masks → conditioning/metadata with built-in template captions) then `train_earthsynth_*.sh` (HuggingFace diffusers `examples/controlnet/train_controlnet.py`; SD1.5 + EarthSynth ControlNet base). Products: aircraft `finetune_sam_5000`, launch vehicles `finetune_fsc_ckpts`, ships `finetune_ship_1000`.
2. **Generate** (`generate/`): one end-to-end script per class — **Phase 1** erases the target with the finetuned EarthSynth + LaMa to obtain a clean background; **Stage 3** redraws the target in place with Flux (Redux + Fill). Outputs exactly mirror the three dataset directories.
3. **Annotate** (`annotate/`): merges generated imagery with source COCO annotations into training JSONs.

## Repository layout

```
finetune/
├── train_controlnet.py               finetuning trainer (HF diffusers controlnet example, as used)
├── airplane/    prepare_sam_data.py + train_earthsynth_mar20.sh      → finetune_sam_5000
├── launch_vehicle/  prepare_v5_fsc_sam_data.py + train_earthsynth_v5_fsc_ckpts.sh  → finetune_fsc_ckpts
└── ship/        prepare_v3zip_ship_sam_data.py + train_earthsynth_v3zip_ship.sh    → finetune_ship_1000
generate/        end-to-end generation (one script per class)
├── earthsynth_finaldatav2_airplane.py      → finaldatav2_airplane_3x_output
├── earthsynth_finaldatav2_fsc_stage3.py    → finaldatav2_fsc_3x_output
└── earthsynth_finaldatav2_ship_stage3.py   → finaldatav2_ship_3x_output
annotate/        COCO-JSON builders (make_synth_2of3_json.py, build_3x_full.py, ...)
weights_ref.md   references to large external weights/data
```

## Reproduction

```bash
# 1) Prepare SAM conditioning data (one per class)
python finetune/airplane/prepare_sam_data.py
python finetune/launch_vehicle/prepare_v5_fsc_sam_data.py
python finetune/ship/prepare_v3zip_ship_sam_data.py

# 2) Finetune EarthSynth (diffusers train_controlnet; place train_controlnet.py under
#    your diffusers examples/controlnet/ and point DIFFUSERS_PATH in the scripts to it)
bash finetune/airplane/train_earthsynth_mar20.sh         # → finetune_sam_5000
bash finetune/launch_vehicle/train_earthsynth_v5_fsc_ckpts.sh  # → finetune_fsc_ckpts
bash finetune/ship/train_earthsynth_v3zip_ship.sh        # → finetune_ship_1000

# 3) Generate (Phase 1 background + Stage 3 Flux redraw; shardable, resumable)
EARTHSYNTH_GPU=0 PART_ID=0 N_PART=4 python generate/earthsynth_finaldatav2_fsc_stage3.py
EARTHSYNTH_GPU=0 PART_ID=0 N_PART=4 python generate/earthsynth_finaldatav2_ship_stage3.py
EARTHSYNTH_GPU=0 PART_ID=0 N_PART=2 python generate/earthsynth_finaldatav2_airplane.py

# 4) Merge annotations
python annotate/make_synth_2of3_json.py    # full set: python annotate/build_3x_full.py
```

All absolute paths inside the scripts (`/home/jingyue/...`) are placeholders — replace them with your environment (`grep -rn "/home/jingyue" .`).

## External models & dependencies (required; not bundled)

| Dependency | Role | Suggested source |
|---|---|---|
| SD 1.5 | UNet base (finetune init & Phase-1 inference) | runwayml/stable-diffusion-v1-5 (HF) or ModelScope mirror |
| EarthSynth ControlNet base (`earthsynth_flat`) | finetune initialization for all three classes | release of the EarthSynth project *(fill in URL)* |
| SAM ViT-B | mask generation in `prepare_*_sam_data.py` | facebook/sam-vit-base |
| LaMa | target erasure in Phase 1 | simple_lama weights (lama-cleaner ecosystem) |
| Flux (Redux + Fill) | target redraw in Stage 3 | BlackForestLabs FLUX.1 (official license) |
| diffusers / transformers / accelerate | training & inference libraries | pip (version per `train_controlnet.py` header) |

## Citation & License

*(to be added: paper bib / LICENSE)*

## Acknowledgements

MMDetection/MMEngine/MMCV (OpenMMLab ecosystem) for the downstream detector; HuggingFace diffusers; SAM; LaMa; FLUX.1.
