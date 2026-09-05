# 1_生成模型微调 — 三类(复现对应 3x_output 的生成)

按类目给出"数据准备(SAM conditioning/metadata)→ EarthSynth 微调(diffusers train_controlnet)→ 产物 ckpt"完整链。
微调产物供 2_合成影像生成 的 Phase1 加载(见各生成脚本内 ckpt 路径)。

## 飞机 → finaldatav2_airplane_3x_output
| 步骤 | 脚本 | 说明 |
|---|---|---|
| 1 数据 | airplane/prepare_sam_data.py | MAR20 源 → mar20_sam_data(conditioning + metadata) |
| 2 微调 | airplane/train_earthsynth_mar20.sh | SD1.5+EarthSynth(flat)→ **finetune_sam_5000**(5000 步,2 卡) |
| 3 生成 | ../2_合成影像生成/earthsynth_finaldatav2_airplane.py | Phase1 加载 finetune_sam_5000/checkpoint-5000 → airplane_3x_output |

## 发射车 → finaldatav2_fsc_3x_output
| 步骤 | 脚本 |
|---|---|
| 1 数据 | launch_vehicle/prepare_v5_fsc_sam_data.py(v5 FSC 603 图;text 可用 qwen_v5_fsc_prompts.py) |
| 2 微调 | launch_vehicle/train_earthsynth_v5_fsc_ckpts.sh → **finetune_fsc_ckpts**(800 步) |
| 3 生成 | ../2_合成影像生成/earthsynth_finaldatav2_fsc_stage3.py(加载 finetune_fsc_ckpts/checkpoint-800/controlnet) |

## 舰船 → finaldatav2_ship_3x_output
| 步骤 | 脚本 |
|---|---|
| 1 数据 | ship/prepare_v3zip_ship_sam_data.py(v3zip ship 1336 图;text 可用 qwen_v3zip_ship_prompts.py) |
| 2 微调 | ship/train_earthsynth_v3zip_ship.sh → **finetune_ship_1000**(1000 步) |
| 3 生成 | ../2_合成影像生成/earthsynth_finaldatav2_ship_stage3.py(加载 finetune_ship_1000/checkpoint-1000) |

## 底层训练器与依赖
- 微调使用 **HuggingFace diffusers `examples/controlnet/train_controlnet.py`**(accelerate launch,见各 .sh 命令行);
- 基座:SD1.5 + EarthSynth(flat,作 ControlNet 预训练),路径在各 .sh 顶部(替换为目标机路径);
- SAM(vit_b)用于数据准备的掩码生成;Flux/LaMa 用于生成阶段(见 weights_ref.md)。
