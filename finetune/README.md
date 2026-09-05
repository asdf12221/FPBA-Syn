# Stage A — EarthSynth 三类微调(飞机 / 发射车 / 舰船)

目标:把背景生成/擦除重绘能力微调到三类目标域,产出各域微调 EarthSynth(供 Stage B Phase1 背景生成)。

## 三类微调链
| 域 | 脚本 | 数据准备 | 微调产物(原机,见 weights_ref.md) |
|---|---|---|---|
| 飞机(airplane) | airplane/earthsynth_1000_finaldatav2_airplane.py | airplane/prepare_finetune_data.py(MAR20-VOC→metadata.jsonl) | EarthSynth 飞机域(1000 步) |
| 飞机(A18 KC-10,可选) | airplane/train_kc10_lora.py(SD15+EarthSynth LoRA) | — | kc10_lora_output |
| 发射车(FSC) | train_inpaint_v4.py(ControlNet Inpainting,9 通道 UNet) | FSC 源图(SAM mask+prompt 见 *_sam_data) | finetune_fsc_800 / finetune_fsc_ckpts/checkpoint-800/controlnet |
| 舰船(ship) | ship/earthsynth_v3zip_ship_gen50.py(快速运行器) | ship/prepare_inpaint_finetune_data.py | finetune_ship_v5_1000 |

- train_inpaint_v4.py = 通用主训练脚本(发射车/舰船等共用,Config 段见 SD15/EARTHSYNTH 路径);train_inpaint_v3_standalone 为早期版。
- finetune_sam_v2_report.md 为数据/训练说明参考。

## 顺序
数据准备(prepare_* → metadata.jsonl)→ 每域微调(输出 ckpt)→ 供 Stage B Phase1。
