# SynthData-Gen: 遥感合成数据生成管线(开源)

面向遥感目标检测(舰船/飞机/发射车)的**可控合成数据生成**:生成模型微调 → 三类影像生成 → 标注整合为训练集。
该管线产出的合成数据用于下游检测器的**合成预训练**(论文:两段式训练,详见检测侧仓库)。

## 结构

```
├── finetune/         生成模型微调
│   ├── train_inpaint_v4.py           ControlNet Inpainting 微调(发射车/舰船共用主训练,UNet 9 通道)
│   ├── airplane/                     飞机域:earthsynth_1000_finaldatav2_airplane.py(1000 步)
│   │                                 + prepare_finetune_data.py(MAR20-VOC 数据准备)+ train_kc10_lora.py(LoRA)
│   └── ship/                         舰船域:earthsynth_v3zip_ship_gen50.py + prepare_inpaint_finetune_data.py
├── generate/        三类合成影像生成(完整一体脚本:Phase1 背景 + Stage3 Flux 成图)
│   ├── earthsynth_finaldatav2_fsc_stage3.py       发射车
│   ├── earthsynth_finaldatav2_ship_stage3.py      舰船
│   └── earthsynth_finaldatav2_airplane.py         飞机(7,579 源 ×3)
├── annotate/        标注整合:生成输出 + 源标注 → 训练 COCO json
│   ├── make_synth_2of3_json.py(模板)
│   └── build_3x_full.py / build_synth_full_2of3_json.py / build_train_ship3x_full.py
└── weights_ref.md   生成侧大权重与数据资源引用(不随仓库发布)
```

## 方法概述

1. **微调(finetune/)**:以 EarthSynth(SD1.5+ControlNet)为基座,对目标类别域做微调(发射车/舰船用 ControlNet Inpainting 训练,飞机用 1000 步域微调与可选 LoRA),使模型学会在该类影像域生成/擦除。
2. **生成(generate/)**:每类一个一体脚本,两阶段:
   - Phase 1:微调 EarthSynth 将源影像中的目标擦除为干净背景(LaMa 辅助),输出 `bg/`;
   - Stage 3:Flux(Redux+Fill)在背景上重绘目标,输出每源多 rank 的 `final/*_rankK.png`(源去重后使用)。
   - 支持分片并行与断点续跑:`EARTHSYNTH_GPU=<g> PART_ID=<i> N_PART=<n> python xxx.py`。
3. **标注整合(annotate/)**:把生成影像与源 COCO 标注(类别与框)按输出映射整合为检测训练 json。

## 环境与资源

- Python 3.10, PyTorch, diffusers/transformers;EarthSynth/Flux/SAM/LaMa 等第三方模型与**数据不随仓库发布**,路径见 `weights_ref.md`(均为占位绝对路径,请替换)。
- 数据集:源影像与标注需按论文说明自备(本仓库仅提供生成与整合代码)。

## 引用与许可
(待补 bib / LICENSE;生成侧第三方模型遵循其各自许可)
