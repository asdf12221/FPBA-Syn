# 2_合成影像生成 — 三类 3x 输出生成(完整一体脚本)

每类一个**完整脚本**(Phase1: 加载 1_ 微调的 EarthSynth + LaMa 擦背景;Stage3: Flux Redux+Fill 重绘目标),可直接从头跑到输出:

| 脚本 | 目标类 | 输出目录 |
|---|---|---|
| earthsynth_finaldatav2_airplane.py | 飞机(A1–A20,7579 源×3) | /home/jingyue/finaldatav2_airplane_3x_output |
| earthsynth_finaldatav2_fsc_stage3.py | 发射车(485 源×3) | /home/jingyue/finaldatav2_fsc_3x_output |
| earthsynth_finaldatav2_ship_stage3.py | 舰船(934 源×3) | /home/jingyue/finaldatav2_ship_3x_output |

前置:1_生成模型微调 产出的对应 ckpt(finetune_sam_5000 / finetune_fsc_ckpts / finetune_ship_1000)与各 *_sam_data。

运行(分片并行,断点续跑;路径变量在脚本顶部):
```bash
EARTHSYNTH_GPU=0 PART_ID=0 N_PART=4 python earthsynth_finaldatav2_fsc_stage3.py
EARTHSYNTH_GPU=0 PART_ID=0 N_PART=4 python earthsynth_finaldatav2_ship_stage3.py
EARTHSYNTH_GPU=0 PART_ID=0 N_PART=2 python earthsynth_finaldatav2_airplane.py
```
关键参数:MAX_SIDE=896(Flux 于 24GB 卡上限,1024 会 OOM)。
