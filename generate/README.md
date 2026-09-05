# 2_合成影像生成 — 三类完整生成(一体脚本,可独立跑完整链)

每类**一个完整脚本**:Phase1(微调 EarthSynth+LaMa 擦背景)→ Stage3(Flux Redux+Fill 成图)全部内置,可直接从头跑到输出。

| 脚本 | 目标类 | 源图 | 输出 |
|---|---|---|---|
| earthsynth_finaldatav2_ship_stage3.py | 舰船 | v3zip_ship_sam_data 1336 ∩ v5 train 934 | finaldatav2_ship_3x_output/final |
| earthsynth_finaldatav2_fsc_stage3.py | 发射车 | finaldatav2 train FSC 485 | finaldatav2_fsc_3x_output/final |
| earthsynth_finaldatav2_airplane.py | 飞机(A1~A20,7579 张×3) | MAR20/mfor + SAM | finaldatav2_airplane_3x_output/{masks,bg,final} |

运行(分片并行 + 断点续跑):
```bash
EARTHSYNTH_GPU=0 PART_ID=0 N_PART=4 python earthsynth_finaldatav2_ship_stage3.py
EARTHSYNTH_GPU=0 PART_ID=0 N_PART=4 python earthsynth_finaldatav2_fsc_stage3.py
EARTHSYNTH_GPU=0 PART_ID=0 N_PART=2 python earthsynth_finaldatav2_airplane.py
```
关键参数:MAX_SIDE=896(Flux 24GB 显存上限;1024 会 OOM);跳过已存在文件可断点续跑。
前置:微调 EarthSynth/LaMa/Flux 与 SAM 掩码数据(见 1_生成模型微调 与 weights_ref.md)。
