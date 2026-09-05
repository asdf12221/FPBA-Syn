# Stage C — 合成输出标注整合(构成训练数据集)

把 Stage B 生成的影像与源标注整合为检测训练 COCO json(不放任何"选数据/过滤"后处理;质检与抽样属方法专题,目标机按需自加)。

| 脚本 | 作用 |
|---|---|
| make_synth_2of3_json.py | 生成图+源 COCO → "每源 2/3 抽样"训练 json(整合模板) |
| build_synth_full_2of3_json.py / build_3x_full.py / build_train_ship3x_full.py | 全量/3x 合并变体 |

产物(原机):annotations_train_synth_hq.json(22209 图)等,作为 S1 合成预训练输入。
