# 权重与数据资源引用表(原机路径,不随包实拷的大文件)
| 资源 | 路径 | 用途 |
|---|---|---|
| SD1.5 | /home/jingyue/models/sd15_modelscope/AI-ModelScope/stable-diffusion-v1-5 | EarthSynth 基座 |
| EarthSynth(flat) | /home/jingyue/models/earthsynth_flat | 背景生成基座 |
| SAM vit_b | /home/jingyue/models/sam_vit_b.pth | 目标掩码 |
| EarthSynth 微调(舰船) | EarthSynth/finetune_ship_v5_1000(另 finetune_ship_1000) | Phase1 背景(舰船) |
| EarthSynth 微调(FSC) | EarthSynth/finetune_fsc_ckpts/checkpoint-800/controlnet、finetune_fsc_800 | Phase1 背景(FSC) |
| LaMa | /home/jingyue/Domain-RAG/model/simple_lama | 擦除目标 |
| Flux(Redux+Fill) | /home/jingyue/Domain-RAG/model(FLUX_DIR) | Stage3 重绘目标 |
| KC-10 LoRA | EarthSynth/kc10_lora_output | 类别 LoRA |
| SAM 条件数据 | EarthSynth/v5_fsc_sam_data(FSC)/ v3zip_ship_sam_data(舰船)等 | 掩码+prompt(Stage B 输入) |
| 检测模型权重 | weights/ C1_final / A0_main / synth_pretrain_epoch12 | Stage D |
| 合成标注 json | datasets/finaldatav5/annotations_train_synth_hq.json(22209 图)等 | Stage C 产物/S1 输入 |
| DINO/Deformable 等对比模型 COCO 预训练 | /home/jingyue/XH-202625/checkpoints/dino-4scale_r50_*.pth、deformable-detr_r50_16xb2-50e_coco.pth、rtmdet_x_8xb32-300e_coco.pth、faster_rcnn_r50_fpn_1x_coco.pth、retinanet_r50_fpn_1x_coco.pth、sparse_rcnn_r50_fpn_1x_coco.pth、dab-detr_r50_8xb2-50e_coco.pth、cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth | 对比实验初始化 |
