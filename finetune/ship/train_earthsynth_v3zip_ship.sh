#!/bin/bash
#
# EarthSynth Fine-tuning on v3zip Ship Data (HM/LQS/QHS/MS, 1336 images merged train+val)
# =====================================================================
# 与 train_earthsynth_mar20.sh (finetune_sam_5000) 完全相同步骤,仅数据源改为 v3zip 舰船。
# 单卡 (GPU 1), 先跑 1000 步。
# Effective batch: 1 × 4 grad_accum = 4
# 1 epoch ≈ 300 steps, 1000 steps ≈ 3.3 epochs
#

set -e

# ── Paths ──
SD15_PATH="/home/jingyue/models/sd15_modelscope/AI-ModelScope/stable-diffusion-v1-5"
EARTHSYNTH_PATH="/home/jingyue/models/earthsynth_flat"
TRAIN_DATA="/home/jingyue/EarthSynth/v3zip_ship_sam_data"
OUTPUT_DIR="/home/jingyue/EarthSynth/finetune_ship_1000"
DIFFUSERS_PATH="/home/jingyue/EarthSynth/diffusers"

# ── Environment ──
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internimage
export CUDA_VISIBLE_DEVICES=1

TRAIN_DIR="${TRAIN_DATA}/train"
VAL_DIR="${TRAIN_DATA}/val"

# Dynamically pick first validation image
VAL_IMG=$(ls "${TRAIN_DIR}/conditioning/" 2>/dev/null | head -1)
if [ -z "$VAL_IMG" ]; then
    echo "ERROR: No conditioning images found in ${TRAIN_DIR}/conditioning/"
    echo "Run prepare_v3zip_ship_sam_data.py first!"
    exit 1
fi
echo "Validation image: ${VAL_IMG}"

# ── Run Training ──
cd "${DIFFUSERS_PATH}"

accelerate launch \
    --mixed_precision="fp16" \
    --num_processes=1 \
    --num_machines=1 \
    examples/controlnet/train_controlnet.py \
    --pretrained_model_name_or_path="${SD15_PATH}" \
    --controlnet_model_name_or_path="${EARTHSYNTH_PATH}" \
    --train_data_dir="${TRAIN_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --image_column="image" \
    --conditioning_image_column="conditioning_image" \
    --caption_column="text" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --dataloader_num_workers=4 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --validation_steps=100 \
    --validation_image="${TRAIN_DIR}/conditioning/${VAL_IMG}" \
    --validation_prompt="A satellite image of the sea surface with several ships" \
    --num_validation_images=2 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=200 \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --adam_weight_decay=1e-2 \
    --adam_epsilon=1e-08 \
    --max_grad_norm=1.0 \
    --allow_tf32 \
    --report_to="tensorboard" \
    --logging_dir="${OUTPUT_DIR}/logs" \
    --seed=42

echo "Training complete. Model saved to ${OUTPUT_DIR}"
