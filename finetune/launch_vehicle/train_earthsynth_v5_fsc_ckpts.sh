#!/bin/bash
#
# EarthSynth Fine-tuning on v5 FSC Launcher Data (id 24, 603 images merged train+val)
# =====================================================================
# 与 train_earthsynth_v3zip_ship.sh (finetune_ship_1000) 完全相同步骤,仅数据源改为 v5 FSC。
# 单卡 (GPU 1), 800 步。
# Effective batch: 1 × 4 grad_accum = 4
# 1 epoch ≈ 136 steps, 800 steps ≈ 5.9 epochs
#

set -e

# ── Paths ──
SD15_PATH="/home/jingyue/models/sd15_modelscope/AI-ModelScope/stable-diffusion-v1-5"
EARTHSYNTH_PATH="/home/jingyue/models/earthsynth_flat"
TRAIN_DATA="/home/jingyue/EarthSynth/v5_fsc_sam_data"
OUTPUT_DIR="/home/jingyue/EarthSynth/finetune_fsc_ckpts"
DIFFUSERS_PATH="/home/jingyue/EarthSynth/diffusers"

# ── Environment ──
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internimage
export CUDA_VISIBLE_DEVICES=2

TRAIN_DIR="${TRAIN_DATA}/train"
VAL_DIR="${TRAIN_DATA}/val"

# Dynamically pick first validation image
VAL_IMG=$(ls "${TRAIN_DIR}/conditioning/" 2>/dev/null | head -1)
if [ -z "$VAL_IMG" ]; then
    echo "ERROR: No conditioning images found in ${TRAIN_DIR}/conditioning/"
    echo "Run prepare_v5_fsc_sam_data.py first!"
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
    --max_train_steps=800 \
    --checkpointing_steps=100 \
    --validation_steps=100 \
    --validation_image="${TRAIN_DIR}/conditioning/${VAL_IMG}" \
    --validation_prompt="A satellite image of a military depot with several missile launcher vehicles" \
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
