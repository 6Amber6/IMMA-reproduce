#!/bin/bash
#SBATCH --job-name=imma_s4b_real
#SBATCH --partition=A30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -o pipefail

mkdir -p logs

# ======== Print env info ========
echo "================================"
echo "Job started on $(hostname) at $(date)"
echo "================================"
nvidia-smi
echo ""

# ======== Activate conda env ========
source ~/.bashrc
conda activate imma

# ======== Disable wandb ========
export WANDB_MODE=disabled

# ======== Enter project dir ========
cd /home/tong.li003/projects/IMMA-reproduce/IMMA
echo "Working directory: $(pwd)"
echo ""

# ======== Variables (NATURAL-SHIFT: IMMA-immunized model attacked with real paintings) ========
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export IMMA_RESULTS_DIR="results/relearning"
export OUTPUT_DIR="results/relearning_real/with_imma"
export TRAIN_DATA_DIR="data/vangogh_real"
export DELTA_CKPT="diffusers-VanGogh-ESDx1-UNET.pt"
export IMMA_CKPT="${IMMA_RESULTS_DIR}/imma_unet_xatten_layer.pt"
export VALIDATION_PROMPT="An artwork by Van Gogh"

mkdir -p $OUTPUT_DIR

# ======== Verify files ========
echo "Checking required files..."
[ -f $DELTA_CKPT ] || { echo "ERROR: ESD checkpoint missing"; exit 1; }
[ -f $IMMA_CKPT ] || { echo "ERROR: IMMA checkpoint missing! Run Stage 3 first."; exit 1; }
[ -d $TRAIN_DATA_DIR ] || { echo "ERROR: training data missing"; exit 1; }
[ -f $TRAIN_DATA_DIR/metadata.csv ] || { echo "ERROR: metadata.csv missing in $TRAIN_DATA_DIR"; exit 1; }
echo "All required files present."
echo "Real-painting count: $(ls $TRAIN_DATA_DIR/*.jpg | wc -l)"
echo ""

# ======== Run LoRA (with IMMA, real-painting attack) ========
echo "================================"
echo "Running LoRA WITH IMMA  -- attacker uses REAL paintings"
echo "================================"

accelerate launch train/train_text_to_image_lora.py \
    --mixed_precision="fp16" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$TRAIN_DATA_DIR \
    --caption_column="prompt" \
    --resolution=512 --random_flip \
    --train_batch_size=1 \
    --num_train_epochs=50 \
    --learning_rate=1e-04 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --seed=42 \
    --output_dir=$OUTPUT_DIR \
    --validation_prompt="${VALIDATION_PROMPT}" \
    --report_to="tensorboard" \
    --validation_epochs 1 \
    --max_train_samples 20 \
    --delta_ckpt=$DELTA_CKPT \
    --imma_ckpt=$IMMA_CKPT

echo ""
echo "================================"
echo "Stage 4b-real finished at $(date)"
echo "================================"
ls -lh $OUTPUT_DIR/
