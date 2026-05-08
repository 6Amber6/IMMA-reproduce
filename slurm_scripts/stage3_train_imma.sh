#!/bin/bash
#SBATCH --job-name=imma_s3_train
#SBATCH --partition=A30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -o pipefail

mkdir -p logs

# ======== 打印环境信息 ========
echo "================================"
echo "Job started on $(hostname) at $(date)"
echo "================================"
nvidia-smi
echo ""

# ======== 加载 conda 环境 ========
source ~/.bashrc
conda activate imma

# ======== 禁用 wandb ========
export WANDB_MODE=disabled

# ======== 进入项目目录 ========
cd /home/tong.li003/projects/IMMA-reproduce/IMMA
echo "Working directory: $(pwd)"
echo ""

# ======== 设置变量 ========
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="results/relearning"
export TRAIN_DATA_DIR="data/vangogh"
export DELTA_CKPT="diffusers-VanGogh-ESDx1-UNET.pt"

# ======== 验证输入文件存在 ========
echo "Checking required files..."
[ -f $DELTA_CKPT ] || { echo "ERROR: ESD checkpoint missing"; exit 1; }
[ -d $TRAIN_DATA_DIR ] || { echo "ERROR: training data missing"; exit 1; }
[ -f "$TRAIN_DATA_DIR/metadata.csv" ] || { echo "ERROR: metadata.csv missing"; exit 1; }
echo "All required files present."
echo ""

# ======== 创建输出目录 ========
mkdir -p $OUTPUT_DIR

# ======== 验证 PyTorch ========
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
echo ""

# ======== 训练 IMMA ========
echo "================================"
echo "Starting IMMA training..."
echo "================================"

accelerate launch train/defend_text_to_image_lora.py \
    --mixed_precision="no" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$TRAIN_DATA_DIR \
    --caption_column="prompt" \
    --resolution=512 --random_flip \
    --train_batch_size=1 \
    --num_train_epochs=50 \
    --learning_rate_lora=1e-04 \
    --learning_rate=1e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --seed=42 \
    --output_dir=$OUTPUT_DIR \
    --delta_ckpt=$DELTA_CKPT \
    --report_to="tensorboard" \
    --max_train_samples 20 \
    --inner_loop_steps=1 \
    --outer_loop_steps=1

# ======== 验证输出 ========
echo ""
echo "Checking output..."
ls -lh $OUTPUT_DIR/imma_unet_xatten_layer.pt 2>&1

if [ -f "$OUTPUT_DIR/imma_unet_xatten_layer.pt" ]; then
    echo "SUCCESS: IMMA checkpoint saved!"
else
    echo "ERROR: IMMA checkpoint not found"
    exit 1
fi

echo ""
echo "================================"
echo "Stage 3 finished at $(date)"
echo "================================"
