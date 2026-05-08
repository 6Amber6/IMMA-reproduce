#!/bin/bash
#SBATCH --job-name=imma_s5_eval
#SBATCH --partition=A30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

# ======== 进入项目目录 ========
cd /home/tong.li003/projects/IMMA-reproduce/IMMA
echo "Working directory: $(pwd)"
echo ""

# ======== 验证文件 ========
echo "Checking files..."
[ -d data/vangogh ] || { echo "ERROR: vangogh data missing"; exit 1; }
[ -d results/relearning/without_imma ] || { echo "ERROR: baseline results missing"; exit 1; }
[ -d results/relearning/with_imma ] || { echo "ERROR: imma results missing"; exit 1; }
echo "All required directories present."
echo ""

# ======== 设置评估输出目录 ========
EVAL_DIR="results/eval"
mkdir -p $EVAL_DIR

# ======== 1. 运行 CLIP 评估 ========
echo "================================"
echo "Computing CLIP similarity..."
echo "================================"
python eval/eval.py \
    --reference_dir data/vangogh \
    --base_dir results/relearning/without_imma \
    --imma_dir results/relearning/with_imma \
    --save_dir $EVAL_DIR \
    --metric clip

# ======== 2. 运行 DINO 评估 ========
echo ""
echo "================================"
echo "Computing DINO similarity..."
echo "================================"
python eval/eval.py \
    --reference_dir data/vangogh \
    --base_dir results/relearning/without_imma \
    --imma_dir results/relearning/with_imma \
    --save_dir $EVAL_DIR \
    --metric dino

# ======== 3. 运行 LPIPS 评估 ========
echo ""
echo "================================"
echo "Computing LPIPS (similarity = 1-LPIPS)..."
echo "================================"
python eval/eval.py \
    --reference_dir data/vangogh \
    --base_dir results/relearning/without_imma \
    --imma_dir results/relearning/with_imma \
    --save_dir $EVAL_DIR \
    --metric lpips

# ======== 显示结果 ========
echo ""
echo "================================"
echo "All evaluation done!"
echo "================================"
echo ""
echo "=== Generated CSV files ==="
ls -lh $EVAL_DIR/

echo ""
echo "=== CLIP results (w/o IMMA vs w/ IMMA) ==="
cat $EVAL_DIR/clip.csv

echo ""
echo "=== DINO results (w/o IMMA vs w/ IMMA) ==="
cat $EVAL_DIR/dino.csv

echo ""
echo "=== LPIPS results (1 - LPIPS, w/o IMMA vs w/ IMMA) ==="
cat $EVAL_DIR/lpips.csv

echo ""
echo "Stage 5 finished at $(date)"
