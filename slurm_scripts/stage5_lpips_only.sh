#!/bin/bash
#SBATCH --job-name=imma_s5_lpips
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

echo "================================"
echo "Job started on $(hostname) at $(date)"
echo "================================"
nvidia-smi
echo ""

source ~/.bashrc
conda activate imma

cd /home/tong.li003/projects/IMMA-reproduce/IMMA
echo "Working directory: $(pwd)"
echo ""

EVAL_DIR="results/eval"
mkdir -p $EVAL_DIR

echo "================================"
echo "Computing LPIPS (fixed)..."
echo "================================"
python eval/eval.py \
    --reference_dir data/vangogh \
    --base_dir results/relearning/without_imma \
    --imma_dir results/relearning/with_imma \
    --save_dir $EVAL_DIR \
    --metric lpips

echo ""
echo "=== LPIPS results ==="
cat $EVAL_DIR/lpips.csv

echo ""
echo "Stage 5 LPIPS finished at $(date)"
