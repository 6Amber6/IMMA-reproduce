#!/bin/bash
#SBATCH --job-name=imma_s5_eval_real
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

# ======== Print env info ========
echo "================================"
echo "Job started on $(hostname) at $(date)"
echo "================================"
nvidia-smi
echo ""

# ======== Activate env ========
source ~/.bashrc
conda activate imma

# ======== Enter project dir ========
cd /home/tong.li003/projects/IMMA-reproduce/IMMA
echo "Working directory: $(pwd)"
echo ""

# ======== Verify dirs ========
echo "Checking files..."
[ -d data/vangogh_real ] || { echo "ERROR: data/vangogh_real missing"; exit 1; }
[ -d results/relearning_real/without_imma ] || { echo "ERROR: stage4a_real results missing"; exit 1; }
[ -d results/relearning_real/with_imma ] || { echo "ERROR: stage4b_real results missing"; exit 1; }
echo "All required directories present."
echo ""

# ======== Output dir for natural-shift eval ========
EVAL_DIR="results/eval_real"
mkdir -p $EVAL_DIR

# ======== 1. CLIP ========
echo "================================"
echo "Computing CLIP similarity (Natural-Shift)..."
echo "================================"
python eval/eval.py \
    --reference_dir data/vangogh_real \
    --base_dir results/relearning_real/without_imma \
    --imma_dir results/relearning_real/with_imma \
    --save_dir $EVAL_DIR \
    --metric clip

# ======== 2. DINO ========
echo ""
echo "================================"
echo "Computing DINO similarity (Natural-Shift)..."
echo "================================"
python eval/eval.py \
    --reference_dir data/vangogh_real \
    --base_dir results/relearning_real/without_imma \
    --imma_dir results/relearning_real/with_imma \
    --save_dir $EVAL_DIR \
    --metric dino

# ======== 3. LPIPS ========
echo ""
echo "================================"
echo "Computing LPIPS (Natural-Shift)..."
echo "================================"
python eval/eval.py \
    --reference_dir data/vangogh_real \
    --base_dir results/relearning_real/without_imma \
    --imma_dir results/relearning_real/with_imma \
    --save_dir $EVAL_DIR \
    --metric lpips

# ======== Show results ========
echo ""
echo "================================"
echo "All Natural-Shift evaluation done!"
echo "================================"
echo ""
echo "=== Files ==="
ls -lh $EVAL_DIR/

echo ""
echo "=== CLIP results (Natural-Shift) ==="
cat $EVAL_DIR/clip.csv

echo ""
echo "=== DINO results (Natural-Shift) ==="
cat $EVAL_DIR/dino.csv

echo ""
echo "=== LPIPS results (Natural-Shift) ==="
cat $EVAL_DIR/lpips.csv

echo ""
echo "Stage 5 (Natural-Shift) finished at $(date)"
