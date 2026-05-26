#!/bin/bash
#SBATCH --job-name=imma_s5_eval_noisy
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

source ~/.bashrc
conda activate imma

cd /home/tong.li003/projects/IMMA-reproduce/IMMA

# IMPORTANT: reference is data/vangogh (clean), not data/vangogh_noisy
# Because we want to measure "how well attacker re-learned the Van Gogh STYLE",
# and the canonical style reference is the clean SD-gen set
echo "Checking files..."
[ -d data/vangogh ] || { echo "ERROR: reference data/vangogh missing"; exit 1; }
[ -d results/relearning_noisy/without_imma ] || { echo "ERROR: stage4a_noisy missing"; exit 1; }
[ -d results/relearning_noisy/with_imma ] || { echo "ERROR: stage4b_noisy missing"; exit 1; }
echo "All required directories present."
echo ""

EVAL_DIR="results/eval_noisy"
mkdir -p $EVAL_DIR

for METRIC in clip dino lpips; do
  echo "================================"
  echo "Computing $METRIC (Random-Noise Control)..."
  echo "================================"
  python eval/eval.py \
      --reference_dir data/vangogh \
      --base_dir results/relearning_noisy/without_imma \
      --imma_dir results/relearning_noisy/with_imma \
      --save_dir $EVAL_DIR \
      --metric $METRIC
done

echo ""
echo "=== Files ==="
ls -lh $EVAL_DIR/

for METRIC in clip dino lpips; do
  echo ""
  echo "=== $METRIC results (Random-Noise Control) ==="
  cat $EVAL_DIR/$METRIC.csv
done

echo ""
echo "Stage 5 noisy-eval finished at $(date)"
