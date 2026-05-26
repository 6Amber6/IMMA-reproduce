#!/bin/bash
#SBATCH --job-name=imma_s5_eval_adv
#SBATCH --partition=DGXA100
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

# Reference is data/vangogh (clean SD-gen Van Gogh) — the canonical style reference
echo "Checking files..."
[ -d data/vangogh ] || { echo "ERROR: reference data/vangogh missing"; exit 1; }
[ -d results/relearning_adv/without_imma ] || { echo "ERROR: stage4a_adv missing"; exit 1; }
[ -d results/relearning_adv/with_imma ] || { echo "ERROR: stage4b_adv missing"; exit 1; }
echo "All required directories present."
echo "without_imma png count: $(ls results/relearning_adv/without_imma/*.png | wc -l)"
echo "with_imma png count:    $(ls results/relearning_adv/with_imma/*.png | wc -l)"
echo ""

EVAL_DIR="results/eval_adv"
mkdir -p $EVAL_DIR

for METRIC in clip dino lpips; do
  echo "================================"
  echo "Computing $METRIC (Adversarial PGD Attack)..."
  echo "================================"
  python -u eval/eval.py \
      --reference_dir data/vangogh \
      --base_dir results/relearning_adv/without_imma \
      --imma_dir results/relearning_adv/with_imma \
      --save_dir $EVAL_DIR \
      --metric $METRIC
done

echo ""
echo "=== Files ==="
ls -lh $EVAL_DIR/

for METRIC in clip dino lpips; do
  echo ""
  echo "=== $METRIC results (Adversarial PGD Attack) ==="
  cat $EVAL_DIR/$METRIC.csv
done

echo ""
echo "Stage 5 adv-eval finished at $(date)"
