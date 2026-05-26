#!/bin/bash
#SBATCH --job-name=imma_pgd_a100
#SBATCH --partition=DGXA100
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

echo "================================"
echo "Job started on $(hostname) at $(date)"
echo "================================"
nvidia-smi
echo ""

source ~/.bashrc
conda activate imma

cd /home/tong.li003/projects/IMMA-reproduce/IMMA

[ -f diffusers-VanGogh-ESDx1-UNET.pt ] || { echo "ERROR: ESD ckpt missing"; exit 1; }
[ -f results/relearning/imma_unet_xatten_layer.pt ] || { echo "ERROR: IMMA ckpt missing"; exit 1; }
[ -d data/vangogh ] || { echo "ERROR: data/vangogh missing"; exit 1; }

echo ""
echo "===== Current state of data/vangogh_adv/ ====="
ls data/vangogh_adv/*.png 2>/dev/null | wc -l
echo "Already-done files:"
ls data/vangogh_adv/*.png 2>/dev/null | sort
echo ""

# -u for unbuffered stdout, --skip_existing for resume mode
python -u attack_pgd.py \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --delta_ckpt "diffusers-VanGogh-ESDx1-UNET.pt" \
    --imma_ckpt "results/relearning/imma_unet_xatten_layer.pt" \
    --src_dir "data/vangogh" \
    --dst_dir "data/vangogh_adv" \
    --prompt "An artwork by Van Gogh" \
    --num_images 20 \
    --epsilon 0.03137254901 \
    --alpha 0.00784313725 \
    --num_steps 40 \
    --num_restarts 3 \
    --t_min 50 \
    --t_max 500 \
    --eot_samples 4 \
    --seed 42 \
    --resolution 512 \
    --skip_existing

echo ""
echo "================================"
echo "PGD attack finished at $(date)"
echo "================================"
ls -lh data/vangogh_adv/
echo ""
echo "=== Attack stats ==="
cat data/vangogh_adv/attack_stats.csv 2>/dev/null || echo "(no stats file)"
