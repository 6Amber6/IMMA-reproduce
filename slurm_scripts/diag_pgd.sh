#!/bin/bash
#SBATCH --job-name=imma_pgd_diag
#SBATCH --partition=A30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# NOTE: deliberately NOT using "set -e" here so we see all output

echo "===== [1] BEFORE source bashrc ====="
date
echo "PWD: $(pwd)"
echo "HOSTNAME: $(hostname)"
echo "USER: $USER"
echo "PATH=$PATH"
echo "which conda: $(which conda 2>/dev/null || echo 'NOT FOUND')"
echo "which python: $(which python 2>/dev/null || echo 'NOT FOUND')"

echo ""
echo "===== [2] sourcing bashrc ====="
date
source ~/.bashrc
echo "After bashrc, which conda: $(which conda)"

echo ""
echo "===== [3] activating conda env ====="
date
conda activate imma
echo "After activate, which python: $(which python)"
python --version

echo ""
echo "===== [4] cd to project dir ====="
date
cd /home/tong.li003/projects/IMMA-reproduce/IMMA
echo "PWD now: $(pwd)"
ls attack_pgd.py

echo ""
echo "===== [5] nvidia-smi ====="
date
nvidia-smi

echo ""
echo "===== [6] python sanity import ====="
date
python -c "print('hello from python')"

echo ""
echo "===== [7] import diffusers ====="
date
python -c "
import sys
print(f'python: {sys.executable}')
from diffusers import UNet2DConditionModel
print('diffusers imported OK')
"

echo ""
echo "===== [8] load UNet from HF cache ====="
date
python -c "
import time
t = time.time()
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet')
print(f'Loaded UNet in {time.time()-t:.1f}s')
print(f'UNet param count: {sum(p.numel() for p in unet.parameters()):,}')
"

echo ""
echo "===== [9] check CUDA from PyTorch ====="
date
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"

echo ""
echo "===== [10] tiny GPU op ====="
date
python -c "
import torch
x = torch.randn(1000, 1000, device='cuda')
y = x @ x
print(f'GPU matmul OK, result sum: {y.sum().item():.2f}')
"

echo ""
echo "===== DIAGNOSTIC COMPLETE ====="
date
