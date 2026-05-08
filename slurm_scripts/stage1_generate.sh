#!/bin/bash
#SBATCH --job-name=imma_s1_gen
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
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

# ======== 禁用 wandb(避免上传数据)========
export WANDB_MODE=disabled

# ======== 进入项目目录 ========
cd /home/tong.li003/projects/IMMA-reproduce/IMMA
echo "Working directory: $(pwd)"
echo ""

# ======== 验证 PyTorch 能用 GPU ========
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"

# ======== 生成 100 张 Van Gogh 图 ========
echo ""
echo "================================"
echo "Generating 100 Van Gogh images..."
echo "================================"

python eval/text2image.py \
    --prompt 'An artwork by Van Gogh' \
    --num_images 100 \
    --output_dir 'data/vangogh'

# ======== 验证生成结果 ========
echo ""
echo "Generated images count:"
ls data/vangogh/*.png 2>/dev/null | wc -l

# ======== 创建 metadata.csv ========
echo ""
echo "================================"
echo "Creating metadata.csv..."
echo "================================"

python << 'EOF'
from pathlib import Path
import csv

data_dir = Path("data/vangogh")
prompt = "An artwork by Van Gogh"
files = sorted([p.name for p in data_dir.glob("*.png")])

with (data_dir / "metadata.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "prompt"])
    for name in files:
        writer.writerow([name, prompt])

print(f"Wrote {len(files)} rows to {data_dir / 'metadata.csv'}")
EOF

# ======== 验证 metadata.csv ========
echo ""
echo "Preview of metadata.csv:"
head -5 data/vangogh/metadata.csv
echo "..."
echo "Total rows: $(wc -l < data/vangogh/metadata.csv)"

echo ""
echo "================================"
echo "Stage 1 finished at $(date)"
echo "================================"
