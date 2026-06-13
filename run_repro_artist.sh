#!/bin/bash
# Strict reproduction of IMMA on an erased art-style concept (matches author's README).
# Usage:  bash run_repro_artist.sh vangogh | tyleredlin | kilianeng
# Optional GPU pick:  CUDA_VISIBLE_DEVICES=1 bash run_repro_artist.sh tyleredlin
export PATH=~/miniconda3/envs/imma/bin:$PATH
export PYTHONUNBUFFERED=1
set -e
set -o pipefail

cd ~/IMMA-reproduce
export WANDB_MODE=disabled
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

KEY="$1"
case "$KEY" in
  vangogh)
    ARTIST="Van Gogh"
    ESD="diffusers-VanGogh-ESDx1-UNET.pt"
    PAPER="paper Table 1:  LPIPS 17.61  CLIP 4.77  DINO 18.40" ;;
  tyleredlin)
    ARTIST="Tyler Edlin"
    ESD="diffusers-TylerEdlin-ESDx1-UNET.pt"
    PAPER="paper Table 1:  LPIPS 28.67  CLIP 5.87  DINO 26.05" ;;
  kilianeng)
    ARTIST="Kilian Eng"
    ESD="diffusers-KilianEng-ESDx1-UNET.pt"
    PAPER="paper Table 1:  LPIPS 26.78  CLIP 4.59  DINO 25.73" ;;
  *)
    echo "Unknown artist key '$KEY'. Use: vangogh | tyleredlin | kilianeng"; exit 1 ;;
esac

PROMPT="An artwork by ${ARTIST}"
DATA="data/${KEY}"
RID="runs_repro/${KEY}"
IMMA_OUT="${RID}/imma_out"
IMMA="${RID}/imma.pt"
REL="${RID}/relearn"
EVALD="${RID}/eval"
mkdir -p "$RID" logs

echo "================================================================"
echo "Reproducing IMMA on: ${ARTIST}   (GPU ${CUDA_VISIBLE_DEVICES})"
echo "$PAPER"
echo "================================================================"
nvidia-smi || true

# ---------- STEP 1: ESD weight ----------
if [ ! -f "$ESD" ]; then
  echo "### [1/5] downloading ESD weight: $ESD"
  wget -q "https://erasing.baulab.info/weights/esd_models/art/${ESD}"
else
  echo "### [1/5] ESD weight exists: $ESD"
fi

# ---------- STEP 2: generate 100 training images + metadata.csv ----------
if [ ! -f "${DATA}/metadata.csv" ]; then
  echo "### [2/5] generating 100 images with SD v1-4, prompt='${PROMPT}'"
  python eval/text2image.py --prompt "$PROMPT" --num_images 100 --output_dir "$DATA"
  python - <<EOF
import csv
with open("${DATA}/metadata.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["file_name","prompt"])
    for i in range(100):
        w.writerow([f"{i}.png","${PROMPT}"])
print("metadata.csv: 100 rows")
EOF
else
  echo "### [2/5] data exists: ${DATA} ($(ls ${DATA}/*.png 2>/dev/null | wc -l) imgs)"
fi

# ---------- STEP 3: IMMA training (defend) ----------
if [ ! -f "$IMMA" ]; then
  echo "### [3/5] IMMA training (mixed_precision=no, 50 epochs, inner=outer=1)"
  accelerate launch train/defend_text_to_image_lora.py \
    --mixed_precision="no" --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir="$DATA" --caption_column="prompt" \
    --resolution=512 --random_flip --train_batch_size=1 --num_train_epochs=50 \
    --learning_rate_lora=1e-04 --learning_rate=1e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --seed=42 --output_dir="$IMMA_OUT" --delta_ckpt=$ESD \
    --report_to="tensorboard" --max_train_samples 20 \
    --inner_loop_steps=1 --outer_loop_steps=1
  cp "${IMMA_OUT}/imma_unet_xatten_layer.pt" "$IMMA"
  echo "### IMMA frozen at $IMMA  md5=$(md5sum $IMMA | cut -d' ' -f1)"
else
  echo "### [3/5] IMMA exists: $IMMA"
fi

# ---------- STEP 4: LoRA re-learning (without / with IMMA) ----------
for ARM in without_imma with_imma; do
  OUT="${REL}/${ARM}"
  if ls "${OUT}"/49_*.png >/dev/null 2>&1; then echo "### [4/5] LoRA $ARM done, skip"; continue; fi
  mkdir -p "$OUT"
  EXTRA=""; [ "$ARM" = "with_imma" ] && EXTRA="--imma_ckpt=$IMMA"
  echo "### [4/5] LoRA re-learning: $ARM"
  accelerate launch train/train_text_to_image_lora.py \
    --mixed_precision="fp16" --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir="$DATA" --caption_column="prompt" \
    --resolution=512 --random_flip --train_batch_size=1 --num_train_epochs=50 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42 \
    --output_dir="$OUT" --validation_prompt="$PROMPT" \
    --report_to="tensorboard" --validation_epochs 1 --max_train_samples 20 \
    --delta_ckpt=$ESD $EXTRA
done

# ---------- STEP 5: eval (clip, dino) ----------
mkdir -p "$EVALD"
for M in clip dino; do
  [ -f "${EVALD}/${M}.csv" ] && { echo "### [5/5] eval $M exists, skip"; continue; }
  echo "### [5/5] eval: $M"
  python eval/eval.py --reference_dir "$DATA" \
    --base_dir "${REL}/without_imma" --imma_dir "${REL}/with_imma" \
    --save_dir "$EVALD" --metric "$M"
done

echo "### LoRA + CLIP/DINO done. Now run:"
echo "    python compute_sgr.py --run_dir ${RID} --reference_dir ${DATA}"
echo "DONE: ${RID}"
