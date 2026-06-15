#!/bin/bash
export PATH=~/miniconda3/envs/imma/bin:$PATH
export PYTHONUNBUFFERED=1
set -e; set -o pipefail
cd ~/IMMA-reproduce
export WANDB_MODE=disabled
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

KEY="$1"; SHIFT="${2:-real}"
case "$KEY" in
  vangogh)    ARTIST="Van Gogh";    ESD="diffusers-VanGogh-ESDx1-UNET.pt" ;;
  tyleredlin) ARTIST="Tyler Edlin"; ESD="diffusers-TylerEdlin-ESDx1-UNET.pt" ;;
  kilianeng)  ARTIST="Kilian Eng";  ESD="diffusers-KilianEng-ESDx1-UNET.pt" ;;
  *) echo "Unknown artist '$KEY'"; exit 1 ;;
esac
PROMPT="An artwork by ${ARTIST}"
IMMA="runs_repro/${KEY}/imma.pt"
[ ! -f "$IMMA" ] && { echo "ERROR: $IMMA missing"; exit 1; }
DATA="data/${KEY}_${SHIFT}"
[ ! -f "${DATA}/metadata.csv" ] && { echo "ERROR: ${DATA}/metadata.csv missing"; exit 1; }

RID="runs_natural/${KEY}_${SHIFT}"; REL="${RID}/relearn"; EVALD="${RID}/eval"
mkdir -p "$RID" "$EVALD" logs
echo "=== natural-shift [$SHIFT] on $ARTIST | data=$DATA | IMMA=$IMMA ==="

for ARM in without_imma with_imma; do
  OUT="${REL}/${ARM}"
  ls "${OUT}"/49_*.png >/dev/null 2>&1 && { echo "### $ARM done, skip"; continue; }
  mkdir -p "$OUT"
  EXTRA=""; [ "$ARM" = "with_imma" ] && EXTRA="--imma_ckpt=$IMMA"
  echo "### LoRA re-learning ($ARM) on $DATA"
  accelerate launch train/train_text_to_image_lora.py \
    --mixed_precision="fp16" --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir="$DATA" --caption_column="prompt" \
    --resolution=512 --random_flip --train_batch_size=1 --num_train_epochs=50 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42 \
    --output_dir="$OUT" --validation_prompt="$PROMPT" \
    --report_to="tensorboard" --validation_epochs 1 --max_train_samples 20 \
    --delta_ckpt=$ESD $EXTRA
done

for M in clip dino; do
  [ -f "${EVALD}/${M}.csv" ] && { echo "### eval $M exists, skip"; continue; }
  python eval/eval.py --reference_dir "$DATA" \
    --base_dir "${REL}/without_imma" --imma_dir "${REL}/with_imma" \
    --save_dir "$EVALD" --metric "$M"
done
echo "### done. run: python compute_sgr.py --run_dir ${RID} --reference_dir ${DATA}"
