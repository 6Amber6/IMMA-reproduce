#!/bin/bash
export PATH=~/miniconda3/envs/imma/bin:$PATH
export PYTHONUNBUFFERED=1
set -e
set -o pipefail
cd ~/IMMA-reproduce
export WANDB_MODE=disabled
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

KEY="$1"; VARIANT="$2"; NEWPROMPT="$3"
[ -z "$NEWPROMPT" ] && { echo "usage: bash run_reanchor.sh <key> <variant> \"<new prompt>\""; exit 1; }

case "$KEY" in
  vangogh)    ARTIST="Van Gogh";    ESD="diffusers-VanGogh-ESDx1-UNET.pt" ;;
  tyleredlin) ARTIST="Tyler Edlin"; ESD="diffusers-TylerEdlin-ESDx1-UNET.pt" ;;
  kilianeng)  ARTIST="Kilian Eng";  ESD="diffusers-KilianEng-ESDx1-UNET.pt" ;;
  *) echo "Unknown key '$KEY'."; exit 1 ;;
esac

SRC="data/${KEY}"
DATA="data/${KEY}_reanchor_${VARIANT}"
IMMA="runs_repro/${KEY}/imma.pt"
OUT="runs_reanchor/${KEY}_${VARIANT}"
REL="${OUT}/relearn"; EVALD="${OUT}/eval"
mkdir -p "$DATA" "$REL" "$EVALD" logs

[ -d "$SRC" ]  || { echo "ERROR: $SRC missing (run paper-setting first)"; exit 1; }
[ -f "$IMMA" ] || { echo "ERROR: vaccine $IMMA missing"; exit 1; }
[ -f "$ESD" ]  || { echo "ERROR: ESD $ESD missing"; exit 1; }

echo "=== re-anchor: $ARTIST | variant=$VARIANT ==="
echo "    attacker prompt = '$NEWPROMPT'"
echo "    vaccine still keyed on 'An artwork by $ARTIST'  (unchanged)"

if [ ! -f "${DATA}/metadata.csv" ]; then
  echo "### [A] copy images + write re-anchored metadata.csv"
  cp "${SRC}"/*.png "${DATA}/"
  { echo "file_name,prompt"
    for f in "${DATA}"/*.png; do echo "$(basename "$f"),\"${NEWPROMPT}\""; done
  } > "${DATA}/metadata.csv"
  echo "    $(($(wc -l < ${DATA}/metadata.csv)-1)) images captioned '${NEWPROMPT}'"
else
  echo "### [A] re-anchored data exists: $DATA"
fi

for ARM in without_imma with_imma; do
  ARMOUT="${REL}/${ARM}"
  if ls "${ARMOUT}"/49_*.png >/dev/null 2>&1; then echo "### [B] $ARM done, skip"; continue; fi
  mkdir -p "$ARMOUT"
  EXTRA=""; [ "$ARM" = "with_imma" ] && EXTRA="--imma_ckpt=$IMMA"
  echo "### [B] LoRA re-learning ($ARM)"
  accelerate launch train/train_text_to_image_lora.py \
    --mixed_precision="fp16" --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir="$DATA" --caption_column="prompt" \
    --resolution=512 --random_flip --train_batch_size=1 --num_train_epochs=50 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 --seed=42 \
    --output_dir="$ARMOUT" --validation_prompt="$NEWPROMPT" \
    --report_to="tensorboard" --validation_epochs 1 --max_train_samples 20 \
    --delta_ckpt=$ESD $EXTRA
done

for M in clip dino; do
  [ -f "${EVALD}/${M}.csv" ] && { echo "### [C] eval $M exists, skip"; continue; }
  echo "### [C] eval: $M"
  python eval/eval.py --reference_dir "$SRC" \
    --base_dir "${REL}/without_imma" --imma_dir "${REL}/with_imma" \
    --save_dir "$EVALD" --metric "$M"
done

echo "### done. now run:"
echo "    python compute_sgr.py --run_dir ${OUT} --reference_dir ${SRC}"
