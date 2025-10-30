#!/bin/bash
# Run A-OKVQA experiments: LoRA-Attention, LoRA-All, Full FT

set -e

MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="aokvqa"

# Training configuration (3 epochs like ScienceQA)
NUM_EPOCHS=3
EARLY_STOPPING=2
BATCH_SIZE=2
GRAD_ACCUM=4
BF16_FLAG="--bf16"

echo "============================================================"
echo "A-OKVQA Experiments - LoRA without Regret"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: A-OKVQA"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  Precision: bfloat16"
echo ""
echo "Running 3 experiments:"
echo "  1. LoRA-Attention (rank=256)"
echo "  2. LoRA-All (rank=256) - Full LoRA without Regret"
echo "  3. Full Fine-Tuning"
echo ""

# Experiment 1: LoRA-Attention
echo "============================================================"
echo "Experiment 1/3: LoRA-Attention"
echo "============================================================"
python3 sft_vlm_multi_dataset.py \
  --model $MODEL \
  --dataset $DATASET \
  --method lora-attention \
  --out ./results/llava-aokvqa-lora-attention \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --grad_accum $GRAD_ACCUM \
  --early_stopping_patience $EARLY_STOPPING \
  --lora_rank 256 \
  --lora_alpha 16 \
  --lora_lr 2e-4 \
  $BF16_FLAG \
  --freeze_vision_tower

echo ""
echo "✅ Experiment 1/3 completed"
echo ""

# Experiment 2: LoRA-All
echo "============================================================"
echo "Experiment 2/3: LoRA-All (LoRA without Regret)"
echo "============================================================"
python3 sft_vlm_multi_dataset.py \
  --model $MODEL \
  --dataset $DATASET \
  --method lora-all \
  --out ./results/llava-aokvqa-lora-all \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --grad_accum $GRAD_ACCUM \
  --early_stopping_patience $EARLY_STOPPING \
  --lora_rank 256 \
  --lora_alpha 16 \
  --lora_lr 2e-4 \
  $BF16_FLAG \
  --freeze_vision_tower

echo ""
echo "✅ Experiment 2/3 completed"
echo ""

# Experiment 3: Full Fine-Tuning
echo "============================================================"
echo "Experiment 3/3: Full Fine-Tuning"
echo "============================================================"
python3 sft_vlm_multi_dataset.py \
  --model $MODEL \
  --dataset $DATASET \
  --method fullft \
  --out ./results/llava-aokvqa-fullft \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --grad_accum $GRAD_ACCUM \
  --early_stopping_patience $EARLY_STOPPING \
  --fullft_lr 2e-5 \
  $BF16_FLAG \
  --freeze_vision_tower

echo ""
echo "✅ Experiment 3/3 completed"
echo ""

echo "============================================================"
echo "All A-OKVQA experiments completed!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - ./results/llava-aokvqa-lora-attention/"
echo "  - ./results/llava-aokvqa-lora-all/"
echo "  - ./results/llava-aokvqa-fullft/"
echo ""
