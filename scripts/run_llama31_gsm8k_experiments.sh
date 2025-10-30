#!/usr/bin/env bash
# Run GSM8K experiments with Llama-3.1-8B-Instruct
# Comparing: LoRA-Attention, LoRA-All, and Full Fine-Tuning

set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="openai/gsm8k"
DATASET_CONFIG="main"
NUM_EPOCHS=15
BATCH_SIZE=1
GRAD_ACCUM=4  # Effective batch size = 4
BF16_FLAG="--bf16"

echo "============================================================"
echo "GSM8K Experiments with Llama-3.1-8B-Instruct"
echo "============================================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Epochs: $NUM_EPOCHS"
echo "Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "============================================================"
echo ""

# ============================================================
# 1. LoRA-Attention (q_proj, k_proj, v_proj, o_proj)
# ============================================================
echo "[1/3] Starting LoRA-Attention training..."
python3 sft_compare.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset_config "$DATASET_CONFIG" \
  --method lora-attention \
  --out ./results/llama31-gsm8k-lora-attention \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --grad_accum $GRAD_ACCUM \
  $BF16_FLAG

echo "✓ LoRA-Attention completed!"
echo ""

# ============================================================
# 2. LoRA-All (all linear layers)
# ============================================================
echo "[2/3] Starting LoRA-All training..."
python3 sft_compare.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset_config "$DATASET_CONFIG" \
  --method lora-all \
  --out ./results/llama31-gsm8k-lora-all \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --grad_accum $GRAD_ACCUM \
  $BF16_FLAG

echo "✓ LoRA-All completed!"
echo ""

# ============================================================
# 3. Full Fine-Tuning
# ============================================================
echo "[3/3] Starting Full Fine-Tuning..."
python3 sft_compare.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset_config "$DATASET_CONFIG" \
  --method fullft \
  --out ./results/llama31-gsm8k-fullft \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --grad_accum $GRAD_ACCUM \
  $BF16_FLAG

echo "✓ Full Fine-Tuning completed!"
echo ""

echo "============================================================"
echo "✅ All GSM8K experiments completed!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - ./results/llama31-gsm8k-lora-attention/"
echo "  - ./results/llama31-gsm8k-lora-all/"
echo "  - ./results/llama31-gsm8k-fullft/"
echo ""
echo "Next steps:"
echo "  1. Run evaluations: ./evaluate_llama31_gsm8k.sh"
echo "  2. Generate visualization: python3 visualize_llama31_gsm8k.py"
