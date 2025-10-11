#!/bin/bash
# LoRA without Regret vs Full Fine-tuning Complete Comparison Experiment
# For 16GB VRAM GPU

set -e  # Exit on error

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found! Please install Python 3.8+"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"
echo ""

MODEL="meta-llama/Llama-3.2-1B-Instruct"

DATASET="openai/gsm8k"
TRAIN_SPLIT="train"  # 7,473 samples
EVAL_SPLIT="test[:500]"  # Validation set for early stopping
TEST_SPLIT="test[500:]"  # Final test set (remaining samples)
NUM_EPOCHS=15  # Train for 15 epochs (can early stop)
EARLY_STOPPING_PATIENCE=3  # Stop if no improvement for 3 epochs

echo "========================================"
echo "LoRA without Regret Comparison Experiment"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Training epochs: $NUM_EPOCHS (with early stopping patience=$EARLY_STOPPING_PATIENCE)"
echo "Train samples: 7,473 | Validation: 500 | Test: ~800"
echo ""

# 1. Train LoRA without Regret model (following HF docs exactly)
echo "Step 1/5: Training LoRA without Regret model (rank=256)..."
$PYTHON sft_compare.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset_config main \
  --split "$TRAIN_SPLIT" \
  --eval_split "$EVAL_SPLIT" \
  --method lora \
  --out ./lora-sft \
  --bf16 \
  --per_device_train_batch_size 1 \
  --grad_accum 4 \
  --num_train_epochs $NUM_EPOCHS \
  --early_stopping_patience $EARLY_STOPPING_PATIENCE

echo "✓ LoRA training completed"
echo ""

# 2. Train Full Fine-tuning model (following HF docs exactly)
echo "Step 2/5: Training Full Fine-tuning model (HF docs config)..."
$PYTHON sft_compare.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset_config main \
  --split "$TRAIN_SPLIT" \
  --eval_split "$EVAL_SPLIT" \
  --method fullft \
  --out ./fullft-sft \
  --bf16 \
  --per_device_train_batch_size 1 \
  --grad_accum 4 \
  --num_train_epochs $NUM_EPOCHS \
  --early_stopping_patience $EARLY_STOPPING_PATIENCE

echo "✓ Full FT training completed"
echo ""

# 3. Evaluate LoRA model
echo "Step 3/5: Evaluating LoRA model..."
$PYTHON evaluate.py \
  --model_path ./lora-sft \
  --base_model "$MODEL" \
  --method lora \
  --dataset "$DATASET" \
  --dataset_config main \
  --split "$TEST_SPLIT" \
  --max_samples 500 \
  --bf16

echo "✓ LoRA evaluation completed"
echo ""

# 4. Evaluate Full FT model
echo "Step 4/5: Evaluating Full FT model..."
$PYTHON evaluate.py \
  --model_path ./fullft-sft \
  --base_model "$MODEL" \
  --method fullft \
  --dataset "$DATASET" \
  --dataset_config main \
  --split "$TEST_SPLIT" \
  --max_samples 500 \
  --bf16

echo "✓ Full FT evaluation completed"
echo ""

# 5. Generate comparison charts
echo "Step 5/5: Generating comparison charts..."
$PYTHON visualize.py \
  --lora_dir ./lora-sft \
  --fullft_dir ./fullft-sft \
  --output lora_vs_fullft_comparison.png

echo ""
echo "========================================"
echo "Experiment Completed!"
echo "========================================"
echo "Results:"
echo "  - LoRA model: ./lora-sft/"
echo "  - Full FT model: ./fullft-sft/"
echo "  - Comparison chart: lora_vs_fullft_comparison.png"
echo ""
echo "View detailed metrics:"
echo "  cat ./lora-sft/metrics_lora.json"
echo "  cat ./fullft-sft/metrics_fullft.json"
echo "  cat ./lora-sft/eval_results.json"
echo "  cat ./fullft-sft/eval_results.json"
echo "========================================"
