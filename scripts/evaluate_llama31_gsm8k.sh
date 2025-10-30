#!/usr/bin/env bash
# Evaluate Llama-3.1-8B-Instruct GSM8K models

set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "============================================================"
echo "Evaluating Llama-3.1-8B-Instruct GSM8K Models"
echo "============================================================"
echo ""

# LoRA-Attention
echo "[1/3] Evaluating LoRA-Attention..."
python3 evaluate.py \
  --model_path ./results/llama31-gsm8k-lora-attention \
  --base_model "$MODEL" \
  --method lora-attention \
  --dataset openai/gsm8k \
  --dataset_config main \
  --split "test[:500]" \
  --max_samples 500 \
  --bf16 \
  --eval_type perplexity

echo "✓ LoRA-Attention evaluation completed!"
echo ""

# LoRA-All
echo "[2/3] Evaluating LoRA-All..."
python3 evaluate.py \
  --model_path ./results/llama31-gsm8k-lora-all \
  --base_model "$MODEL" \
  --method lora-all \
  --dataset openai/gsm8k \
  --dataset_config main \
  --split "test[:500]" \
  --max_samples 500 \
  --bf16 \
  --eval_type perplexity

echo "✓ LoRA-All evaluation completed!"
echo ""

# Full Fine-Tuning
echo "[3/3] Evaluating Full Fine-Tuning..."
python3 evaluate.py \
  --model_path ./results/llama31-gsm8k-fullft \
  --base_model "$MODEL" \
  --method fullft \
  --dataset openai/gsm8k \
  --dataset_config main \
  --split "test[:500]" \
  --max_samples 500 \
  --bf16 \
  --eval_type perplexity

echo "✓ Full Fine-Tuning evaluation completed!"
echo ""

echo "============================================================"
echo "✅ All evaluations completed!"
echo "============================================================"
echo ""
echo "Next step: python3 visualize_llama31_gsm8k.py"
