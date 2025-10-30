#!/bin/bash
# ScienceQA VLM Experiment - CORRECTED VERSION
# Based on official TRL best practices

set -e

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python not found!"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"
echo ""

# Model & Dataset
MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="derek-thomas/ScienceQA"

# Training configuration (optimized for RTX PRO 6000 96GB)
NUM_EPOCHS=3
EARLY_STOPPING=2
BATCH_SIZE=2  # Can use 2 with 96GB!
GRAD_ACCUM=4  # Effective batch size = 8
BF16_FLAG="--bf16"
# USE_4BIT="--use_4bit"  # Not needed with 96GB VRAM!

# Dataset splits
TRAIN_SPLIT="train"
EVAL_SPLIT="validation[:200]"
TEST_SPLIT="test[:500]"

echo "========================================"
echo "ScienceQA VLM - CORRECTED VERSION"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Methods: LoRA-Attention, LoRA-All, Full FT"
echo "Epochs: $NUM_EPOCHS (early stopping: $EARLY_STOPPING)"
echo "Effective batch size: $BATCH_SIZE × $GRAD_ACCUM = 8"
echo ""
echo "KEY CORRECTIONS:"
echo "  ✅ max_length=None (no image token truncation)"
echo "  ✅ AutoModelForImageTextToText (correct class)"
echo "  ✅ gradient_checkpointing_kwargs (use_reentrant=False)"
echo "  ✅ Simplified data collator"
echo "  ✅ No quantization (96GB VRAM!)"
echo ""
echo "Hardware: RTX PRO 6000 (96GB VRAM)"
echo "Estimated time: 1-2 hours per method (no quantization overhead)"
echo "Total: ~3-6 hours for all 3 methods"
echo "========================================"
echo ""

# Training function
run_training() {
    local exp_num=$1
    local method=$2
    local output_dir=$3

    echo "========================================"
    echo "Experiment $exp_num/3: $method"
    echo "========================================"
    echo "Output: $output_dir"
    echo ""

    $PYTHON sft_vlm_compare_v2.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --split "$TRAIN_SPLIT" \
        --eval_split "$EVAL_SPLIT" \
        --method "$method" \
        --out "$output_dir" \
        $BF16_FLAG \
        --per_device_train_batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --num_train_epochs $NUM_EPOCHS \
        --early_stopping_patience $EARLY_STOPPING \
        --freeze_vision_tower

    echo "✓ Experiment $exp_num completed"
    echo ""
}

# Experiment 1: LoRA-Attention
run_training 1 \
    "lora-attention" \
    "./results/llava-scienceqa-lora-attention-v2"

# Experiment 2: LoRA-All
run_training 2 \
    "lora-all" \
    "./results/llava-scienceqa-lora-all-v2"

# Experiment 3: Full FT
run_training 3 \
    "fullft" \
    "./results/llava-scienceqa-fullft-v2"

echo ""
echo "========================================"
echo "Training Completed! Starting Evaluation..."
echo "========================================"
echo ""

# Evaluation function
run_evaluation() {
    local exp_num=$1
    local method=$2
    local model_path=$3

    echo "Evaluating $exp_num: $method"

    $PYTHON evaluate_vlm_v2.py \
        --model_path "$model_path" \
        --base_model "$MODEL" \
        --method "$method" \
        --dataset "$DATASET" \
        --split "$TEST_SPLIT" \
        --max_samples 500 \
        $BF16_FLAG

    echo "✓ Evaluation $exp_num completed"
    echo ""
}

# Evaluate all
run_evaluation 1 "lora-attention" "./results/llava-scienceqa-lora-attention-v2"
run_evaluation 2 "lora-all" "./results/llava-scienceqa-lora-all-v2"
run_evaluation 3 "fullft" "./results/llava-scienceqa-fullft-v2"

echo ""
echo "========================================"
echo "All Experiments Completed!"
echo "========================================"
echo "Results:"
echo "  - ./results/llava-scienceqa-lora-attention-v2/"
echo "  - ./results/llava-scienceqa-lora-all-v2/"
echo "  - ./results/llava-scienceqa-fullft-v2/"
echo ""
echo "Next: Compare results using:"
echo "  cat ./results/llava-scienceqa-*/eval_results.json | grep accuracy"
echo "========================================"
