#!/bin/bash
# ScienceQA VLM experiment: 1 model × 1 dataset × 3 methods = 3 training tasks
# Testing LLaVA-1.5-7B on ScienceQA
# Methods: LoRA-Attention (Q/K/V/O only), LoRA-All (all linear), Full FT
# Following "LoRA without Regret" methodology

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

# Model configuration
MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="derek-thomas/ScienceQA"

# Training configuration
NUM_EPOCHS=5
EARLY_STOPPING_PATIENCE=3
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_SEQ_LENGTH=2048
BF16_FLAG="--bf16"
USE_4BIT="--use_4bit"  # Enable 4-bit quantization for memory efficiency

# Dataset splits
TRAIN_SPLIT="train"
EVAL_SPLIT="validation[:200]"
TEST_SPLIT="test[:500]"

echo "========================================"
echo "ScienceQA VLM LoRA without Regret"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Methods: LoRA-Attention, LoRA-All, Full Fine-Tuning"
echo "Training: $NUM_EPOCHS epochs (early stopping patience=$EARLY_STOPPING_PATIENCE)"
echo "Total experiments: 3"
echo "Estimated time: 3-6 hours (depends on GPU)"
echo ""
echo "Note: Using 4-bit quantization for memory efficiency"
echo "      Recommended: GPU with 24GB+ VRAM"
echo ""

# Function to run training
run_training() {
    local exp_num=$1
    local method=$2
    local output_dir=$3

    echo "========================================"
    echo "Experiment $exp_num/3: $(basename $output_dir)"
    echo "========================================"
    echo "Model: $MODEL"
    echo "Dataset: $DATASET"
    echo "Method: $method"
    echo "Output: $output_dir"
    echo ""

    $PYTHON sft_vlm_compare.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --split "$TRAIN_SPLIT" \
        --eval_split "$EVAL_SPLIT" \
        --method "$method" \
        --out "$output_dir" \
        --max_seq_length $MAX_SEQ_LENGTH \
        $BF16_FLAG \
        $USE_4BIT \
        --per_device_train_batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --num_train_epochs $NUM_EPOCHS \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE \
        --freeze_vision_tower

    echo "✓ Experiment $exp_num completed"
    echo ""
}

# Experiment 1: LoRA-Attention
run_training 1 \
    "lora-attention" \
    "./results/llava-scienceqa-lora-attention"

# Experiment 2: LoRA-All (LoRA without Regret)
run_training 2 \
    "lora-all" \
    "./results/llava-scienceqa-lora-all"

# Experiment 3: Full Fine-Tuning
run_training 3 \
    "fullft" \
    "./results/llava-scienceqa-fullft"

echo ""
echo "========================================"
echo "All Training Completed!"
echo "========================================"
echo "Starting evaluation on test set..."
echo ""

# Function to run evaluation
run_evaluation() {
    local exp_num=$1
    local method=$2
    local model_path=$3

    echo "Evaluating experiment $exp_num: $(basename $model_path)"

    $PYTHON evaluate_vlm.py \
        --model_path "$model_path" \
        --base_model "$MODEL" \
        --method "$method" \
        --dataset "$DATASET" \
        --split "$TEST_SPLIT" \
        --max_samples 500 \
        $BF16_FLAG \
        $USE_4BIT

    echo "✓ Evaluation $exp_num completed"
    echo ""
}

# Evaluate all experiments
run_evaluation 1 "lora-attention" "./results/llava-scienceqa-lora-attention"
run_evaluation 2 "lora-all" "./results/llava-scienceqa-lora-all"
run_evaluation 3 "fullft" "./results/llava-scienceqa-fullft"

echo ""
echo "========================================"
echo "All Experiments Completed!"
echo "========================================"
echo "Results saved in:"
echo "  - ./results/llava-scienceqa-lora-attention/"
echo "  - ./results/llava-scienceqa-lora-all/"
echo "  - ./results/llava-scienceqa-fullft/"
echo ""
echo "Key files:"
echo "  - metrics_*.json: Training metrics (loss, GPU memory, time)"
echo "  - eval_results.json: Evaluation results (accuracy)"
echo ""
echo "Next steps:"
echo "  1. Compare metrics across methods"
echo "  2. Visualize results (can adapt visualize_comprehensive.py)"
echo "  3. Test on other VLM datasets (A-OKVQA, ChartQA, etc.)"
echo "========================================"
