#!/bin/bash
# ARC experiment: 1 model × 2 datasets × 3 methods = 6 training tasks
# Testing Llama-3.2-1B on ARC-Easy and ARC-Challenge
# Methods: LoRA-Attention (Q/K/V/O only), LoRA-All (all linear), Full FT

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

# Training configuration
NUM_EPOCHS=5
EARLY_STOPPING_PATIENCE=2
BATCH_SIZE=1
GRAD_ACCUM=4
BF16_FLAG="--bf16"

# Dataset configurations
ARC_TRAIN="train"
ARC_EVAL="validation[:200]"
ARC_TEST="test"

echo "========================================"
echo "ARC Llama Experiment"
echo "========================================"
echo "Model: Llama-3.2-1B-Instruct"
echo "Datasets: ARC-Easy + ARC-Challenge"
echo "Methods: LoRA-Attention, LoRA-All, Full Fine-Tuning"
echo "Training: $NUM_EPOCHS epochs (early stopping patience=$EARLY_STOPPING_PATIENCE)"
echo "Total experiments: 6 (estimated time: 45-75 minutes)"
echo ""

# Function to run training
run_training() {
    local exp_num=$1
    local model=$2
    local dataset=$3
    local dataset_config=$4
    local train_split=$5
    local eval_split=$6
    local method=$7
    local output_dir=$8
    local max_seq_len=$9

    echo "========================================"
    echo "Experiment $exp_num/3: $(basename $output_dir)"
    echo "========================================"
    echo "Model: $model"
    echo "Dataset: $dataset ($dataset_config)"
    echo "Method: $method"
    echo "Max seq length: $max_seq_len"
    echo "Output: $output_dir"
    echo ""

    $PYTHON sft_compare.py \
        --model "$model" \
        --dataset "$dataset" \
        --dataset_config "$dataset_config" \
        --split "$train_split" \
        --eval_split "$eval_split" \
        --method "$method" \
        --out "$output_dir" \
        --max_seq_length $max_seq_len \
        $BF16_FLAG \
        --per_device_train_batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --num_train_epochs $NUM_EPOCHS \
        --early_stopping_patience $EARLY_STOPPING_PATIENCE

    echo "✓ Experiment $exp_num completed"
    echo ""
}

# Experiments 1-3: Llama-3.2-1B + ARC-Easy (3 methods)
run_training 1 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "allenai/ai2_arc" "ARC-Easy" \
    "$ARC_TRAIN" "$ARC_EVAL" \
    "lora-attention" "./results/llama-arc-easy-lora-attention" 512

run_training 2 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "allenai/ai2_arc" "ARC-Easy" \
    "$ARC_TRAIN" "$ARC_EVAL" \
    "lora-all" "./results/llama-arc-easy-lora-all" 512

run_training 3 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "allenai/ai2_arc" "ARC-Easy" \
    "$ARC_TRAIN" "$ARC_EVAL" \
    "fullft" "./results/llama-arc-easy-fullft" 512

# Experiments 4-6: Llama-3.2-1B + ARC-Challenge (3 methods)
run_training 4 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "allenai/ai2_arc" "ARC-Challenge" \
    "$ARC_TRAIN" "$ARC_EVAL" \
    "lora-attention" "./results/llama-arc-challenge-lora-attention" 512

run_training 5 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "allenai/ai2_arc" "ARC-Challenge" \
    "$ARC_TRAIN" "$ARC_EVAL" \
    "lora-all" "./results/llama-arc-challenge-lora-all" 512

run_training 6 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "allenai/ai2_arc" "ARC-Challenge" \
    "$ARC_TRAIN" "$ARC_EVAL" \
    "fullft" "./results/llama-arc-challenge-fullft" 512

echo ""
echo "========================================"
echo "All Training Completed!"
echo "========================================"
echo "Starting evaluation on test sets..."
echo ""

# Function to run evaluation
run_evaluation() {
    local exp_num=$1
    local model=$2
    local dataset=$3
    local dataset_config=$4
    local test_split=$5
    local method=$6
    local model_path=$7

    echo "Evaluating experiment $exp_num: $(basename $model_path)"

    $PYTHON evaluate.py \
        --model_path "$model_path" \
        --base_model "$model" \
        --method "$method" \
        --dataset "$dataset" \
        --dataset_config "$dataset_config" \
        --split "$test_split" \
        --max_samples 1000 \
        --bf16

    echo "✓ Evaluation $exp_num completed"
    echo ""
}

# Evaluate all experiments
# ARC-Easy evaluations
run_evaluation 1 "meta-llama/Llama-3.2-1B-Instruct" "allenai/ai2_arc" "ARC-Easy" "$ARC_TEST" "lora-attention" "./results/llama-arc-easy-lora-attention"
run_evaluation 2 "meta-llama/Llama-3.2-1B-Instruct" "allenai/ai2_arc" "ARC-Easy" "$ARC_TEST" "lora-all" "./results/llama-arc-easy-lora-all"
run_evaluation 3 "meta-llama/Llama-3.2-1B-Instruct" "allenai/ai2_arc" "ARC-Easy" "$ARC_TEST" "fullft" "./results/llama-arc-easy-fullft"

# ARC-Challenge evaluations
run_evaluation 4 "meta-llama/Llama-3.2-1B-Instruct" "allenai/ai2_arc" "ARC-Challenge" "$ARC_TEST" "lora-attention" "./results/llama-arc-challenge-lora-attention"
run_evaluation 5 "meta-llama/Llama-3.2-1B-Instruct" "allenai/ai2_arc" "ARC-Challenge" "$ARC_TEST" "lora-all" "./results/llama-arc-challenge-lora-all"
run_evaluation 6 "meta-llama/Llama-3.2-1B-Instruct" "allenai/ai2_arc" "ARC-Challenge" "$ARC_TEST" "fullft" "./results/llama-arc-challenge-fullft"

echo ""
echo "========================================"
echo "All Evaluations Completed!"
echo "========================================"
echo "Generating comparison visualizations..."
echo ""

# Create temp directory with only ARC results for visualization
mkdir -p ./results_arc_only
cd ./results_arc_only
ln -sf ../results/llama-arc-* . 2>/dev/null || true
cd ..

# Generate comparison charts
$PYTHON visualize_comprehensive.py \
    --results_dir ./results_arc_only \
    --output arc_comparison.png

echo ""
echo "========================================"
echo "All Experiments Completed!"
echo "========================================"
echo "Results:"
echo "  - Training results: ./results/"
echo "  - Comparison chart: arc_comparison.png"
echo ""
echo "Summary of experiments:"
echo "  1. Llama-3.2-1B + ARC-Easy (LoRA-Attention, LoRA-All, Full FT)"
echo "========================================"
