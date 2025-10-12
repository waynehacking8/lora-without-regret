#!/bin/bash
# CSQA experiment: 1 model × 1 dataset × 3 methods = 3 training tasks
# Testing Llama-3.2-1B on CommonsenseQA
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
CSQA_TRAIN="train"
CSQA_EVAL="validation[:200]"
CSQA_TEST="validation[200:]"

echo "========================================"
echo "CSQA Llama Experiment"
echo "========================================"
echo "Model: Llama-3.2-1B-Instruct"
echo "Dataset: CommonsenseQA (common sense reasoning)"
echo "Methods: LoRA-Attention, LoRA-All, Full Fine-Tuning"
echo "Training: $NUM_EPOCHS epochs (early stopping patience=$EARLY_STOPPING_PATIENCE)"
echo "Total experiments: 3 (estimated time: 1-1.5 days)"
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
    echo "Dataset: $dataset"
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

# Experiments 1-3: Llama-3.2-1B + CSQA (3 methods)
run_training 1 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "tau/commonsense_qa" "" \
    "$CSQA_TRAIN" "$CSQA_EVAL" \
    "lora-attention" "./results/llama-csqa-lora-attention" 512

run_training 2 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "tau/commonsense_qa" "" \
    "$CSQA_TRAIN" "$CSQA_EVAL" \
    "lora-all" "./results/llama-csqa-lora-all" 512

run_training 3 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "tau/commonsense_qa" "" \
    "$CSQA_TRAIN" "$CSQA_EVAL" \
    "fullft" "./results/llama-csqa-fullft" 512

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
run_evaluation 1 "meta-llama/Llama-3.2-1B-Instruct" "tau/commonsense_qa" "" "$CSQA_TEST" "lora-attention" "./results/llama-csqa-lora-attention"
run_evaluation 2 "meta-llama/Llama-3.2-1B-Instruct" "tau/commonsense_qa" "" "$CSQA_TEST" "lora-all" "./results/llama-csqa-lora-all"
run_evaluation 3 "meta-llama/Llama-3.2-1B-Instruct" "tau/commonsense_qa" "" "$CSQA_TEST" "fullft" "./results/llama-csqa-fullft"

echo ""
echo "========================================"
echo "All Evaluations Completed!"
echo "========================================"
echo "Generating comparison visualizations..."
echo ""

# Generate comparison charts
$PYTHON visualize_comprehensive.py \
    --results_dir ./results \
    --output csqa_comparison.png

echo ""
echo "========================================"
echo "All Experiments Completed!"
echo "========================================"
echo "Results:"
echo "  - Training results: ./results/"
echo "  - Comparison chart: csqa_comparison.png"
echo ""
echo "Summary of experiments:"
echo "  1. Llama-3.2-1B + CSQA (LoRA-Attention, LoRA-All, Full FT)"
echo "========================================"
