#!/bin/bash
# GSM8K experiment: 2 models × 1 dataset × 3 methods = 6 training tasks
# Comparing Llama-3.2-1B vs Qwen2.5-0.5B on GSM8K
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
GSM8K_TRAIN="train"
GSM8K_EVAL="test[:500]"
GSM8K_TEST="test[500:]"

echo "========================================"
echo "GSM8K Model Comparison Experiment"
echo "========================================"
echo "Models: Llama-3.2-1B, Qwen2.5-0.5B"
echo "Dataset: GSM8K (math reasoning)"
echo "Methods: LoRA-Attention, LoRA-All, Full Fine-Tuning"
echo "Training: $NUM_EPOCHS epochs (early stopping patience=$EARLY_STOPPING_PATIENCE)"
echo "Total experiments: 6 (estimated time: 2-2.5 days)"
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
    echo "Experiment $exp_num/6: $(basename $output_dir)"
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

# Experiments 1-3: Llama-3.2-1B + GSM8K (3 methods)
run_training 1 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "openai/gsm8k" "main" \
    "$GSM8K_TRAIN" "$GSM8K_EVAL" \
    "lora-attention" "./results/llama-gsm8k-lora-attention" 1024

run_training 2 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "openai/gsm8k" "main" \
    "$GSM8K_TRAIN" "$GSM8K_EVAL" \
    "lora-all" "./results/llama-gsm8k-lora-all" 1024

run_training 3 \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "openai/gsm8k" "main" \
    "$GSM8K_TRAIN" "$GSM8K_EVAL" \
    "fullft" "./results/llama-gsm8k-fullft" 1024

# Experiments 4-6: Qwen2.5-0.5B + GSM8K (3 methods)
run_training 4 \
    "Qwen/Qwen2.5-0.5B-Instruct" \
    "openai/gsm8k" "main" \
    "$GSM8K_TRAIN" "$GSM8K_EVAL" \
    "lora-attention" "./results/qwen-gsm8k-lora-attention" 1024

run_training 5 \
    "Qwen/Qwen2.5-0.5B-Instruct" \
    "openai/gsm8k" "main" \
    "$GSM8K_TRAIN" "$GSM8K_EVAL" \
    "lora-all" "./results/qwen-gsm8k-lora-all" 1024

run_training 6 \
    "Qwen/Qwen2.5-0.5B-Instruct" \
    "openai/gsm8k" "main" \
    "$GSM8K_TRAIN" "$GSM8K_EVAL" \
    "fullft" "./results/qwen-gsm8k-fullft" 1024

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
run_evaluation 1 "meta-llama/Llama-3.2-1B-Instruct" "openai/gsm8k" "main" "$GSM8K_TEST" "lora-attention" "./results/llama-gsm8k-lora-attention"
run_evaluation 2 "meta-llama/Llama-3.2-1B-Instruct" "openai/gsm8k" "main" "$GSM8K_TEST" "lora-all" "./results/llama-gsm8k-lora-all"
run_evaluation 3 "meta-llama/Llama-3.2-1B-Instruct" "openai/gsm8k" "main" "$GSM8K_TEST" "fullft" "./results/llama-gsm8k-fullft"
run_evaluation 4 "Qwen/Qwen2.5-0.5B-Instruct" "openai/gsm8k" "main" "$GSM8K_TEST" "lora-attention" "./results/qwen-gsm8k-lora-attention"
run_evaluation 5 "Qwen/Qwen2.5-0.5B-Instruct" "openai/gsm8k" "main" "$GSM8K_TEST" "lora-all" "./results/qwen-gsm8k-lora-all"
run_evaluation 6 "Qwen/Qwen2.5-0.5B-Instruct" "openai/gsm8k" "main" "$GSM8K_TEST" "fullft" "./results/qwen-gsm8k-fullft"

echo ""
echo "========================================"
echo "All Evaluations Completed!"
echo "========================================"
echo "Generating comparison visualizations..."
echo ""

# Generate comparison charts
$PYTHON visualize_comprehensive.py \
    --results_dir ./results \
    --output gsm8k_comparison.png

echo ""
echo "========================================"
echo "All Experiments Completed!"
echo "========================================"
echo "Results:"
echo "  - Training results: ./results/"
echo "  - Comparison chart: gsm8k_comparison.png"
echo ""
echo "Summary of experiments:"
echo "  1. Llama-3.2-1B + GSM8K (LoRA-Attention, LoRA-All, Full FT)"
echo "  2. Qwen2.5-0.5B + GSM8K (LoRA-Attention, LoRA-All, Full FT)"
echo "========================================"
