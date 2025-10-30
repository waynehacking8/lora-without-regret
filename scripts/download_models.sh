#!/bin/bash
# 預先下載 VLM 模型和 ScienceQA 資料集
# 避免訓練時重複下載，節省時間

set -e  # Exit on error

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python not found! Please install Python 3.8+"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"
echo ""

# Check if huggingface_hub is installed
$PYTHON -c "import huggingface_hub" 2>/dev/null || {
    echo "❌ huggingface_hub not installed!"
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
}

# Token file path
TOKEN_FILE="./huggingface_token.txt"

# Check if token file exists
if [ ! -f "$TOKEN_FILE" ]; then
    echo "❌ Token 文件不存在: $TOKEN_FILE"
    echo ""
    echo "請創建文件並填入你的 HuggingFace token:"
    echo "  1. 訪問 https://huggingface.co/settings/tokens"
    echo "  2. 創建或複製 token"
    echo "  3. 保存到 $TOKEN_FILE"
    exit 1
fi

echo "================================================"
echo "VLM 模型和資料集下載"
echo "================================================"
echo "Token 文件: $TOKEN_FILE"
echo ""
echo "將下載:"
echo "  - 模型: llava-hf/llava-1.5-7b-hf (~14GB)"
echo "  - 資料集: derek-thomas/ScienceQA (~1GB)"
echo ""
echo "預計時間: 10-30 分鐘 (取決於網速)"
echo "================================================"
echo ""

# Ask for confirmation
read -p "開始下載? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消下載"
    exit 0
fi

# Run download script
$PYTHON download_models.py \
    --token_file "$TOKEN_FILE" \
    --models llava-hf/llava-1.5-7b-hf \
    --datasets derek-thomas/ScienceQA

echo ""
echo "================================================"
echo "下載完成! 現在可以運行實驗了:"
echo "  bash run_scienceqa_experiments.sh"
echo "================================================"
