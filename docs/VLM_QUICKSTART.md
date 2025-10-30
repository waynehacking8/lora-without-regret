# VLM 實驗快速入門指南

## 🎯 目標

將 "LoRA without Regret" 方法論應用於 Vision-Language Models (VLMs)，驗證其在多模態任務上的有效性。

## 📋 前置需求

### 硬體需求
- **GPU**: NVIDIA GPU with 16GB+ VRAM
  - 推薦: RTX 4090 (24GB), A5000 (24GB), A100 (40GB)
  - 最低: RTX 3090 (24GB) with 4-bit quantization
- **RAM**: 32GB+
- **儲存空間**: ~50GB (模型 + 資料集 + 結果)

### 軟體需求
```bash
# Python 3.8+
python3 --version

# CUDA 11.8+ / 12.1+
nvidia-smi
```

## 🚀 安裝依賴

```bash
# 創建虛擬環境 (推薦)
python3 -m venv venv
source venv/bin/activate

# 安裝核心依賴
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安裝 Transformers 和 TRL
pip install transformers>=4.40.0
pip install trl>=0.9.0
pip install peft>=0.10.0
pip install datasets>=2.18.0

# 安裝其他依賴
pip install accelerate>=0.27.0
pip install bitsandbytes>=0.43.0  # For 4-bit quantization
pip install pillow  # For image processing
pip install tqdm
pip install numpy
pip install matplotlib  # For visualization (optional)
```

## 📊 資料集準備

ScienceQA 會自動從 HuggingFace Hub 下載，無需手動準備。

```python
# 測試資料集載入
from datasets import load_dataset

ds = load_dataset("derek-thomas/ScienceQA", split="train[:10]")
print(f"Loaded {len(ds)} samples")
print(f"Columns: {ds.column_names}")
```

## 🏃 運行實驗

### 方法 1: 完整自動化實驗 (推薦)

```bash
# 運行所有 3 種方法 (LoRA-Attention, LoRA-All, Full FT)
bash run_scienceqa_experiments.sh
```

**預計時間**: 3-6 小時 (取決於 GPU)

### 方法 2: 單獨運行實驗

#### Step 1: LoRA-Attention (基準)

```bash
python3 sft_vlm_compare.py \
  --model llava-hf/llava-1.5-7b-hf \
  --dataset derek-thomas/ScienceQA \
  --method lora-attention \
  --out ./results/llava-scienceqa-lora-attention \
  --bf16 --use_4bit \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --grad_accum 4
```

**預計時間**: 1-2 小時

#### Step 2: LoRA-All (LoRA without Regret) ⭐

```bash
python3 sft_vlm_compare.py \
  --model llava-hf/llava-1.5-7b-hf \
  --dataset derek-thomas/ScienceQA \
  --method lora-all \
  --out ./results/llava-scienceqa-lora-all \
  --bf16 --use_4bit \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --grad_accum 4
```

**預計時間**: 1-2 小時

#### Step 3: Full Fine-Tuning

```bash
python3 sft_vlm_compare.py \
  --model llava-hf/llava-1.5-7b-hf \
  --dataset derek-thomas/ScienceQA \
  --method fullft \
  --out ./results/llava-scienceqa-fullft \
  --bf16 --use_4bit \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --grad_accum 4
```

**預計時間**: 1-2 小時

## 📈 評估模型

訓練完成後，評估各個模型：

```bash
# 評估 LoRA-Attention
python3 evaluate_vlm.py \
  --model_path ./results/llava-scienceqa-lora-attention \
  --base_model llava-hf/llava-1.5-7b-hf \
  --method lora-attention \
  --dataset derek-thomas/ScienceQA \
  --split "test[:500]" \
  --bf16 --use_4bit

# 評估 LoRA-All
python3 evaluate_vlm.py \
  --model_path ./results/llava-scienceqa-lora-all \
  --base_model llava-hf/llava-1.5-7b-hf \
  --method lora-all \
  --dataset derek-thomas/ScienceQA \
  --split "test[:500]" \
  --bf16 --use_4bit

# 評估 Full FT
python3 evaluate_vlm.py \
  --model_path ./results/llava-scienceqa-fullft \
  --base_model llava-hf/llava-1.5-7b-hf \
  --method fullft \
  --dataset derek-thomas/ScienceQA \
  --split "test[:500]" \
  --bf16 --use_4bit
```

## 📊 查看結果

### 訓練指標
```bash
# 查看訓練指標 (loss, GPU memory, training time)
cat ./results/llava-scienceqa-lora-attention/metrics_lora-attention.json
cat ./results/llava-scienceqa-lora-all/metrics_lora-all.json
cat ./results/llava-scienceqa-fullft/metrics_fullft.json
```

### 評估結果
```bash
# 查看測試準確率
cat ./results/llava-scienceqa-lora-attention/eval_results.json
cat ./results/llava-scienceqa-lora-all/eval_results.json
cat ./results/llava-scienceqa-fullft/eval_results.json
```

## 🎨 比較結果

建立簡單的比較腳本：

```python
import json

methods = ["lora-attention", "lora-all", "fullft"]

print(f"{'Method':<20} {'Accuracy':<12} {'GPU Memory':<12} {'Time (min)':<12}")
print("="*60)

for method in methods:
    # Load eval results
    eval_path = f"./results/llava-scienceqa-{method}/eval_results.json"
    metrics_path = f"./results/llava-scienceqa-{method}/metrics_{method}.json"

    with open(eval_path) as f:
        eval_data = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    accuracy = eval_data['accuracy']
    gpu_mem = max([m['gpu_memory_gb'] for m in metrics])
    time = metrics[-1]['elapsed_time_min']

    print(f"{method:<20} {accuracy:<12.2%} {gpu_mem:<12.1f} {time:<12.1f}")
```

## 🔧 常見問題

### Q1: GPU 記憶體不足 (OOM)

**解決方案**:
1. 使用 4-bit quantization: `--use_4bit`
2. 減少 batch size: `--per_device_train_batch_size 1`
3. 增加 gradient accumulation: `--grad_accum 8`
4. 減少 max_seq_length: `--max_seq_length 1024`

### Q2: 訓練速度太慢

**解決方案**:
1. 使用更強的 GPU (A100 > RTX 4090 > RTX 3090)
2. 增加 batch size (如果 GPU 記憶體允許)
3. 減少 training samples: `--max_samples 1000`

### Q3: 如何添加新的 VLM 資料集？

編輯 `sft_vlm_compare.py` 中的 `convert_to_vlm_format` 函數：

```python
def convert_to_vlm_format(example):
    """Add your dataset format here"""
    if "your_dataset_field" in example:
        # Your custom processing
        return {
            "image": example['image'],
            "text": your_formatted_text,
        }
    # ... existing formats
```

### Q4: LoRA rank 應該設多少？

- **小資料集** (<10K): rank=128-256
- **中型資料集** (10K-50K): rank=256-512
- **大型資料集** (>50K): rank=512-1024

原則: rank 越高，容量越大，但記憶體和訓練時間也越多。

## 📚 進階配置

### 調整學習率

```bash
# 如果訓練不穩定，降低學習率
python3 sft_vlm_compare.py \
  --lora_lr 5e-5 \    # 從 1e-4 降到 5e-5
  --fullft_lr 5e-6 \  # 從 1e-5 降到 5e-6
  ...
```

### 不凍結 Vision Tower

```bash
# 如果想微調整個模型（需要更多記憶體）
python3 sft_vlm_compare.py \
  --freeze_vision_tower False \
  ...
```

### 使用更高的 LoRA rank

```bash
python3 sft_vlm_compare.py \
  --lora_rank 512 \    # 從 256 提高到 512
  --lora_alpha 32 \    # 相應調整 alpha
  ...
```

## 🎯 預期結果

基於原始 LLM 實驗的發現：

| 方法 | 預期準確率 | GPU 記憶體 | 訓練時間 |
|------|-----------|-----------|---------|
| **LoRA-Attention** | ~60-70% | ~12-16 GB | 1-2 hrs |
| **LoRA-All** | ~70-75% | ~16-20 GB | 1-2 hrs |
| **Full FT** | ~70-75% | ~24-32 GB | 1.5-2.5 hrs |

**關鍵發現** (預期):
- LoRA-All 應該與 Full FT 相當或更好
- LoRA-All 節省 ~30-40% GPU 記憶體
- 在小資料集上 LoRA-All 可能優於 Full FT

## 📝 引用

如果這個實驗對你有幫助，請引用原始研究：

```bibtex
@article{lora_without_regret,
  title={LoRA without Regret},
  author={HuggingFace Team},
  year={2024},
  url={https://huggingface.co/docs/trl/main/en/lora_without_regret}
}
```

## 💬 支援

遇到問題？
1. 檢查 [常見問題](#常見問題)
2. 查看原始 README.md
3. 檢查 HuggingFace TRL 文檔

祝實驗順利！ 🚀
