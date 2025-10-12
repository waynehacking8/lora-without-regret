# LoRA without Regret - Experiments

比較 LoRA-Attention、LoRA-All 和 Full Fine-Tuning 的實驗項目

## 目錄結構

```
.
├── results/              # 已完成的實驗結果
│   ├── llama-gsm8k-*    # GSM8K 數學推理 (3個方法)
│   ├── llama-csqa-*     # CommonsenseQA 常識問答 (3個方法)
│   └── llama-arc-*      # ARC 推理測試 (進行中)
│
├── visualizations/       # 專業可視化圖表
│   ├── gsm8k_professional.png
│   └── csqa_professional.png
│
├── archives/            # 歸檔文件
│   ├── logs/           # 訓練日誌
│   ├── visualizations/ # 舊版比較圖表
│   ├── old_experiments/# 舊實驗和不完整實驗
│   └── wandb/          # W&B 訓練記錄
│
└── scripts/            # 實驗腳本
    ├── sft_compare.py  # 訓練腳本
    ├── evaluate.py     # 評估腳本
    ├── visualize_comprehensive.py  # 可視化腳本
    └── run_arc_experiments.sh      # ARC 實驗
```

---

## Model Configuration

- **Base Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Framework**: TRL SFTTrainer + PEFT
- **Precision**: BF16

---

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | **256** |
| LoRA Alpha | **16** |
| LoRA Dropout | **0.0** |
| Init LoRA Weights | True |
| Task Type | CAUSAL_LM |

### Target Modules

**LoRA-Attention (4 modules):**
- `q_proj`, `k_proj`, `v_proj`, `o_proj`

**LoRA-All (7 modules):**
- `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, `down_proj`

---

## Training Hyperparameters

| Parameter | GSM8K | CSQA | ARC |
|-----------|-------|------|-----|
| **Dataset** | `gsm8k` | `commonsense_qa` | `allenai/ai2_arc` |
| **Learning Rate (LoRA)** | 2e-4 | 2e-4 | 2e-4 |
| **Learning Rate (Full FT)** | 2e-5 | 2e-5 | 2e-5 |
| **LR Scheduler** | Linear Decay | Linear Decay | Linear Decay |
| **Epochs** | 5 | 5 | 5 |
| **Early Stopping Patience** | 2 | 2 | 2 |
| **Batch Size** | 1 | 1 | 1 |
| **Gradient Accumulation** | 4 | 4 | 4 |
| **Effective Batch Size** | 4 | 4 | 4 |
| **Max Sequence Length** | 512 | 512 | 512 |

---

## Experimental Results

### GSM8K (Mathematical Reasoning)

| Method | Training Time | GPU Memory | Final Perplexity |
|--------|--------------|------------|------------------|
| **LoRA-Attention** | 12.9 min | 6.6 GB | 3.497 |
| **LoRA-All** | 13.1 min | 9.3 GB | 3.458 |
| **Full FT** | 11.5 min | 13.3 GB | 3.360 |

**可視化**: `visualizations/gsm8k_professional.png`

### CSQA (Commonsense QA)

| Method | Training Time | GPU Memory | Test Accuracy |
|--------|--------------|------------|---------------|
| **LoRA-Attention** | 8.0 min | 5.0 GB | 0.546 |
| **LoRA-All** | 10.0 min | 7.5 GB | 0.582 |
| **Full FT** | 10.5 min | 11.6 GB | 0.595 |

**可視化**: `visualizations/csqa_professional.png`

### ARC (AI2 Reasoning Challenge)

| Method | Dataset Config | Status |
|--------|----------------|--------|
| **LoRA-Attention** | ARC-Easy + ARC-Challenge | Planned |
| **LoRA-All** | ARC-Easy + ARC-Challenge | Planned |
| **Full FT** | ARC-Easy + ARC-Challenge | Planned |

---

## 運行實驗

```bash
# ARC 實驗 (簡單 + 困難題目)
bash run_arc_experiments.sh

# 生成可視化
python3 visualize_comprehensive.py --results_dir results --output comparison.png
```

---

## Key Observations

1. **LoRA Rank 256**: Much higher than typical LoRA setups (usually 8-64), providing more capacity
2. **LoRA Alpha 16**: Low alpha relative to rank (16/256 = 0.0625 scaling factor)
3. **Learning Rate**: LoRA uses 10x higher LR than Full FT (2e-4 vs 2e-5)
4. **Memory Efficiency**: LoRA-Attention uses ~50% less memory than Full FT
5. **Performance**: Full FT slightly outperforms LoRA, but LoRA-All is very competitive

---

## 文件說明

- `sft_compare.py`: 主訓練腳本，支持 LoRA 和 Full FT
- `evaluate.py`: 評估腳本，支持 perplexity 和 accuracy 評估
- `visualize_comprehensive.py`: 專業可視化腳本，生成 2×2 網格比較圖
- `run_arc_experiments.sh`: ARC 實驗腳本 (ARC-Easy + ARC-Challenge)
- `HYPERPARAMETERS.md`: 詳細超參數文檔 (已整合至此 README)
