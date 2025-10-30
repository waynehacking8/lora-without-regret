# 模型下載指南

## 📥 下載狀態

當前正在下載：
- ✅ **Token 驗證**: 成功
- 🔄 **LLaVA-1.5-7B**: 下載中 (~14GB)
- ⏳ **ScienceQA**: 待下載 (~1GB)

---

## 📊 下載信息

### LLaVA-1.5-7B 模型
- **大小**: ~14GB
- **文件數**: 17 個
- **包含**:
  - 模型權重 (safetensors)
  - Tokenizer
  - Processor (vision + text)
  - 配置文件

### ScienceQA 資料集
- **大小**: ~1GB
- **Splits**:
  - Train: ~6K samples (with images)
  - Validation: ~1K samples
  - Test: ~2K samples
- **格式**: Images + Multiple choice questions

---

## ⏱️ 預計時間

| 網速 | 預計時間 |
|------|---------|
| 100 Mbps | 20-30 分鐘 |
| 500 Mbps | 5-10 分鐘 |
| 1 Gbps | 3-5 分鐘 |

---

## 🔍 檢查下載進度

### 方法 1: 查看腳本輸出
下載腳本正在背景運行，會顯示進度條和狀態訊息。

### 方法 2: 檢查快取目錄
```bash
# HuggingFace 預設快取位置
ls -lh ~/.cache/huggingface/hub/

# 查看已下載的模型
ls -lh ~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/
```

### 方法 3: 監控磁碟使用
```bash
# 查看快取目錄大小
du -sh ~/.cache/huggingface/

# 即時監控
watch -n 5 'du -sh ~/.cache/huggingface/'
```

---

## ✅ 下載完成後

下載完成後，你會看到：

```
✅ 所有下載成功!

下一步:
  運行實驗: bash run_scienceqa_experiments.sh

📁 下載位置: /home/wayne/.cache/huggingface
```

### 驗證下載

```bash
# 驗證模型文件
python3 -c "
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
print('✅ 模型驗證成功!')
"

# 驗證資料集
python3 -c "
from datasets import load_dataset
ds = load_dataset('derek-thomas/ScienceQA', split='train[:10]')
print(f'✅ 資料集驗證成功! ({len(ds)} samples)')
"
```

---

## 🔄 重新下載

如果下載中斷或失敗：

```bash
# 重新運行下載腳本（會自動恢復）
bash download_models.sh

# 或使用 Python 腳本
python3 download_models.py --token_file ./huggingface_token.txt
```

下載會自動從中斷處繼續，不會重複下載已有文件。

---

## 🗑️ 清理快取

如果需要重新下載或清理空間：

```bash
# 刪除特定模型快取
rm -rf ~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/

# 刪除特定資料集快取
rm -rf ~/.cache/huggingface/datasets/derek-thomas___science_qa/

# 清理整個 HuggingFace 快取（謹慎使用！）
rm -rf ~/.cache/huggingface/
```

---

## 💾 磁碟空間需求

### 最低要求
- 模型: 14GB (LLaVA-1.5-7B)
- 資料集: 1GB (ScienceQA)
- 訓練結果: ~5GB (per method)
- **總計**: ~35GB

### 推薦
- 預留 50GB+ 空間
- 考慮多個實驗和模型版本

---

## 📍 快取位置

HuggingFace 使用以下優先順序決定快取位置：

1. `HF_HOME` 環境變數
2. `XDG_CACHE_HOME/huggingface` (Linux)
3. `~/.cache/huggingface` (預設)

### 自定義快取位置

```bash
# 設定環境變數（臨時）
export HF_HOME=/path/to/custom/cache
python3 download_models.py

# 永久設定（加入 ~/.bashrc）
echo 'export HF_HOME=/path/to/custom/cache' >> ~/.bashrc
source ~/.bashrc
```

或使用腳本參數：

```bash
python3 download_models.py --cache_dir /path/to/custom/cache
```

---

## 🌐 網路問題

### 連線問題
如果遇到連線錯誤：

```bash
# 使用鏡像站（中國大陸用戶）
export HF_ENDPOINT=https://hf-mirror.com
python3 download_models.py
```

### 代理設定
```bash
# HTTP/HTTPS 代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
python3 download_models.py
```

---

## 📋 下載其他模型/資料集

### 下載更多模型

```bash
# LLaVA-1.5-13B (更大的模型)
python3 download_models.py \
  --models llava-hf/llava-1.5-13b-hf

# 同時下載多個模型
python3 download_models.py \
  --models \
    llava-hf/llava-1.5-7b-hf \
    llava-hf/llava-1.5-13b-hf
```

### 下載其他 VLM 資料集

```bash
# A-OKVQA
python3 download_models.py \
  --skip_models \
  --datasets HuggingFaceM4/A-OKVQA

# ChartQA
python3 download_models.py \
  --skip_models \
  --datasets ahmed-masry/ChartQA
```

---

## 🚨 常見問題

### Q1: 下載速度很慢
**A**:
- 檢查網路連線
- 考慮使用鏡像站（見上方）
- 嘗試在非高峰時段下載

### Q2: 磁碟空間不足
**A**:
- 清理不需要的快取
- 更改快取位置到更大的磁碟
- 只下載必要的模型

### Q3: Token 驗證失敗
**A**:
- 檢查 token 文件內容是否正確
- 確認 token 沒有過期
- 訪問 https://huggingface.co/settings/tokens 重新生成

### Q4: 下載中斷
**A**:
- 重新運行腳本（會自動恢復）
- HuggingFace Hub 支援斷點續傳
- 不會重複下載已有文件

---

## 📞 需要幫助？

- HuggingFace Hub 文檔: https://huggingface.co/docs/hub
- LLaVA 項目: https://github.com/haotian-liu/LLaVA
- ScienceQA 資料集: https://huggingface.co/datasets/derek-thomas/ScienceQA

---

**狀態**: 🔄 下載進行中
**最後更新**: 2025-10-28
