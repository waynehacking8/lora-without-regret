# VLM 腳本修正說明

## ⚠️ 重要發現

經過查閱 TRL 官方文檔和範例後，發現原始 `sft_vlm_compare.py` 有幾個關鍵問題需要修正。

---

## 🔧 必須修正的問題

### 1. ❌ 模型載入方式錯誤

**錯誤代碼**:
```python
from transformers import LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained(...)
```

**正確代碼**:
```python
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(...)
```

**原因**: TRL SFTTrainer 官方建議使用 `AutoModelForImageTextToText` 以獲得更好的兼容性。

---

### 2. ❌ max_seq_length 設置錯誤

**錯誤配置**:
```python
SFTConfig(
    max_seq_length=2048,  # ❌ 會截斷 image tokens!
    ...
)
```

**正確配置**:
```python
SFTConfig(
    max_length=None,  # ✅ 允許完整序列，不截斷 image tokens
    ...
)
```

**原因**: **VLM 的 image tokens 必須完整保留！** 截斷會導致訓練錯誤或性能下降。這是 TRL 官方文檔明確指出的關鍵配置。

**官方說明**:
> "For VLMs, truncating may remove image tokens, leading to errors during training. Set `max_length=None`."

---

### 3. ❌ gradient_checkpointing 配置缺失

**錯誤**: 沒有設置 `gradient_checkpointing_kwargs`

**正確配置**:
```python
training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
```

**原因**: VLM 訓練需要 `use_reentrant=False` 以避免梯度檢查點的兼容性問題。

---

### 3.5. ❌ **processing_class 參數錯誤** ⭐ **最關鍵的修正！**

**錯誤代碼**:
```python
trainer = SFTTrainer(
    model=model,
    processing_class=processor.tokenizer,  # ❌ 錯誤！
    ...
)
```

**正確代碼**:
```python
trainer = SFTTrainer(
    model=model,
    processing_class=processor,  # ✅ 正確！傳遞完整的 Processor
    ...
)
```

**錯誤原因**:
TRL 的 `SFTTrainer` 使用以下邏輯來判斷模型是否為 VLM:

```python
# TRL 源代碼驗證邏輯
if isinstance(processing_class, ProcessorMixin):
    self._is_vlm = True  # ✅ 識別為 VLM
elif isinstance(processing_class, PreTrainedTokenizerBase):
    self._is_vlm = False  # ❌ 識別為純文本模型
```

當傳遞 `processor.tokenizer` 時:
- `processor.tokenizer` 是 `PreTrainedTokenizerBase` 實例
- TRL 判定 `_is_vlm = False`
- 資料集有 `images` 列但模型不是 VLM
- **拋出錯誤**: `ValueError: The dataset appears to be vision-related...but the provided model does not seem to be a vision-language model`

當傳遞 `processor` 時:
- `processor` 是 `ProcessorMixin` 實例
- TRL 判定 `_is_vlm = True`
- ✅ 訓練可以正常進行

**官方來源**: `trl/trainer/sft_trainer.py` 第 743 行驗證邏輯

---

### 4. ⚠️ 自定義 data collator 可能不必要

**當前做法**:
```python
def collate_fn(examples, processor):
    # 自定義處理 images 和 text
    ...
```

**官方做法**:
```python
# SFTTrainer 自動處理，不需要自定義 collator
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # 直接傳遞原始資料集
    ...
)
```

**說明**: TRL 有內建的 `DataCollatorForVisionLanguageModeling`，會自動處理 images 列。除非有特殊需求，否則不需要自定義。

---

### 5. ⚠️ 資料集格式需求

**SFTTrainer 對 VLM 的要求**:
1. 資料集必須有 `images` 列（PIL Image 格式）
2. 對話格式應該是標準的 messages 格式
3. 不需要手動預處理成 text 格式

**ScienceQA 格式檢查**:
- ✅ 有 `image` 列
- ❓ 需要轉換成 messages 格式（待確認）
- ❓ 列名是 `image` 還是 `images`（待確認）

---

### 6. ⚠️ bf16 vs fp16

**查找結果**:
- 部分資源建議 LLaVA 使用 **fp16** 而不是 bf16
- LLaVA 預設存儲為 float16

**建議**:
- 測試兩種精度
- 查看 LLaVA 模型卡推薦配置

---

### 7. ⚠️ LoRA rank 配置

**原配置**: rank=256 (遵循 LoRA without Regret)

**常見 VLM 配置**: rank=16-64

**建議**:
- 先用 rank=128 測試
- 如果記憶體允許，可以用 256
- rank=16 可能太小，無法充分利用容量

---

## 📋 修正後的關鍵配置

```python
from transformers import AutoModelForImageTextToText
from trl import SFTTrainer, SFTConfig

# 1. 使用正確的模型類
model = AutoModelForImageTextToText.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,  # 或 bfloat16
    device_map="auto",
)

# 2. 正確的訓練配置
training_args = SFTConfig(
    output_dir="./output",
    max_length=None,  # ⭐ 關鍵！
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
)

# 3. 設置 gradient checkpointing
training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

# 4. 簡化的訓練器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 直接傳遞
    peft_config=peft_config,
)
```

---

## 🔍 需要驗證的事項

### 1. ScienceQA 資料集格式
```python
# 檢查資料集結構
from datasets import load_dataset
ds = load_dataset("derek-thomas/ScienceQA", split="train[:10]")

# 必須確認:
# - 是否有 'images' 或 'image' 列?
# - 圖片是 PIL Image 格式還是路徑?
# - 需要轉換成 messages 格式嗎?
```

### 2. 對話格式轉換
ScienceQA 原始格式:
```python
{
    "question": "...",
    "choices": ["A", "B", "C", "D"],
    "answer": 2,
    "image": PIL.Image
}
```

需要轉換成 TRL 期望的格式嗎？
```python
{
    "messages": [
        {"role": "user", "content": "question + choices"},
        {"role": "assistant", "content": "answer"}
    ],
    "images": [PIL.Image]
}
```

### 3. Processor 的使用
- SFTTrainer 會自動呼叫 processor 嗎？
- 還是需要手動預處理？

---

## 🎯 建議的行動步驟

1. **先停止當前下載**（如果需要）
2. **檢查 ScienceQA 資料集格式**
3. **基於官方範例重寫腳本**
4. **創建小規模測試**（10-100 samples）
5. **驗證訓練可以正常運行**
6. **再進行完整實驗**

---

## 📚 參考資源

- **TRL 官方範例**: `trl/examples/scripts/sft_vlm.py`
- **TRL 文檔**: https://huggingface.co/docs/trl/sft_trainer
- **官方教程**: https://www.philschmid.de/fine-tune-multimodal-llms-with-trl
- **LLaVA 模型卡**: https://huggingface.co/llava-hf/llava-1.5-7b-hf

---

---

### 4. ❌ **PIL Image 序列化問題**

**錯誤**: dataset.map() 會自動序列化 PIL Image 成字典

```python
formatted_dataset = dataset.map(
    format_fn,
    remove_columns=dataset.column_names,
    # 錯誤:沒有指定 features,導致 PIL Image 被序列化
)
# 結果:images = [{'bytes': b'...', 'path': None}]  # 字典,不是 PIL Image!
```

**正確做法**:

```python
from datasets import Features, Sequence, Value, Image as ImageFeature

features = Features({
    'messages': [{'role': Value('string'), 'content': Value('string')}],
    'images': Sequence(ImageFeature(decode=True))  # 保留 PIL Images!
})

formatted_dataset = dataset.map(
    format_fn,
    remove_columns=dataset.column_names,
    features=features,  # ✅ 指定 features 保留 PIL Image
)
# 結果:images = [<PIL.Image>]  # 真正的 PIL Image!
```

**錯誤訊息**:
```
TypeError: only a single or a list of entries is supported but got type=<class 'dict'>
```

---

## ⏳ 下一步

等待：
1. ✅ 模型下載完成
2. ✅ 確認 ScienceQA 資料格式
3. ✅ 修復所有關鍵問題
4. 🔄 驗證訓練可以正常運行

**狀態**: 🔄 正在測試最終修復版本
**最後更新**: 2025-10-29

**已修復的關鍵問題**:
1. ✅ processing_class 參數(傳遞 processor 而非 processor.tokenizer)
2. ✅ wandb 配置(添加 report_to="none")
3. ✅ PIL Image 序列化(使用 features 參數保留格式)
