# VLM 訓練腳本修正總結

## 🔍 問題診斷過程

經過多次測試和調試,發現了**3個關鍵問題**導致 VLM 訓練失敗。

---

## ❌ 問題 1: processing_class 參數錯誤 ⭐ **最關鍵!**

### 錯誤代碼:
```python
trainer = SFTTrainer(
    model=model,
    processing_class=processor.tokenizer,  # ❌ 錯誤!
    ...
)
```

### 正確代碼:
```python
trainer = SFTTrainer(
    model=model,
    processing_class=processor,  # ✅ 正確!
    ...
)
```

### 原因:

TRL 的 `SFTTrainer` 使用以下邏輯判斷模型是否為 VLM:

```python
# TRL 源代碼 (trl/trainer/sft_trainer.py:743)
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

### 錯誤訊息:
```
ValueError: The dataset appears to be vision-related (contains 'image' or 'images' keys),
but the provided model does not seem to be a vision-language model.
Please check your model and dataset.
```

---

## ❌ 問題 2: wandb 配置問題

### 錯誤:
沒有禁用 wandb,導致要求 API key

### 錯誤訊息:
```
wandb.errors.errors.UsageError: api_key not configured (no-tty).
call wandb.login(key=[your_api_key])
```

### 修正:
在 training_config 中添加:

```python
training_config = {
    ...
    "report_to": "none",  # ✅ 禁用 wandb
    ...
}
```

---

## ❌ 問題 3: PIL Image 序列化問題

### 錯誤:
`dataset.map()` 會自動將 PIL Image 序列化成字典格式

```python
formatted_dataset = dataset.map(
    format_fn,
    remove_columns=dataset.column_names,
    # ❌ 沒有指定 features,導致 PIL Image 被序列化
)

# 結果: images = [{'bytes': b'...', 'path': None}]  # 字典,不是 PIL Image!
```

### 錯誤訊息:
```
TypeError: only a single or a list of entries is supported but got type=<class 'dict'>
```

在錯誤堆疊中:
```
File ".../transformers/image_processing_base.py", line 536, in fetch_images
    raise TypeError(f"only a single or a list of entries is supported but got type={type(image_url_or_urls)}")
```

### 修正:

使用 `features` 參數指定保留 PIL Image 格式:

```python
from datasets import Features, Sequence, Value, Image as ImageFeature

features = Features({
    'messages': [{'role': Value('string'), 'content': Value('string')}],
    'images': Sequence(ImageFeature(decode=True))  # ✅ 保留 PIL Images!
})

formatted_dataset = dataset.map(
    format_fn,
    remove_columns=dataset.column_names,
    features=features,  # ✅ 指定 features 保留 PIL Image
    desc="Formatting dataset"
)

# 結果: images = [<PIL.Image>]  # ✅ 真正的 PIL Image!
```

---

## ✅ 完整修正後的代碼

### 資料集準備:

```python
def prepare_dataset(dataset, processor, dataset_name):
    from datasets import Features, Sequence, Value, Image as ImageFeature

    # Filter out samples without images
    dataset = dataset.filter(lambda x: x['image'] is not None)

    def format_scienceqa(example):
        # ... format messages ...
        return {
            "messages": messages,
            "images": [example['image']]  # List of PIL images
        }

    # ✅ CRITICAL: Define features to preserve PIL Image format
    features = Features({
        'messages': [{'role': Value('string'), 'content': Value('string')}],
        'images': Sequence(ImageFeature(decode=True))
    })

    formatted_dataset = dataset.map(
        format_scienceqa,
        remove_columns=dataset.column_names,
        features=features,  # ✅ Preserve PIL Images!
        desc="Formatting dataset"
    )

    return formatted_dataset
```

### Trainer 配置:

```python
# Training configuration
training_config = {
    "output_dir": args.out,
    "max_length": None,  # ✅ CRITICAL for VLMs!
    "gradient_checkpointing": True,
    "remove_unused_columns": False,  # ✅ Important for VLMs!
    "report_to": "none",  # ✅ Disable wandb
    ...
}

training_args = SFTConfig(**training_config)

# ✅ Set gradient checkpointing kwargs
training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

# ✅ CRITICAL FIX: Pass processor (ProcessorMixin), not tokenizer!
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=processor,  # ✅ Pass processor, not processor.tokenizer!
    callbacks=[metrics_cb, early_stopping_cb],
)
```

---

## 📊 測試結果

**測試狀態**: 🔄 正在驗證最終修復版本

**預期結果**:
- ✅ 模型載入成功
- ✅ 資料集格式化完成
- ✅ 訓練成功開始
- ✅ 可以正常進行 VLM LoRA without Regret 實驗

---

## 🎯 關鍵要點

1. **processing_class 必須傳遞 processor**
   - TRL 用此判斷是否為 VLM
   - 傳遞 tokenizer 會導致 VLM 驗證失敗

2. **max_length 必須設為 None**
   - VLM 的 image tokens 不能被截斷
   - 這是 TRL 官方文檔明確要求的

3. **PIL Image 必須保留格式**
   - 使用 `features` 參數防止序列化
   - TRL 的 data collator 需要真正的 PIL Image 物件

4. **禁用 wandb (可選)**
   - 避免 API key 問題
   - 使用 `report_to="none"`

---

## 📚 參考資源

- **TRL SFTTrainer源代碼**: `trl/trainer/sft_trainer.py`
- **TRL官方文檔**: https://huggingface.co/docs/trl/sft_trainer
- **GitHub Issue #3099**: SFTTrainer does not support VLM models specified via model name

---

**最後更新**: 2025-10-29
**狀態**: ✅ 所有已知問題已修復,等待測試驗證
