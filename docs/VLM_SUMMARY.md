# VLM 實驗框架總結

## ✅ 已完成的工作

### 📁 新增檔案 (5個)

1. **`sft_vlm_compare.py`** (15KB)
   - VLM 訓練主腳本
   - 支援 LLaVA-1.5-7B
   - 實現 LoRA-Attention, LoRA-All, Full FT
   - 整合 ScienceQA 資料集處理
   - 支援 4-bit quantization 節省記憶體
   - 凍結 Vision Tower 選項

2. **`evaluate_vlm.py`** (8.8KB)
   - VLM 評估腳本
   - 支援多選題準確率評估
   - 自動答案提取 (A/B/C/D/E)
   - 支援 LoRA adapter 載入和合併
   - 生成詳細預測結果

3. **`run_scienceqa_experiments.sh`** (4.4KB)
   - 自動化實驗運行腳本
   - 依序執行 3 種方法
   - 自動訓練 + 評估
   - 包含進度顯示和時間估計

4. **`VLM_QUICKSTART.md`** (7.3KB)
   - 完整快速入門指南
   - 硬體/軟體需求說明
   - 安裝步驟
   - 運行範例
   - 常見問題解答
   - 進階配置選項

5. **`requirements-vlm.txt`** (623B)
   - VLM 實驗所需依賴清單
   - 包含版本限制
   - 註明安裝順序

### 📝 更新檔案

1. **`README.md`**
   - 新增 VLM 實驗章節
   - 更新目錄結構
   - 新增 VLM 配置說明
   - 新增資料集推薦
   - 新增硬體需求表

---

## 🎯 核心特點

### 1. 遵循 "LoRA without Regret" 方法論

✅ **高 LoRA Rank**: 256 (與原實驗一致)
✅ **10倍學習率差異**: LoRA 1e-4 vs Full FT 1e-5
✅ **應用於所有層**: LoRA-All 包含 attention + MLP
✅ **小批次大小**: Effective batch size = 4
✅ **Early Stopping**: Patience = 3 epochs

### 2. VLM 特定優化

✅ **Vision Tower 凍結**: 公平比較，只微調 LLM 部分
✅ **4-bit Quantization**: 大幅降低記憶體需求
✅ **長序列支援**: Max length = 2048 (適合 image tokens)
✅ **圖片預處理**: 整合 AutoProcessor
✅ **多選題評估**: 智能答案提取

### 3. 易用性設計

✅ **一鍵運行**: `bash run_scienceqa_experiments.sh`
✅ **自動下載**: 資料集自動從 HuggingFace Hub 載入
✅ **詳細日誌**: 訓練指標和評估結果自動保存
✅ **錯誤處理**: 過濾無圖片樣本
✅ **彈性配置**: 豐富的命令列參數

---

## 📊 實驗設計

### 對照實驗

| 方法 | Target Modules | 參數量 | 預期記憶體 |
|------|---------------|--------|-----------|
| **LoRA-Attention** | Q/K/V/O (4個) | ~33M | 12-16 GB |
| **LoRA-All** | Attention + MLP (7個) | ~58M | 16-20 GB |
| **Full FT** | All parameters | ~7B | 24-32 GB |

### 評估指標

- **準確率** (Accuracy): 多選題正確率
- **GPU 記憶體** (Peak memory usage)
- **訓練時間** (Total training time)
- **訓練損失** (Loss curve)

---

## 🚀 使用流程

### 快速開始 (3 步驟)

```bash
# 1. 安裝依賴
pip install -r requirements-vlm.txt

# 2. 運行實驗
bash run_scienceqa_experiments.sh

# 3. 查看結果
cat ./results/llava-scienceqa-*/eval_results.json
```

### 單獨訓練

```bash
# LoRA-All (推薦)
python3 sft_vlm_compare.py \
  --model llava-hf/llava-1.5-7b-hf \
  --method lora-all \
  --out ./results/llava-scienceqa-lora-all \
  --bf16 --use_4bit
```

### 評估

```bash
python3 evaluate_vlm.py \
  --model_path ./results/llava-scienceqa-lora-all \
  --method lora-all \
  --bf16 --use_4bit
```

---

## 📈 預期結果

基於原始 LLM 實驗的發現：

### 性能預測

| 方法 | 準確率 | 相對表現 |
|------|--------|---------|
| LoRA-Attention | ~65% | Baseline |
| LoRA-All | ~72% | +7% |
| Full FT | ~70% | +5% |

**關鍵預期**:
1. ✨ **LoRA-All 可能超越 Full FT** (如同 ARC-Challenge 結果)
2. 💾 **記憶體節省 30-40%** (LoRA vs Full FT)
3. ⚡ **訓練時間相近** (差異 < 20%)
4. 📊 **LoRA-All 顯著優於 LoRA-Attention** (+7% 或更多)

### 驗證假設

1. **H1**: LoRA-All 在 VLM 上也能達到或超越 Full FT
2. **H2**: 應用 LoRA 於 MLP 層對 VLM 同樣重要
3. **H3**: 高學習率 (10x) 對 VLM LoRA 訓練至關重要
4. **H4**: 小資料集 (ScienceQA ~6K) 上 LoRA 的 regularization 效果顯著

---

## 🔮 後續擴展

### 短期 (1-2週)

- [ ] 添加 A-OKVQA 資料集支援
- [ ] 添加 ChartQA 資料集支援
- [ ] 實作 VLM 可視化腳本
- [ ] 測試不同 LoRA rank (128, 512)

### 中期 (1個月)

- [ ] 測試更大的 VLM (LLaVA-1.5-13B, LLaVA-NeXT)
- [ ] 不凍結 Vision Tower 的實驗
- [ ] 比較不同 vision encoder (CLIP vs SigLIP)
- [ ] 多資料集綜合評估

### 長期 (2-3個月)

- [ ] 擴展到其他 VLM 架構 (Qwen-VL, InternVL)
- [ ] 測試 DoRA (Weight-Decomposed LoRA)
- [ ] 實作多任務學習
- [ ] 撰寫技術報告或論文

---

## 📚 參考資源

### 核心論文/文章
1. [LoRA without Regret (HuggingFace)](https://huggingface.co/docs/trl/main/en/lora_without_regret)
2. [Why You Should Use LoRA (Thinking Machines)](https://thinkingmachines.ai/blog/lora/)
3. [LoRA: Low-Rank Adaptation (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)

### 模型和資料集
- LLaVA: https://github.com/haotian-liu/LLaVA
- ScienceQA: https://huggingface.co/datasets/derek-thomas/ScienceQA

### 工具和框架
- TRL: https://github.com/huggingface/trl
- PEFT: https://github.com/huggingface/peft
- Transformers: https://github.com/huggingface/transformers

---

## 💡 關鍵洞察

### 從 LLM 到 VLM 的遷移

1. **相同原則仍適用**:
   - 高 rank (256)
   - 10x 學習率
   - 應用於所有層

2. **VLM 特定考量**:
   - Vision Tower 凍結策略
   - 更長的序列長度需求
   - 圖片預處理的影響

3. **多模態的優勢**:
   - 圖片提供額外監督信號
   - 可能需要更高容量 (higher rank?)
   - 泛化能力可能更強

### 實驗設計原則

1. **控制變量**: 凍結 vision tower 保證公平比較
2. **記憶體效率**: 4-bit quantization 讓 Full FT 可行
3. **自動化**: 減少人為錯誤，提高可重複性
4. **可擴展**: 易於添加新資料集和模型

---

## ✨ 成果亮點

1. **完整的 VLM 實驗框架**: 從訓練到評估的端到端流程
2. **遵循最佳實踐**: 基於最新研究的配置
3. **生產就緒**: 穩健的錯誤處理和日誌記錄
4. **詳細文檔**: 新手友善的快速入門指南
5. **可重現性**: 所有超參數明確記錄

---

## 🙏 致謝

本實驗框架基於:
- HuggingFace 團隊的 "LoRA without Regret" 研究
- Thinking Machines 的 LoRA 分析文章
- 原始 LLM 實驗的發現和方法論

---

**狀態**: ✅ 開發完成，可開始實驗
**最後更新**: 2025-10-28
**維護者**: [Your Name]
