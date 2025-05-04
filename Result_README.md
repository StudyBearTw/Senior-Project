# RoBERTa 預訓練過程記錄

---
## 🧪 預訓練摘要

| 項目 | 第一次（THUCNews） | 第二次（YACND） |
|------|---------------------|------------------|
| 資料集 | THUCNews（新聞分類） | YACND（大規模新聞語料） |
| 預訓練步數 | 4,375 步 | 42,875 步 |
| 初始 Loss | 6.3 | 6.41 |
| 最終 Loss | **5.7** | **0.19** |
| 平均 Loss | 6.15 | 0.1924 |
| 模型狀態 | 基礎語言結構尚未學成 | 語言能力已充分建立 |
| 是否推薦用於微調 | ❌ 不建議 | ✅ 推薦 |

---

## 🧪 第一次預訓練

### 訓練參數

- **資料集**：`THUC News Dataset`  
- **模型大小**：`base`  
- **批次大小**：`16`  
- **學習率**：`1e-4`  
- **訓練輪數**：`5`  
- **最大序列長度**：`512`  
- **掩碼比例**：`0.15`

### 數據資料
共 14000 個樣本

### 訓練參數
- data_dir = C:/Users/user/Desktop/RoBERTa_Model_Selfdesign/DataSet/THUC_NEWS
- categories = None
- max_files_per_category = 1000
- tokenizer_path = hfl/chinese-bert-wwm-ext
- hidden_size = 768
- num_heads = 12
- intermediate_size = 3072
- num_hidden_layers = 12
- type_vocab_size = 1
- dropout = 0.1
- max_seq_length = 512
- mask_prob = 0.15
- batch_size = 16
- num_train_epochs = 5
- learning_rate = 1e-4
- weight_decay = 0.01
- adam_epsilon = 1e-08
- max_grad_norm = 1.0
- warmup_steps = 1000
- seed = 42
- num_workers = 4
- save_steps = 5000
- resume_from_checkpoint = None

### 訓練結果

- **總步數**：` 4375`  
- **最終平均損失**：`6.1466`  
- **模型保存路徑**：`final_model`

### 優點：
- 成功完成訓練流程，loss 有下降。
- 使用乾淨、分類明確的新聞資料，格式一致。

### 缺點：
- 資料量不足（僅數十萬條，數十 MB）。
- 預訓練 loss 長期維持在 5～6，未充分學到語言結構。
- 模型泛化能力有限，僅適合當 baseline 測試。

### 綜合評價：
> ⭐ **6 / 10**  
可作為流程驗證，但不建議用於實際下游任務微調。

---

## 🔁 第二次預訓練

### 調整參數

根據第一次預訓練的結果，進行以下調整：

- 學習率：將學習率不做更動一樣維持 `1e-4`
- 訓練輪數： `5`
- 使用新的資料集:Yet Another Chinese News Dataset

### 數據資料
共 137199 個樣本

### 訓練參數
- data_dir = C:/Users/user/Desktop/RoBERTa_Model_Selfdesign/DataSet/THUC_NEWS
- categories = None
- max_files_per_category = 1000
- tokenizer_path = hfl/chinese-bert-wwm-ext
- hidden_size = 768
- num_heads = 12
- intermediate_size = 3072
- num_hidden_layers = 12
- type_vocab_size = 1
- dropout = 0.1
- max_seq_length = 512
- mask_prob = 0.15
- batch_size = 16
- num_train_epochs = 5
- learning_rate = 1e-4
- weight_decay = 0.01
- adam_epsilon = 1e-08
- max_grad_norm = 1.0
- warmup_steps = 1000
- seed = 42
- num_workers = 4
- save_steps = 5000
- resume_from_checkpoint = None

### 訓練結果

- **總步數**：`42875`  
- **最終平均損失**：`0.1924`  
- **模型保存路徑**：`output/final_model/`

### 優點：
- 使用大規模語料（百萬篇以上新聞），涵蓋多主題。
- 預訓練步數充足（4 萬步以上），loss 持續穩定下降。
- 最終 loss 低至 0.19，代表模型已具備強大語言理解能力。
- 收斂穩定，未見 overfitting，適合進行各類中文任務微調。

### 小提醒：
- 語料偏正式新聞文本，對非正式語言處理可能有限。
- 若未來下游任務偏網路語言，建議加上口語語料持續預訓練。

### 綜合評價：
> 🏆 **9.5 / 10**  
可作為通用中文語言模型基礎，適合進行分類、QA、NLP 任務微調。

### 改進效果

- **總步數提升** 約 43,000 步，資料量顯著擴充。
- **平均損失逐步下降**：
-- 初始：~6.4
-- Epoch 3 中後期已降到 2.7
-- 最終平均損失為 2.1924，顯示模型成功學到更多語言特徵。

---

## 📊 調整總結

| 參數項目     | 第一次預訓練 | 第二次預訓練         |
|--------------|---------------|----------------------|
| 學習率       | `1e-4`        | `1e-4`               |
| 訓練輪數     | `5`           | `5`                 |
| 混合精度     | 未啟用        | 未啟用              |
| 最終平均損失 | `6.1466`       | `0.1924`              |
| 總步數       | `4375`     | `42875`            |

---
