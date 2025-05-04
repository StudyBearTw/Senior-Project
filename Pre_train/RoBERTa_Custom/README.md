# RoBERTa 中文模型參數與架構說明

## 🔧 RoBERTaModel 的初始化參數

RoBERTaModel 是基礎語言模型，其初始化參數如下：

| 參數名稱               | 預設值    | 說明                                                         |
|------------------------|-----------|--------------------------------------------------------------|
| `vocab_size`           | 21128 | 詞彙表大小，即模型支援的詞彙數量。                          |
| `max_position_embeddings` | 512 | 最大位置嵌入數量，決定模型支援的最大序列長度。             |
| `hidden_size`          | 768       | 隱藏層大小，每個 token 的向量維度。                         |
| `num_heads`            | 12        | 多頭注意力機制的頭數。                                       |
| `intermediate_size`    | 3072      | 前饋層的中間層大小。                                         |
| `num_hidden_layers`    | 12        | Transformer 編碼器層數。                                     |
| `type_vocab_size`      | 1         | 類型詞彙表大小，常用於區分句子對（例如句子 A/B）。         |
| `dropout`              | 0.1       | Dropout 比例，用於防止過擬合。                              |

---

## 🧠 RobertaForMaskedLM 的初始化參數

RobertaForMaskedLM 是基於 RoBERTaModel 的 MLM（掩碼語言模型）變體，額外包含語言模型頭：

| 參數名稱               | 預設值    | 說明                                                         |
|------------------------|-----------|--------------------------------------------------------------|
| `vocab_size`           | 21128 | 詞彙表大小，即模型支援的詞彙數量。                          |
| `max_position_embeddings` | 512 | 最大位置嵌入數量，決定模型支援的最大序列長度。             |
| `hidden_size`          | 768       | 隱藏層大小，每個 token 的向量維度。                         |
| `num_heads`            | 12        | 多頭注意力機制的頭數。                                       |
| `intermediate_size`    | 3072      | 前饋層的中間層大小。                                         |
| `num_hidden_layers`    | 12        | Transformer 編碼器層數。                                     |
| `type_vocab_size`      | 1         | 類型詞彙表大小，常用於區分句子對。                          |
| `dropout`              | 0.1       | Dropout 比例，用於防止過擬合。                              |

### ➕ 附加組件：
- **`lm_head`**：語言模型頭，用於生成掩碼 token 的預測分數。
- **`loss_fn`**：交叉熵損失函數，用於計算 MLM 預測誤差。

---

## 🧱 RoBERTa 模型結構

### 1. 嵌入層（`RobertaEmbeddings`）：
- 功能：將輸入 token ID 轉換為向量表示。
- 組成：包括詞嵌入、位置嵌入和類型嵌入的加總。

### 2. 編碼器層（`RobertaEncoder`）：
- 組成：多層 Transformer 編碼器。
- 每層結構：包含多頭自注意力機制與前饋神經網路。

### 3. 輸出（依需求而定）：
| 輸出名稱            | 說明                                                         |
|---------------------|--------------------------------------------------------------|
| `last_hidden_state` | 最後一層所有 token 的隱藏狀態（主要輸出）。                  |
| `pooler_output`     | 第一個 token（通常是 `[CLS]`）的輸出，常用於分類任務。       |
| `hidden_states`     | 每一層的隱藏狀態（需設定 `output_hidden_states=True`）。     |
| `attentions`        | 每層的注意力權重（需設定 `output_attentions=True`）。        |

