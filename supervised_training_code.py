import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, AutoTokenizer, get_scheduler
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import os

# ✅ 檢查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備: {device}")

# ✅ 載入 Pre-trained RoBERTa
model_path = r"C:\Users\Administrator\Desktop\roberta_mlm_trained"
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)  # 二分類

# ✅ 載入 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ 載入新聞數據集（請確保 JSON 格式正確）
dataset_path = r"C:\Users\Administrator\Desktop\ModelTraining\fake_and_real_news.json"
dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

# ✅ 文字轉 Token
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )

dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
dataset = dataset.train_test_split(test_size=0.1)  # 切割成 訓練集 90% / 測試集 10%

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ✅ 建立 DataLoader
batch_size = 8  # 如果 GPU 記憶體不足，調小 batch_size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

# ✅ 初始化 Optimizer、Scheduler
learning_rate = 2e-5
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

num_training_steps = len(train_dataloader) * 3  # 設定 epochs=3
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
)

# ✅ 設定損失函數
loss_fn = nn.CrossEntropyLoss()

# ✅ 訓練函式
def train_model(model, train_dataloader, eval_dataloader, epochs=3):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        # ✅ 評估模型
        model.eval()
        eval_loss = 0
        preds, labels = [], []

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()

                preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                labels.extend(batch["labels"].cpu().numpy())

        avg_eval_loss = eval_loss / len(eval_dataloader)
        accuracy = accuracy_score(labels, preds)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Eval Loss={avg_eval_loss:.4f}, Accuracy={accuracy:.4f}")

        # ✅ 每 1 個 Epoch 儲存一次
        save_path = f"C:\\Users\\Administrator\\Desktop\\ModelTraining\\roberta_finetuned_epoch_{epoch+1}"
        model.save_pretrained(save_path)
        print(f"✅ 模型已儲存至: {save_path}")

    return model

# ✅ 開始訓練
print("🚀 開始 Supervised Training...")
model = train_model(model, train_dataloader, eval_dataloader, epochs=3)

# ✅ 儲存最終模型
final_model_path = r"C:\Users\Administrator\Desktop\ModelTraining\roberta_finetuned_final"
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"🎉 訓練完成，模型已儲存至: {final_model_path}")
