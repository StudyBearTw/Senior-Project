from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ✅ 載入已訓練的模型
model_path = r"C:\Users\Administrator\Desktop\ModelTraining\roberta_finetuned_final"
model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ 進行預測
def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_label = torch.argmax(logits, dim=-1).item()
    return "Fake News" if pred_label == 1 else "Real News"

# 測試新文章
news_article = "這是一篇關於 AI 的新聞，內容是..."
print(f"預測結果: {predict_news(news_article)}")
