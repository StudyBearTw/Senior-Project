# Senior-Project
College senior project: News detection (using RoBERTa model + BERT Tokenizer)
## 1.Research Objectives
1. Develop a model specifically for detecting the authenticity of Chinese news headlines.
2. Explore tokenization and representation learning techniques suitable for short texts (news headlines).
3. Train a model from scratch, including pre-training and supervised training.
4. Design an interpretable news headline classification system that provides a reasonable explanation mechanism.
## 2.Technical Selection
### Model Architecture:
Use RoBERTa as the base model, focusing on short-text processing.
### Tokenizer:
Utilize Tsinghua University's Chinese tokenizer to ensure compatibility with Chinese text.
### Data Sources:
Collect publicly available news headline datasets (e.g., Chinese Fake News Dataset).
### Training Approach:
Perform unsupervised pre-training.(use Masked Language Modeling)
Follow up with supervised training to fine-tune the model.
Potentially incorporate BiLSTM or other structures to enhance short-text feature extraction.
### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-score
