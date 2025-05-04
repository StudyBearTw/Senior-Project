import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import logging
import sys
import argparse
from datetime import datetime
import pandas as pd  # 用於處理清理後的 CSV 資料集

# 导入您的自定义模型类
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RoBERTa_Custom.model import RobertaForMaskedLM

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# MLM 数据集类
class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 建立 masked_input 和 labels
        masked_input_ids, labels = self.mask_tokens(input_ids.clone())
        return {
            'input_ids': masked_input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
    
    def mask_tokens(self, input_ids):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
            for val in labels.unsqueeze(0).tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).squeeze(0)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算被 mask 的 token 的 loss
        
        # 替换策略：80% [MASK], 10% random token, 10% unchanged
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return input_ids, labels

# 加载清理后的数据集
def load_cleaned_data(file_path):
    """
    加载清理后的 CSV 数据集
    :param file_path: 清理后的 CSV 文件路径
    :return: 文本数据列表
    """
    logger.info(f"加载清理后的数据集: {file_path}")
    data = pd.read_csv(file_path)
    if "desc" not in data.columns:
        raise ValueError("清理后的数据集中未找到 'desc' 列")
    texts = data["desc"].dropna().tolist()
    logger.info(f"成功加载 {len(texts)} 条文本数据")
    return texts

# 训练函数
def train(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载 tokenizer
    logger.info("加载 tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    
    # 加载清理后的数据
    logger.info("加载清理后的文本数据...")
    texts = load_cleaned_data(args.data_dir)
    
    # 创建数据集
    logger.info("创建 MLM 数据集...")
    dataset = MLMDataset(
        texts, 
        tokenizer, 
        max_len=args.max_seq_length,
        mask_prob=args.mask_prob
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    logger.info(f"数据集创建完成，共 {len(dataset)} 个样本")
    logger.info(f"数据加载器创建完成，共 {len(dataloader)} 个批次")
    
    # 创建模型
    logger.info("创建模型...")
    model = RobertaForMaskedLM(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.max_seq_length,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        type_vocab_size=args.type_vocab_size,
        dropout=args.dropout
    )
    
    # 加载预训练检查点（如果有）
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info(f"从检查点加载模型: {args.resume_from_checkpoint}")
        model.load_state_dict(torch.load(args.resume_from_checkpoint))
    
    model.to(device)
    
    # 优化器和学习率调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    # 计算总训练步数
    total_steps = len(dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 记录训练参数
    logger.info("***** 训练参数 *****")
    for key, value in vars(args).items():
        logger.info(f"  {key} = {value}")
    
    # 训练循环
    logger.info("***** 开始训练 *****")
    global_step = 0
    tr_loss = 0.0
    model.train()
    
    for epoch in range(args.num_train_epochs):
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        for step, batch in enumerate(epoch_iterator):
            # 将批次移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            
            # 反向传播
            loss.backward()
            tr_loss += loss.item()
            
            # 梯度剪裁
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            
            # 更新进度条
            epoch_iterator.set_postfix({"loss": loss.item()})
            
            # 保存检查点
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # 保存模型
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
                
                # 保存 tokenizer
                tokenizer.save_pretrained(output_dir)
                
                # 保存训练参数
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info(f"保存模型检查点到 {output_dir}")
                
                # 记录训练损失
                avg_loss = tr_loss / args.save_steps
                logger.info(f"步骤 {global_step} 的平均损失: {avg_loss:.4f}")
                tr_loss = 0.0
    
    # 保存最终模型
    final_output_dir = os.path.join(args.output_dir, "final_model")
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), os.path.join(final_output_dir, "model.pt"))
    tokenizer.save_pretrained(final_output_dir)
    torch.save(args, os.path.join(final_output_dir, "training_args.bin"))
    
    logger.info(f"训练完成! 最终模型保存到 {final_output_dir}")
    return global_step, tr_loss / global_step

def main():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--data_dir", default="C:/Users/user/Desktop/RoBERTa_Model_Selfdesign/DataSet/cleaned_news_collection.csv", type=str, help="清理后的数据集路径")
    parser.add_argument("--output_dir", default="C:/Users/user/Desktop/RoBERTa_Model_Selfdesign/output", type=str, help="输出目录")
    parser.add_argument("--tokenizer_path", default="hfl/chinese-bert-wwm-ext", type=str, help="tokenizer路径")
    
    # 模型参数
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层大小")
    parser.add_argument("--num_heads", default=12, type=int, help="注意力头数")
    parser.add_argument("--intermediate_size", default=3072, type=int, help="中间层大小")
    parser.add_argument("--num_hidden_layers", default=12, type=int, help="隐藏层数量")
    parser.add_argument("--type_vocab_size", default=1, type=int, help="token类型词表大小")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout比例")
    
    # 训练参数
    parser.add_argument("--max_seq_length", default=512, type=int, help="最大序列长度")
    parser.add_argument("--mask_prob", default=0.15, type=float, help="MLM掩码比例")
    parser.add_argument("--batch_size", default=16, type=int, help="批次大小")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="训练轮数")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁剪最大范数")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="预热步数")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
    parser.add_argument("--num_workers", default=4, type=int, help="数据加载器工作进程数")
    parser.add_argument("--save_steps", default=5000, type=int, help="每多少步保存一次模型")
    parser.add_argument("--resume_from_checkpoint", default=None, type=str, help="从检查点恢复训练")
    
    args = parser.parse_args()
    
    # 开始训练
    global_step, avg_loss = train(args)
    logger.info(f"全局步数: {global_step}, 平均损失: {avg_loss:.4f}")

if __name__ == "__main__":
    main()