# mlm_loss.py
"""
MLM 损失计算模块
可用于自定义 RoBERTa 模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLMHead(nn.Module):
    """
    MLM 预测头
    用于预测被 mask 的 token
    """
    def __init__(self, hidden_size, vocab_size, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        
        # 用于初始化
        self.decoder.bias = self.bias
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits

class MLMLoss(nn.Module):
    """
    MLM 损失计算
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        """
        计算 MLM 损失
        
        Args:
            logits: 模型预测输出，形状为 [batch_size, seq_len, vocab_size]
            labels: 标签，形状为 [batch_size, seq_len]，其中非 mask 位置为 -100
            
        Returns:
            损失值
        """
        # 获取 batch_size 和 seq_len
        batch_size, seq_len, vocab_size = logits.size()
        
        # 将预测结果和标签变形为 2D 用于 CrossEntropyLoss
        # logits: [batch_size * seq_len, vocab_size]
        logits = logits.view(-1, vocab_size)
        
        # labels: [batch_size * seq_len]
        labels = labels.view(-1)
        
        # 计算损失，CrossEntropyLoss 会自动忽略标签为 -100 的位置
        loss = self.loss_fn(logits, labels)
        
        return loss

# 测试代码
if __name__ == "__main__":
    # 模拟数据
    hidden_size = 768
    vocab_size = 30000
    batch_size = 2
    seq_len = 512
    
    # 随机生成 hidden_states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 随机生成标签，部分位置设为 -100
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.rand(batch_size, seq_len) > 0.85  # 假设约 15% 位置被 mask
    labels[~mask] = -100
    
    # 测试 MLMHead
    mlm_head = MLMHead(hidden_size, vocab_size)
    logits = mlm_head(hidden_states)
    print(f"MLMHead 输出形状: {logits.shape}")
    
    # 测试 MLMLoss
    mlm_loss = MLMLoss()
    loss = mlm_loss(logits, labels)
    print(f"MLM 损失值: {loss.item()}")