import torch
import torch.nn as nn
import math

class RobertaSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.head_dim
        
        # 分別為 Q、K、V 創建線性層可以更清晰（雖然合併也可以）
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        batch_size, seq_length = hidden_states.size(0), hidden_states.size(1)
        
        # 計算 Q, K, V 投影
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 計算注意力分數
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.scale
        
        # 應用注意力遮罩（如果提供）
        if attention_mask is not None:
            # 確保 attention_mask 有正確的形狀: [batch_size, 1, 1, seq_length]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
                
            attention_scores = attention_scores + attention_mask
        
        # 正規化注意力分數
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # 計算上下文層
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # 輸出投影
        output = self.out_proj(context_layer)
        output = self.proj_dropout(output)
        
        if output_attentions:
            return output, attention_probs
        else:
            return output