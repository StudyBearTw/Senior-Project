import torch
import torch.nn as nn
from .attention import RobertaSelfAttention

class RobertaEncoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super().__init__()
        # 自注意力部分
        self.attention = RobertaSelfAttention(hidden_size, num_heads, dropout)
        self.attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attention_dropout = nn.Dropout(dropout)
        
        # 前饋網絡部分
        self.ffn_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 自注意力部分 (使用 Pre-LayerNorm)
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        
        if output_attentions:
            attention_output, attention_weights = self.attention(
                hidden_states, 
                attention_mask=attention_mask,
                output_attentions=True
            )
            hidden_states = residual + self.attention_dropout(attention_output)
        else:
            attention_output = self.attention(
                hidden_states, 
                attention_mask=attention_mask
            )
            hidden_states = residual + self.attention_dropout(attention_output)
        
        # 前饋網絡部分 (使用 Pre-LayerNorm)
        residual = hidden_states
        hidden_states = self.ffn_layernorm(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs

class RobertaEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RobertaEncoderLayer(hidden_size, intermediate_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=1e-5)
        
    def forward(
        self, 
        hidden_states, 
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                
        # 最後再進行一次層正規化
        hidden_states = self.final_layernorm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
            
        return outputs