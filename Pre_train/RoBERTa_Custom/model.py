import torch
import torch.nn as nn
from .embeddings import RobertaEmbeddings
from .encoder import RobertaEncoder

class RobertaModel(nn.Module):
    """RoBERTa 基礎模型"""
    def __init__(
        self, 
        vocab_size, 
        max_position_embeddings, 
        hidden_size=768, 
        num_heads=12, 
        intermediate_size=3072, 
        num_hidden_layers=12, 
        type_vocab_size=1,
        dropout=0.1
    ):
        super().__init__()
        self.embeddings = RobertaEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout
        )
        self.encoder = RobertaEncoder(
            num_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True
    ):
        # 處理嵌入
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # 編碼器處理
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0]  # [CLS] token
        
        outputs = (sequence_output, pooled_output)
        if output_hidden_states:
            outputs += (encoder_outputs[1],)
        if output_attentions:
            outputs += (encoder_outputs[-1],)
            
        if return_dict:
            return {
                "last_hidden_state": sequence_output,
                "pooler_output": pooled_output,
                "hidden_states": encoder_outputs[1] if output_hidden_states else None,
                "attentions": encoder_outputs[-1] if output_attentions else None
            }
        else:
            return outputs


class RobertaForMaskedLM(nn.Module):
    """用於掩碼語言模型的 RoBERTa 模型"""
    def __init__(
        self,
        vocab_size,
        max_position_embeddings,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        num_hidden_layers=12,
        type_vocab_size=1,
        dropout=0.1
    ):
        super().__init__()
        
        self.roberta = RobertaModel(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            dropout=dropout
        )
        
        self.lm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-5),
            nn.Linear(hidden_size, vocab_size)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,  # 添加 labels 參數
        output_hidden_states=False,
        output_attentions=False
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        sequence_output = outputs["last_hidden_state"]
        prediction_scores = self.lm_head(sequence_output)
        
        loss = None
        if labels is not None:
            # 計算損失，忽略標籤為 -100 的位置
            loss = self.loss_fn(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": prediction_scores,
            "hidden_states": outputs.get("hidden_states", None),
            "attentions": outputs.get("attentions", None)
        }


class RobertaForSequenceClassification(nn.Module):
    """用於序列分類的 RoBERTa 模型"""
    def __init__(
        self,
        vocab_size,
        max_position_embeddings,
        num_labels=2,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        num_hidden_layers=12,
        type_vocab_size=1,
        dropout=0.1,
        classifier_dropout=None
    ):
        super().__init__()
        self.num_labels = num_labels
        
        self.roberta = RobertaModel(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=type_vocab_size,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(classifier_dropout if classifier_dropout is not None else dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,  # 添加 labels 參數
        output_hidden_states=False,
        output_attentions=False
    ):
        # 獲取 RoBERTa 模型輸出
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        pooled_output = outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # 計算損失
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states", None),
            "attentions": outputs.get("attentions", None)
        }