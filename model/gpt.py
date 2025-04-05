"""
GPT-2 model implementation
"""

import torch
import torch.nn as nn
from model.layers import LayerNorm
from model.transformer import TransformerBlock


class GPT(nn.Module):
    def __init__(self, cfg):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.layer_norm = LayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"])

    def forward(self, x):
        batch_size, context_length = x.shape
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(
            torch.arange(context_length, dtype=torch.long, device=x.device))
        x = token_embeddings + position_embeddings

        x = self.dropout(x)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.out_head(x)
        return logits 