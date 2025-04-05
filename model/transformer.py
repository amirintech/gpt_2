"""
Transformer block for the GPT-2 model
"""

import torch.nn as nn
from model.attention import MultiHeadAttention
from model.layers import LayerNorm, FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(
            cfg["embedding_dim"],
            cfg["embedding_dim"],
            cfg["context_length"],
            cfg["n_attention_heads"],
            cfg["dropout_rate"],
            cfg["qkv_bias"],
        )
        self.feed_forward = FeedForward(cfg["embedding_dim"])
        self.layer_norm1 = LayerNorm(cfg["embedding_dim"])
        self.layer_norm2 = LayerNorm(cfg["embedding_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = shortcut + x

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = shortcut + x

        return x 