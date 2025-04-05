"""
Attention mechanism for the GPT-2 model
"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, context_length, n_heads, dropout, qkv_bias):
        super(MultiHeadAttention, self).__init__()

        assert(out_dim % n_heads == 0), "out_dim must be divisible by n_heads"

        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.query_weights = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.key_weights = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.value_weights = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.out_weights = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, context_length, in_dim = x.shape

        queries = self.query_weights(x)
        keys = self.key_weights(x)
        values = self.value_weights(x)

        queries = queries.view(batch_size, context_length, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, context_length, self.n_heads, self.head_dim)
        values = values.view(batch_size, context_length, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = queries @ keys.transpose(-2, -1)
        mask_bool = self.mask.bool()[:context_length, :context_length]
        scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(scores / (keys.shape[-1] ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = attention_weights @ values
        context_vec = context_vec.transpose(1, 2).contiguous().view(
            batch_size, context_length, self.out_dim
        )
        context_vec = self.out_weights(context_vec)

        return context_vec 