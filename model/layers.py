"""
Basic building blocks for the GPT-2 model
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized_x + self.shift


class GELU(nn.Module):
    def __init__(self) -> None:
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) *
                                         (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.layers(x) 