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
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


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


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, context_length, n_heads, dropout, qkv_bias):
        super(MultiHeadAttention, self).__init__()

        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"

        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.query_weights = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.key_weights = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.value_weights = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.out_weights = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
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
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, context_length, self.out_dim)
        )
        context_vec = self.out_weights(context_vec)

        return context_vec


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


class GPT(nn.Module):
    def __init__(self, cfg):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = nn.Embedding(
            cfg["context_length"], cfg["embedding_dim"]
        )
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.layer_norm = LayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"])

    def forward(self, x):
        _, context_length = x.shape
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(
            torch.arange(context_length, dtype=torch.long, device=x.device)
        )
        x = token_embeddings + position_embeddings

        x = self.dropout(x)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.out_head(x)
        return logits
