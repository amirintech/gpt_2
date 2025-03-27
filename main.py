import torch
from gpt_model import GPT

GPT2_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 768,
    "n_attention_heads": 12,
    "n_layers": 12,
    "dropout_rate": 0.1,
    "qkv_bias": False,
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randint(
    0,
    GPT2_CONFIG["vocab_size"],
    (2, GPT2_CONFIG["context_length"]),
    dtype=torch.long,
    device=device,
)
x.shape

model = GPT(GPT2_CONFIG).to(device)

print(model(x).shape)
