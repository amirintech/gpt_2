"""
GPT2 configuration settings
"""

# Default configuration for GPT2 model
GPT2_SMALL_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,       # Max sequence length (n_positions)
    "embedding_dim": 768,         # Hidden size (n_embd)
    "n_attention_heads": 12,      # Number of attention heads
    "n_layers": 12,               # Number of transformer blocks
    "dropout_rate": 0.1,
    "qkv_bias": True
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 2,
    "max_length": 1024,
    "stride": 1024,
    "shuffle": True,
    "drop_last": True,
    "num_workers": 0,
    "learning_rate": 5e-4,
    "num_epochs": 5,
    "eval_frequency": 100,
    "eval_iterations": 100
}

# Generation configuration
GENERATION_CONFIG = {
    "max_new_tokens": 50,
    "temperature": 1.3,
    "top_k": None
} 