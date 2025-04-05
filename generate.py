"""
Text generation utilities for GPT-2 model
"""

import torch
from data import encode_text, decode_tokens
import tiktoken


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    Generate text using the GPT-2 model
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_text(model, prompt, max_new_tokens=50, context_size=1024,
                 temperature=1.0, top_k=None, eos_id=None, device='cpu'):
    """
    Generate text from a prompt string
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_prompt = encode_text(prompt, tokenizer).to(device)
    
    model.to(device)
    output_tokens = generate(
        model, encoded_prompt, max_new_tokens, context_size,
        temperature, top_k, eos_id
    )
    
    return decode_tokens(output_tokens, tokenizer) 