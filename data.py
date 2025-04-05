"""
Dataset and dataloader for GPT-2 model
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(corpus, batch_size=4, max_length=256,
                      stride=128, shuffle=True, drop_last=True,
                      num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(corpus, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


def prepare_dataloaders(corpus, config):
    train_size = int(0.9 * len(corpus))
    train_text, val_text = corpus[:train_size], corpus[train_size:]

    train_loader = create_dataloader(
        train_text,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        stride=config["stride"],
        drop_last=config["drop_last"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"]
    )
    
    val_loader = create_dataloader(
        val_text,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        stride=config["stride"],
        drop_last=False,
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    return train_loader, val_loader


def encode_text(text, tokenizer=None):
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def decode_tokens(token_ids, tokenizer=None):
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist()) 