"""
Utility functions for the GPT-2 model
"""

import os
import requests
import torch


def download_shakespeare():
    """
    Download the Shakespeare dataset if it doesn't exist
    """
    filename = "input.txt"
    if not os.path.exists(filename):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Shakespeare dataset from {url}")
        r = requests.get(url)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(r.text)
    
    return filename


def load_corpus(filename):
    """
    Load a text corpus from a file
    """
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def save_model(model, filename):
    """
    Save model state to a file
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(model_class, config, filename, device='cpu'):
    """
    Load a model from a saved state dict
    """
    model = model_class(config).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    print(f"Model loaded from {filename}")
    return model


def get_device():
    """
    Get the appropriate device for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    
    return device 