"""
Main script for GPT-2 model training and generation
"""

import argparse
import torch
import tiktoken
from model import GPT
from config import GPT2_SMALL_CONFIG, TRAINING_CONFIG, GENERATION_CONFIG
from data import prepare_dataloaders, encode_text
from train import train_model
from generate import generate_text
from utils import download_shakespeare, load_corpus, save_model, load_model, get_device


def train(args):
    """Train a GPT-2 model"""
    device = get_device()
    
    # Load corpus
    corpus_file = download_shakespeare()
    corpus = load_corpus(corpus_file)
    print(f"Loaded corpus with {len(corpus)} characters")
    
    # Prepare data loaders
    train_loader, val_loader = prepare_dataloaders(corpus, TRAINING_CONFIG)
    print(f"Created data loaders with {len(train_loader)} training batches")
    
    # Initialize model and optimizer
    model = GPT(GPT2_SMALL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"])
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Train the model
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device, tokenizer, 
        TRAINING_CONFIG, start_text=args.prompt
    )
    
    # Save the model
    save_model(model, args.save_path)
    
    # Generate sample text
    sample = generate_text(
        model, args.prompt, 
        max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
        temperature=GENERATION_CONFIG["temperature"],
        device=device
    )
    print(f"\nGenerated sample:\n{sample}")


def generate(args):
    """Generate text with a GPT-2 model"""
    device = get_device()
    
    # Load model
    model = load_model(GPT, GPT2_SMALL_CONFIG, args.model_path, device)
    
    # Generate text
    sample = generate_text(
        model, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print(f"\nGenerated text:\n{sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-2 training and text generation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train a GPT-2 model")
    train_parser.add_argument("--prompt", type=str, default="Alone, wind howled", 
                             help="Prompt for generation samples during training")
    train_parser.add_argument("--save-path", type=str, default="gpt2-small.pth",
                             help="Path to save the trained model")
    
    # Generation arguments
    gen_parser = subparsers.add_parser("generate", help="Generate text with GPT-2")
    gen_parser.add_argument("--model-path", type=str, default="gpt2-small.pth",
                           help="Path to the model file")
    gen_parser.add_argument("--prompt", type=str, default="Alone, wind howled",
                           help="Text prompt to start generation")
    gen_parser.add_argument("--max-tokens", type=int, default=100,
                           help="Maximum number of tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=1.0,
                           help="Temperature for sampling (higher = more random)")
    gen_parser.add_argument("--top-k", type=int, default=None,
                           help="Sample from top-k most likely tokens (None = all)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    else:
        # Default to showing a simple demo if no arguments provided
        device = get_device()
        
        # Create a small model for demo
        config = GPT2_SMALL_CONFIG.copy()
        config["n_layers"] = 4  # Use fewer layers for quick demo
        model = GPT(config).to(device)
        
        # Create a sample input
        x = torch.randint(
            0, config["vocab_size"],
            (2, config["context_length"]),
            dtype=torch.long, device=device
        )
        
        # Run inference
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output logits shape: {output.shape}")
        print("\nUse 'python main.py train' to train the model")
        print("Use 'python main.py generate --prompt \"Your text here\"' to generate text")
