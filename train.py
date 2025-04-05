"""
Training utilities for GPT-2 model
"""

import torch
from data import encode_text
from generate import generate


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate loss for a single batch
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten())

    return loss


def calc_loss(data_loader, model, device, num_batches=None):
    """
    Calculate average loss over multiple batches
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate model on train and validation data
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device, 
                tokenizer, config, start_text="Alone, wind howled"):
    """
    Train the GPT model
    """
    num_epochs = config["num_epochs"]
    eval_freq = config["eval_frequency"]
    eval_iter = config["eval_iterations"]
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    start_context = encode_text(start_text, tokenizer).to(device)

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

                # Generate sample text during training
                gen_output = generate(
                    model, start_context, 50, config["max_length"],
                    temperature=1.3
                )
                
                # Print a sample of the generated text
                sample_text = tokenizer.decode(gen_output[0, -50:].tolist())
                print(f"Sample: {sample_text}")

    return train_losses, val_losses, track_tokens_seen 