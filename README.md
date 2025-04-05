# GPT-2 Implementation

A modular implementation of GPT-2 in PyTorch for training and text generation.

## Project Structure

```
.
├── config.py          # Configuration settings
├── data.py            # Dataset and dataloader
├── generate.py        # Text generation functions
├── main.py            # Main script for training and generation
├── model              # Model components
│   ├── __init__.py    # Package initializer
│   ├── attention.py   # Attention mechanism
│   ├── gpt.py         # Main GPT model
│   ├── layers.py      # Basic layers (LayerNorm, GELU, etc.)
│   └── transformer.py # Transformer block
├── requirements.txt   # Dependencies
├── train.py           # Training functions
├── utils.py           # Utility functions
└── playground.ipynb   # Playground jupyter notebook for prototyping
```

### Basic Demo

To run a simple demo that shows model shapes:

```bash
python main.py
```

### Training

To train the model on the Shakespeare dataset:

```bash
python main.py train
```

Optional arguments:

- `--prompt`: Text prompt for generation samples during training (default: "Alone, wind howled")
- `--save-path`: Path to save the trained model (default: "gpt2-small.pth")

### Text Generation

To generate text with a trained model:

```bash
python main.py generate --prompt "Your text here"
```

Optional arguments:

- `--model-path`: Path to the model file (default: "gpt2-small.pth")
- `--prompt`: Text prompt to start generation (default: "Alone, wind howled")
- `--max-tokens`: Maximum number of tokens to generate (default: 100)
- `--temperature`: Temperature for sampling; higher values produce more random output (default: 1.0)
- `--top-k`: Sample from top-k most likely tokens; if None, sample from all tokens (default: None)

## Example

```bash
# Train a model
python main.py train --save-path my_model.pth

# Generate text
python main.py generate --model-path my_model.pth --prompt "To be or not to be" --temperature 1.2
```
