# Simple GPT

This repository contains a single Python script implementing a basic Generative Pre-trained Transformer (GPT) model using PyTorch, inspired by Andrej Karpathy's nanoGPT tutorial.  The model trains on character-level text data from the Tiny Shakespeare dataset and generates new text samples after training.

## Features

- Character-level tokenization with encoder/decoder functions for simple text processing.
- Multi-head self-attention mechanism with causal masking for autoregressive generation.
- Transformer blocks including feed-forward layers and layer normalization.
- Built-in training loop with evaluation on train/validation splits.
- Text generation function using multinomial sampling from softmax probabilities.

The implementation focuses on educational simplicity, using hyperparameters like 6 layers, 6 heads, and 384 embedding dimensions.

## Requirements

- Python 3.8 or higher.
- PyTorch (supports CUDA if available).
- No additional libraries beyond PyTorch's core modules (nn, functional).

Install PyTorch via pip, for example:

```bash
pip install torch
```

The script uses standard PyTorch features and assumes the input data file is present.

## Setup and Usage

1. Clone the repository:

```bash
git clone https://github.com/SaurabhBagade/simple-gpt.git
cd simple-gpt
```

2. Download the Tiny Shakespeare dataset and save it as `input.txt` in the root directory. You can fetch it with:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

The script directly loads this file for training.

3. Install PyTorch if not already installed:

```bash
pip install torch
```

4. Run the training and generation:

```bash
python gpt.py
```

This executes the full script: loads data, initializes the model, trains for 5000 iterations with periodic evaluation, and finally generates and prints 500 characters of new text starting from an empty context.  Training uses batch size 64, block size 256, learning rate 3e-4, and AdamW optimizer.  Progress shows train and validation loss every 500 steps.

To customize, edit hyperparameters at the top of `gpt.py` or modify the generation call at the end.  The model moves to GPU if available; otherwise, it uses CPU.

## Project Structure

```
simple-gpt/
├── gpt.py         # Main script: model definition, training, and generation
└── input.txt      # Input dataset (Tiny Shakespeare text file)
```

The repository consists of just these files, making it easy to run standalone without complex setup.

## Extending the Model

To add features like saving checkpoints or interactive prompts, modify the training loop or generation function in `gpt.py`. For example, add `torch.save(model.state_dict(), 'model.pth')` after training.  The `generate` method supports custom prompts by passing an initial index tensor.

## Contributing

Fork the repository, make changes to `gpt.py`, and submit a pull request. Focus on improvements like adding BPE tokenization or visualization tools.

## License

MIT License. See the LICENSE file or add one if absent.

## Acknowledgments

Based on the nanoGPT implementation by Andrej Karpathy. Refer to the "Let's build GPT" video tutorial for deeper insights.

https://www.youtube.com/watch?v=kCc8FmEb1nY