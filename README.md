# Simple AI AutoMod

This is a simple tool for automatic live chat moderation using AI models.

It is trained using an online large dataset with sample chat messages and their coresponding safety levels.

This model takes an arbitrary string input (chat messages) and outputs a boolean flag for whether the message should be banned.

## How to use

### 1. Install dependencies
```shell
pip3 install -r requirements.txt
pip3 install huggingface_hub
mkdir -p ./local/base-model
mkdir -p ./local/model
```

### 2. Adjust environment
Change the line `device = torch.device("cpu")` to `device = torch.device("cuda")` in `test.py` and `train.py` if the device running the model has NVIDIA GPU with CUDA support.

### 3. Download dataset
Run the script `dataset.py` with `python3`.

### 4. Download base model for training.
```shell
hf download distilbert/distilbert-base-uncased --local-dir ./local/base-model
```

### 4. Train the model
Run `train.py` to train the model. This can take a while.

### 5. Test and use the model
Run `test.py` for testing the model with a few simple prompts. You can adapt the `is_safe()` function in `test.py` to use the model in practical cases.

# License
MIT License. See `LICENSE.md` for more details.
