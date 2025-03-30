from typing import Tuple
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator

from preprocessor import LLMTIMEPreprocessor
from load_qwen import load_qwen_model


class LoRALinear(nn.Module):
    """
    Implements a LoRA-adapted Linear layer, wrapping an existing Linear module.
    Only the low-rank A and B matrices are trainable.
    """
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r

        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))

        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)


def process_sequences(texts, tokenizer, max_length=512, stride=256):
    """
    Tokenizes and chunks text sequences using a sliding window.

    Args:
        texts (list[str]): LLMTIME-formatted time series strings.
        tokenizer: Huggingface tokenizer.
        max_length (int): Max context length for the model.
        stride (int): Overlap size between chunks.

    Returns:
        torch.Tensor: Chunked token ID tensors of shape [num_chunks, max_length].
    """
    all_input_ids = []
    for text in texts:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        seq_ids = encoding.input_ids[0]
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i: i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat([
                    chunk,
                    torch.full((max_length - len(chunk),), tokenizer.pad_token_id)
                ])
            all_input_ids.append(chunk)
    return torch.stack(all_input_ids)


def train_lora_model(
    data_path: str = "../lotka_volterra_data.h5",
    lora_rank: int = 4,
    context_length: int = 512,
    stride: int = None,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    max_steps: int = 1000,
    input_fraction: float = 0.7,
    verbose: bool = True
) -> Tuple[nn.Module, float]:
    """
    Fine-tunes the Qwen2.5-0.5B-Instruct model using LoRA on LLMTIME-encoded Lotka-Volterra data.

    After training, the model is evaluated on a held-out validation set.

    Args:
        data_path (str): Path to .h5 file with 'trajectories' dataset of shape [num_series, 100, 2].
        lora_rank (int): LoRA rank for adaptation matrices.
        context_length (int): Max number of tokens per training chunk.
        stride (int): Chunk overlap; defaults to context_length // 2.
        learning_rate (float): Optimizer learning rate.
        batch_size (int): Training batch size.
        max_steps (int): Max number of optimizer steps.
        input_fraction (float): Fraction of time series used for training (validation = 0.2).
        verbose (bool): Whether to display progress.

    Returns:
        model: The fine-tuned model (in eval mode).
        avg_val_loss (float): The average loss on the validation set.
    """
    model, tokenizer = load_qwen_model()

    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

    with h5py.File(data_path, 'r') as f:
        trajectories = f['trajectories'][:]  # [num_series, 100, 2]
        prey = trajectories[:, :, 0]
        predator = trajectories[:, :, 1]

    num_series = prey.shape[0]
    num_train = int(input_fraction * num_series)
    num_val = int(0.2 * num_series)

    train_indices = np.arange(0, num_train)
    val_indices = np.arange(num_train, num_train + num_val)

    preprocessor = LLMTIMEPreprocessor()

    def prepare_texts(indices):
        texts = []
        for i in indices:
            text, _, _ = preprocessor.preprocess_sample(prey[i], predator[i], num_steps=100)
            texts.append(text)
        return texts

    stride = stride or (context_length // 2)
    train_input_ids = process_sequences(prepare_texts(train_indices), tokenizer, context_length, stride)
    val_input_ids = process_sequences(prepare_texts(val_indices), tokenizer, context_length, context_length)

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(train_input_ids), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_input_ids), batch_size=batch_size, shuffle=False)

    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    model.train()
    steps = 0
    if verbose:
        progress_bar = tqdm(total=max_steps, desc="LoRA Training")
    while steps < max_steps:
        for (batch,) in train_loader:
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()

            steps += 1
            if verbose:
                progress_bar.set_description(f"Step {steps}/{max_steps}")
                progress_bar.set_postfix(loss=loss.item())
            if steps >= max_steps:
                break

    model.eval()
    if verbose:
        print(" Training complete. Running validation...")

    val_losses = []
    with torch.no_grad():
        for (batch,) in val_loader:
            outputs = model(batch, labels=batch)
            val_losses.append(outputs.loss.item())

    avg_val_loss = float(np.mean(val_losses))
    if verbose:
        print(f" Average validation loss: {avg_val_loss:.4f}")

    return model, avg_val_loss

