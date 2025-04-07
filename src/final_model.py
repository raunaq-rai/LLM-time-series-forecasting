import math
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator
import matplotlib.pyplot as plt

from load_qwen import load_qwen_model
from preprocessor import LLMTIMEPreprocessor


class LoRALinear(nn.Module):
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


def process_sequences(texts, tokenizer, max_length=768, stride=384):
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


def evaluate(model, dataloader, accelerator):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for (batch,) in dataloader:
            outputs = model(batch, labels=batch)
            val_losses.append(outputs.loss.item())
    avg_loss = float(np.mean(val_losses))
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def train_lora_model(
    data_path="../lotka_volterra_data.h5",
    context_length=768,
    lora_rank=8,
    learning_rate=1e-4,
    batch_size=4,
    max_steps=5000,
    input_fraction=0.7,
    val_fraction=0.2,
    verbose=True
):
    model, tokenizer = load_qwen_model()

    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

    with h5py.File(data_path, "r") as f:
        trajectories = f["trajectories"][:]
    prey = trajectories[:, :, 0]
    predator = trajectories[:, :, 1]

    num_series = prey.shape[0]
    num_train = int(input_fraction * num_series)
    num_val = int(val_fraction * num_series)

    train_indices = np.arange(0, num_train)
    val_indices = np.arange(num_train, num_train + num_val)

    preprocessor = LLMTIMEPreprocessor()
    def prepare_texts(indices):
        texts = []
        for i in indices:
            text, _, _ = preprocessor.preprocess_sample(prey[i], predator[i], num_steps=100)
            texts.append(text)
        return texts

    train_texts = prepare_texts(train_indices)
    val_texts = prepare_texts(val_indices)

    train_input_ids = process_sequences(train_texts, tokenizer, max_length=context_length, stride=context_length // 2)
    val_input_ids = process_sequences(val_texts, tokenizer, max_length=context_length, stride=context_length)

    train_loader = DataLoader(TensorDataset(train_input_ids), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_input_ids), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    model.train()
    steps = 0
    train_losses = []
    step_counts = []

    if verbose:
        progress_bar = tqdm(total=max_steps, desc="Training")

    while steps < max_steps:
        for (batch,) in train_loader:
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            steps += 1

            train_losses.append(loss.item())
            step_counts.append(steps)

            if verbose:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
            if steps >= max_steps:
                break

    if verbose:
        print("\n Training complete. Evaluating...")

    avg_loss, perplexity = evaluate(model, val_loader, accelerator)

    if verbose:
        print(f" Validation loss: {avg_loss:.4f}")
        print(f" Perplexity: {perplexity:.2f}")

        
        plt.figure(figsize=(10, 4))
        plt.plot(step_counts, train_losses, label="Training Loss", color="blue")
        plt.axhline(y=avg_loss, color="red", linestyle="--", label=f"Validation Loss = {avg_loss:.4f}")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss (Full Range)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    
        zoom_steps = [s for s in step_counts if s >= 2500]
        zoom_losses = train_losses[-len(zoom_steps):]

        plt.figure(figsize=(10, 4))
        plt.plot(zoom_steps, zoom_losses, label="Training Loss (Zoomed)", color="purple")
        plt.axhline(y=avg_loss, color="red", linestyle="--", label=f"Validation Loss = {avg_loss:.4f}")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss (Steps 2500–5000)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    return model, tokenizer, val_loader, avg_loss, perplexity