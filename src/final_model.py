import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from preprocessor import LLMTIMEPreprocessor
from load_qwen import load_qwen_model


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a frozen linear layer, adding trainable low-rank matrices A and B.
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
        return self.original_linear(x) + ((x @ self.A.T) @ self.B.T) * (self.alpha / self.r)


def process_sequences(texts, tokenizer, max_length=768, stride=384):
    """
    Tokenizes LLMTIME-encoded sequences and chunks them using a sliding window.
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


def evaluate_on_sample(model, tokenizer, preprocessor, prey, predator, num_steps=100, verbose=True):
    """
    Decodes a single sample's predictions and compares them with ground truth.
    """
    model.eval()
    with torch.no_grad():
        formatted_text, tokenized_seq, scale_factor = preprocessor.preprocess_sample(prey, predator, num_steps)
        input_ids = tokenized_seq.to(model.device)
        outputs = model.generate(input_ids=input_ids, max_new_tokens=0)
        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse and rescale decoded text
        decoded_pairs = [list(map(float, pair.split(','))) for pair in decoded_text.split(';') if ',' in pair]
        decoded_prey, decoded_predator = zip(*decoded_pairs) if decoded_pairs else ([], [])
        decoded_prey = np.array(decoded_prey) * scale_factor
        decoded_predator = np.array(decoded_predator) * scale_factor

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(prey, label="True Prey", linestyle='--', color='blue')
        plt.plot(decoded_prey, label="Decoded Prey", linestyle='-', color='blue')
        plt.plot(predator, label="True Predator", linestyle='--', color='red')
        plt.plot(decoded_predator, label="Decoded Predator", linestyle='-', color='red')
        plt.xlabel("Time Steps")
        plt.ylabel("Population")
        plt.legend()
        plt.title("True vs Decoded Population")
        plt.show()

        # Print metrics
        true_vals = np.concatenate((prey, predator))
        pred_vals = np.concatenate((decoded_prey, decoded_predator))
        mse = mean_squared_error(true_vals, pred_vals)
        mae = mean_absolute_error(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)
        if verbose:
            print(f"📊 MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return mse, mae, r2


def train_final_model(
    data_path: str = "../lotka_volterra_data.h5",
    lora_rank: int = 8,
    context_length: int = 768,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    max_steps: int = 4200,
    verbose: bool = True
):
    """
    Trains the final LoRA-tuned Qwen model and evaluates on validation and test set.
    """
    model, tokenizer = load_qwen_model()
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

    with h5py.File(data_path, 'r') as f:
        trajectories = f['trajectories'][:]
        prey = trajectories[:, :, 0]
        predator = trajectories[:, :, 1]

    # 70/20/10 split
    num_series = prey.shape[0]
    num_train = int(0.7 * num_series)
    num_val = int(0.2 * num_series)
    train_idx = np.arange(0, num_train)
    val_idx = np.arange(num_train, num_train + num_val)
    test_idx = np.arange(num_train + num_val, num_series)

    preprocessor = LLMTIMEPreprocessor()
    stride = context_length // 2

    def prep(indices):
        return [preprocessor.preprocess_sample(prey[i], predator[i], 100)[0] for i in indices]

    train_ids = process_sequences(prep(train_idx), tokenizer, context_length, stride)
    val_ids = process_sequences(prep(val_idx), tokenizer, context_length, context_length)
    test_ids = process_sequences(prep(test_idx), tokenizer, context_length, context_length)

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(train_ids), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_ids), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_ids), batch_size=batch_size)

    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)

    model.train()
    all_losses = []
    if verbose:
        bar = tqdm(total=max_steps, desc="Final LoRA Training")

    steps = 0
    while steps < max_steps:
        for (batch,) in train_loader:
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            all_losses.append(loss.item())
            steps += 1
            if verbose:
                bar.set_description(f"Step {steps}/{max_steps}")
                bar.set_postfix(loss=loss.item())
            if steps >= max_steps:
                break

    # Validation Loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for (batch,) in val_loader:
            outputs = model(batch, labels=batch)
            val_losses.append(outputs.loss.item())
    avg_val_loss = float(np.mean(val_losses))
    print(f"📉 Final validation loss: {avg_val_loss:.4f}")

    # Test Evaluation
    print("🧪 Running test set evaluation...")
    test_preds, test_targets = [], []
    with torch.no_grad():
        for (batch,) in test_loader:
            output = model(batch, labels=batch)
            test_preds.extend(output.logits.argmax(dim=-1).cpu().numpy())
            test_targets.extend(batch.cpu().numpy())

    test_preds = np.array(test_preds).flatten()
    test_targets = np.array(test_targets).flatten()
    mse = mean_squared_error(test_targets, test_preds)
    mae = mean_absolute_error(test_targets, test_preds)
    r2 = r2_score(test_targets, test_preds)
    print(f"✅ Test MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Loss vs. Training Steps
    plt.plot(all_losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.show()

    # Visualise decoded predictions on sample
    print("\n🔍 Visualising decoded predictions on sample 0...")
    evaluate_on_sample(model, tokenizer, preprocessor, prey[test_idx[0]], predator[test_idx[0]])

    return model


if __name__ == "__main__":
    train_final_model()
