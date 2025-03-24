import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from load_qwen import load_qwen_model
from preprocessor import LLMTIMEPreprocessor


class TrajectoryDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajectories = self.load_data()

    def load_data(self):
        with h5py.File(self.file_path, "r") as f:
            return f["trajectories"][:]

    def get_random_systems(self, num_samples=5, seed=42):
        np.random.seed(seed)
        indices = np.random.choice(len(self.trajectories), num_samples, replace=False)
        return [self.trajectories[i] for i in indices]


class PredictionPipeline:
    def __init__(self, dataset: TrajectoryDataset, input_fraction=0.7):
        self.model, self.tokenizer = load_qwen_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.dataset = dataset
        self.input_fraction = input_fraction

    def trunc_string(self, series, n):
        """Convert first n timesteps to formatted LLMTIME string."""
        truncated = series[:n]
        return ";".join([f"{p:.2f},{q:.2f}" for p, q in truncated])

    def _predict_on_series(self, series):
        input_timesteps = int(self.input_fraction * 100)
        output_timesteps = 100 - input_timesteps

        input_str = self.trunc_string(series, n=input_timesteps)
        tokens = self.tokenizer(input_str, return_tensors='pt')
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Generate predictions for remaining timesteps
        token_preds = self.model.generate(
            tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            max_new_tokens=int(output_timesteps * 10)  # 10 tokens per timestep
        )

        semicolons = (token_preds[0] == 26).nonzero(as_tuple=True)[0]  # ASCII 26 is semicolon
        while len(semicolons) < 100:
            timesteps_needed = 100 - len(semicolons)
            token_preds = self.model.generate(
                token_preds,
                max_new_tokens=int(timesteps_needed * 10 + 10)
            )
            semicolons = (token_preds[0] == 26).nonzero(as_tuple=True)[0]
            if len(token_preds[0]) > 2000:
                print("⚠️ Reached max token length")
                break

        if len(semicolons) >= 100:
            end_idx = semicolons[99].item()
            tokens_1d = token_preds[0][:end_idx]
        else:
            tokens_1d = token_preds[0]

        decoded = self.tokenizer.decode(tokens_1d, skip_special_tokens=True)
        return decoded

    def predict(self, num_tests=5, seed=240901):
        series_list = self.dataset.get_random_systems(num_tests, seed)
        predictions = np.empty(num_tests, dtype=object)

        for idx, system in enumerate(series_list):
            print(f"\n🔹 Test {idx+1} of {num_tests}")
            predictions[idx] = self._predict_on_series(system)

        return predictions, series_list

    def predict_by_index(self, index):
        if index < 0 or index >= len(self.dataset.trajectories):
            raise IndexError(f"Index {index} is out of bounds")

        system = self.dataset.trajectories[index]
        prediction = self._predict_on_series(system)
        return [prediction], [system]  # Return in list form for easy plotting

    def plot_predictions(self, predictions, original_series):
        for i in range(len(predictions)):
            true_vals = original_series[i]
            pred_vals = [list(map(float, pair.split(','))) for pair in predictions[i].split(';') if ',' in pair]

            if not pred_vals:
                print(f"⚠️ No valid predictions for Test {i+1}")
                continue

            true_prey = [x[0] for x in true_vals]
            true_predator = [x[1] for x in true_vals]
            pred_prey = [x[0] for x in pred_vals]
            pred_predator = [x[1] for x in pred_vals]

            min_len = min(len(true_prey), len(pred_prey))
            true_prey, pred_prey = true_prey[:min_len], pred_prey[:min_len]
            true_predator, pred_predator = true_predator[:min_len], pred_predator[:min_len]

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(true_prey, label="True Prey", marker="o", color="blue")
            plt.plot(pred_prey, label="Predicted Prey", linestyle="--", marker="x", color="cyan")
            plt.title(f"Prey - Test {i+1}")
            plt.xlabel("Time Step")
            plt.ylabel("Population")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(true_predator, label="True Predator", marker="o", color="red")
            plt.plot(pred_predator, label="Predicted Predator", linestyle="--", marker="x", color="orange")
            plt.title(f"Predator - Test {i+1}")
            plt.xlabel("Time Step")
            plt.ylabel("Population")
            plt.legend()

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    import sys

    dataset = TrajectoryDataset("lotka_volterra_data.h5")
    pipeline = PredictionPipeline(dataset, input_fraction=0.5)  # Use only first half for prediction

    if len(sys.argv) > 1:
        index = int(sys.argv[1])
        print(f"🔎 Running prediction for trajectory index {index}")
        predictions, series = pipeline.predict_by_index(index)
    else:
        predictions, series = pipeline.predict(num_tests=5)

    pipeline.plot_predictions(predictions, series)
