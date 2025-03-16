import os
import h5py
import numpy as np
from transformers import AutoTokenizer

class LotkaVolterraPreprocessor:
    """
    Implements LLMTIME-based preprocessing for time-series data.
    """

    FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lotka_volterra_data.h5"))

    def __init__(self, file_path=None, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the preprocessor and loads the dataset.

        Args:
            file_path (str, optional): Path to the HDF5 dataset. Defaults to FILE_PATH.
            model_name (str, optional): Name of the tokenizer model.
        """
        self.file_path = file_path or self.FILE_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.trajectories, self.time_points = self.load_dataset()

    def load_dataset(self):
        """Loads the Lotka-Volterra dataset from an HDF5 file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"🚨 Data file not found: {self.file_path}")

        with h5py.File(self.file_path, "r") as f:
            trajectories = f["trajectories"][:]  # Shape: (1000, 100, 2)
            time_points = f["time"][:]  # Shape: (100,)

        print(f"✅ Dataset loaded: {trajectories.shape[0]} samples, {trajectories.shape[1]} time steps")
        return trajectories, time_points

    def format_input(self, sample_index, num_steps=20):
        """
        Formats a time-series sample into LLMTIME structured text.

        Args:
            sample_index (int): Index of the sample.
            num_steps (int): Number of time steps.

        Returns:
            str: Formatted text sequence.
        """
        prey = self.trajectories[sample_index, :num_steps, 0]
        predator = self.trajectories[sample_index, :num_steps, 1]

        formatted_text = (
            "### Time-Series Prediction Task\n"
            f"Time-Series Data (Prey, Predator) for {num_steps} steps:\n"
            + "; ".join([f"({prey[i]:.2f}, {predator[i]:.2f})" for i in range(num_steps)]) +
            "\n\n### Predict the next 10 values:\n"
        )

        return formatted_text

    def tokenize_input(self, text):
        """
        Tokenizes the formatted text using the Qwen2.5 tokenizer.

        Args:
            text (str): Input text.

        Returns:
            dict: Tokenized representation (input_ids).
        """
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    def preprocess_sample(self, sample_index, num_steps=20):
        """
        Preprocesses a single sample.

        Args:
            sample_index (int): Index of the sample.
            num_steps (int): Number of time steps.

        Returns:
            tuple: (raw text, tokenized integers)
        """
        text = self.format_input(sample_index, num_steps)
        tokenized = self.tokenize_input(text)["input_ids"]
        return text, tokenized

if __name__ == "__main__":
    preprocessor = LotkaVolterraPreprocessor()

    # Preprocess and tokenize a sample
    sample_index = 0
    raw_text, tokenized_seq = preprocessor.preprocess_sample(sample_index)

    print("\n📝 Preprocessed Text:\n", raw_text)
    print("\n🔢 Tokenized Sequence (as integers):\n", tokenized_seq.tolist())
