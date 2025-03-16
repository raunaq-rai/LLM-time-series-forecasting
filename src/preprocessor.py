import h5py
import numpy as np
from transformers import AutoTokenizer

class LLMTIMEPreprocessor:
    def __init__(self, file_path="lotka_volterra_data.h5", decimal_places=2, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the preprocessor with scaling factor and decimal precision.
        Args:
            decimal_places (int): Precision for rounding.
            model_name (str): Qwen2.5 tokenizer model.
        """
        self.decimal_places = decimal_places
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the dataset to determine an appropriate scale factor
        data, _ = self.load_data(file_path)
        self.scale_factor = self.compute_scale_factor(data)

    def load_data(self, file_path="lotka_volterra_data.h5"):
        """
        Loads the predator-prey dataset.
        Args:
            file_path (str): Path to the dataset.
        Returns:
            np.array: Scaled and rounded time series.
        """
        with h5py.File(file_path, "r") as f:
            trajectories = f["trajectories"][:]  # Shape (1000, 100, 2)
            time_points = f["time"][:]  # Shape (100,)

        return trajectories, time_points

    def compute_scale_factor(self, data, target_range=(0, 10)):
        """
        Determines the best scaling factor for the dataset.
        Ensures values are within the 0–10 range.
        """
        min_val, max_val = np.min(data), np.max(data)
        scale_factor = target_range[1] / max(abs(min_val), abs(max_val))  # Scale to 0-10 range
        return scale_factor

    def scale_and_format(self, data):
        """
        Scales and formats the time series into LLMTIME-friendly strings.
        Args:
            data (np.array): Time series data.
        Returns:
            list: Formatted sequences.
        """
        scaled_data = np.round(data * self.scale_factor, self.decimal_places)
        formatted_sequences = []
        
        for sequence in scaled_data:
            formatted_seq = ";".join([",".join(map(str, timestep)) for timestep in sequence])
            formatted_sequences.append(formatted_seq)
        
        return formatted_sequences

    def tokenize(self, sequences):
        """
        Tokenizes the formatted sequences using Qwen2.5 tokenizer.
        Args:
            sequences (list): Formatted sequences.
        Returns:
            list: Tokenized sequences.
        """
        tokenized_sequences = [self.tokenizer(seq, return_tensors="pt")["input_ids"].tolist()[0] for seq in sequences]
        return tokenized_sequences

    def preprocess(self, file_path="lotka_volterra_data.h5", sample_size=2):
        """
        Full preprocessing pipeline: Load, scale, format, and tokenize.
        Args:
            file_path (str): Path to dataset.
            sample_size (int): Number of sequences to return as examples.
        """
        data, _ = self.load_data(file_path)
        formatted_sequences = self.scale_and_format(data[:sample_size])
        tokenized_sequences = self.tokenize(formatted_sequences)

        for i in range(sample_size):
            print(f"\n🔹 Example {i+1}:")
            print(f"Formatted: {formatted_sequences[i]}")
            print(f"Tokenized: {tokenized_sequences[i]}")

        return formatted_sequences, tokenized_sequences

if __name__ == "__main__":
    preprocessor = LLMTIMEPreprocessor()
    preprocessor.preprocess()
