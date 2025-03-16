import h5py
import numpy as np
from transformers import AutoTokenizer

class LLMTIMEPreprocessor:
    """
    Preprocesses time-series data for input into the Qwen2.5-Instruct model.

    Enhancements:
    - **Padding Support**: Ensures uniform sequence lengths.
    - **Copying Issue Prevention**: Masks future timesteps to avoid trivial solutions.
    - **Storage of Preprocessed Data**: Saves formatted & tokenized sequences.

    Attributes:
        decimal_places (int): Number of decimal places to round values.
        tokenizer (AutoTokenizer): Qwen2.5 tokenizer.
        scale_factor (float): Computed scale factor for normalization.
        max_length (int): Fixed sequence length for padding (default: 100).
    """

    def __init__(
        self, 
        file_path="lotka_volterra_data.h5", 
        decimal_places=2, 
        model_name="Qwen/Qwen2.5-0.5B-Instruct", 
        max_length=100, 
        mask_future=5
    ):
        """
        Initializes the preprocessor, loads tokenizer, and computes scale factor.

        Args:
            file_path (str): Path to the dataset file.
            decimal_places (int): Rounding precision for scaled values.
            model_name (str): Qwen2.5 tokenizer name.
            max_length (int): Fixed sequence length for padding.
            mask_future (int): Number of future timesteps to mask.
        """
        self.decimal_places = decimal_places
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.mask_future = mask_future  # Number of timesteps to mask

        # Load dataset and determine scaling factor
        data, _ = self.load_data(file_path)
        self.scale_factor = self.compute_scale_factor(data)

    def load_data(self, file_path="lotka_volterra_data.h5"):
        """Loads the predator-prey time-series dataset from an HDF5 file."""
        with h5py.File(file_path, "r") as f:
            trajectories = f["trajectories"][:]  # Shape: (1000, 100, 2)
            time_points = f["time"][:]  # Shape: (100,)
        return trajectories, time_points

    def compute_scale_factor(self, data, target_range=(0, 10)):
        """Computes a scaling factor to normalize data within a defined range."""
        min_val, max_val = np.min(data), np.max(data)
        return target_range[1] / max(abs(min_val), abs(max_val))

    def scale_and_format(self, data):
        """
        Scales, rounds, and formats time-series data into structured text.

        Changes:
        - Future timesteps are masked to prevent direct copying.
        - Supports padding to ensure uniform sequence lengths.
        """
        # Scale & round
        scaled_data = np.round(data * self.scale_factor, self.decimal_places)

        # Apply masking to prevent trivial copying
        masked_data = scaled_data.copy()
        masked_data[:, -self.mask_future:, :] = np.nan  # Mask last `mask_future` timesteps

        # Convert to LLMTIME format
        formatted_sequences = []
        for sequence in masked_data:
            formatted_seq = ";".join(
                [",".join(map(lambda x: "0.0" if np.isnan(x) else str(x), timestep)) for timestep in sequence]
            )
            formatted_sequences.append(formatted_seq)

        return formatted_sequences


    def pad_sequences(self, sequences):
        """
        Pads tokenized sequences to `max_length` using the tokenizer's pad token.
        """
        padded_sequences = [
            seq + [self.tokenizer.pad_token_id] * (self.max_length - len(seq))
            if len(seq) < self.max_length else seq[:self.max_length]
            for seq in sequences
        ]
        return padded_sequences

    def tokenize(self, sequences):
        """Tokenizes and pads formatted sequences using Qwen2.5 tokenizer."""
        tokenized_sequences = [
            self.tokenizer(seq, return_tensors="pt", padding="max_length", max_length=self.max_length)["input_ids"].tolist()[0] 
            for seq in sequences
        ]
        return tokenized_sequences

    def preprocess(self, file_path="lotka_volterra_data.h5", sample_size=2):
        """
        Full preprocessing pipeline:
        - Loads time-series data.
        - Scales, formats, and tokenizes.
        - Applies padding.

        Returns:
            tuple: (formatted sequences, tokenized sequences)
        """
        data, _ = self.load_data(file_path)
        formatted_sequences = self.scale_and_format(data[:sample_size])
        tokenized_sequences = self.tokenize(formatted_sequences)
        tokenized_sequences = self.pad_sequences(tokenized_sequences)

        # Save for later evaluation
        np.save("preprocessed_data.npy", tokenized_sequences)

        # Print results
        for i in range(sample_size):
            print(f"\n🔹 Example {i+1}:")
            print(f"Formatted: {formatted_sequences[i]}")
            print(f"Tokenized: {tokenized_sequences[i]}")

        return formatted_sequences, tokenized_sequences


if __name__ == "__main__":
    preprocessor = LLMTIMEPreprocessor()
    preprocessor.preprocess()
