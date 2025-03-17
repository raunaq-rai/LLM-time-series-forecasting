import os
import h5py
import numpy as np
from load_qwen import load_qwen_model 


class LLMTIMEPreprocessor:
    """
    Implements the LLMTIME preprocessing scheme for multivariate time-series data.
    
    This preprocessor:
    - Scales numeric values to a controlled range (e.g., 0-10) for stability.
    - Rounds values to a fixed decimal precision for uniformity.
    - Encodes time-series sequences into a structured format suitable for tokenization.
    - Uses Qwen2.5's tokenizer to convert formatted text into tokenized input for LLMs.
    """

    FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lotka_volterra_data.h5"))

    def __init__(self, file_path=None, scale_factor=None, decimal_places=3):
        """
        Initializes the LLMTIME preprocessor and loads the dataset.

        Args:
            file_path (str, optional): Path to the HDF5 dataset. Defaults to FILE_PATH.
            scale_factor (float, optional): Scaling factor (α) for normalization. If None, it is auto-determined.
            decimal_places (int): Number of decimal places to round values to for consistency.
        """
        self.file_path = file_path or self.FILE_PATH
        self.decimal_places = decimal_places
        self.trajectories, self.time_points = self.load_dataset()

        # Load tokenizer from load_qwen.py
        self.tokenizer, _, _ = load_qwen_model()

        # Automatically determine scaling factor if not provided
        self.scale_factor = scale_factor or self.auto_scale_factor()
        print(f" Using scale factor: {self.scale_factor:.3f}")

    def load_dataset(self):
        """
        Loads the Lotka-Volterra dataset from an HDF5 file.

        Returns:
            tuple:
                - trajectories (numpy.ndarray): Time-series population data of shape (1000, 100, 2).
                - time_points (numpy.ndarray): Time data of shape (100,).
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f" Data file not found: {self.file_path}")

        with h5py.File(self.file_path, "r") as f:
            trajectories = f["trajectories"][:]  # Shape: (1000, 100, 2)
            time_points = f["time"][:]  # Shape: (100,)

        print(f" Dataset loaded: {trajectories.shape[0]} samples, {trajectories.shape[1]} time steps")
        return trajectories, time_points

    def auto_scale_factor(self):
        """
        Determines an appropriate scale factor based on dataset distribution.

        This method ensures that most data values fall within a reasonable range (e.g., 0-10)
        to improve numerical stability when tokenizing sequences.

        Returns:
            float: Computed scale factor (α).
        """
        # Use the 95th percentile to avoid extreme outliers affecting scaling
        max_prey = np.percentile(self.trajectories[:, :, 0], 95)
        max_predator = np.percentile(self.trajectories[:, :, 1], 95)

        # Scale factor ensures majority of values fall in [0,10] range
        return max(max_prey, max_predator) / 10

    def scale_and_format(self, values):
        """
        Applies numeric scaling and rounds values to a fixed precision.

        Args:
            values (numpy.ndarray): Array of numerical values.

        Returns:
            list[str]: List of scaled and formatted numeric values.
        """
        scaled = values / self.scale_factor  # Normalize
        rounded = np.round(scaled, self.decimal_places)  # Round to fixed decimal places
        return [f"{x:.{self.decimal_places}f}" for x in rounded]

    def format_input(self, sample_index, num_steps=50):
        """
        Converts a time-series sequence into LLMTIME structured text.

        The format follows:
        - Values at each time step are separated by commas (`,`) → prey, predator
        - Different time steps are separated by semicolons (`;`)

        Example Output:
            "0.95,1.04; 0.74,0.78; 0.68,0.56; ..."

        Args:
            sample_index (int): Index of the sample.
            num_steps (int): Number of time steps to include in the formatted input.

        Returns:
            str: Preprocessed text representation of the time series.
        """
        prey = self.trajectories[sample_index, :num_steps, 0]
        predator = self.trajectories[sample_index, :num_steps, 1]

        # Scale and format values
        prey_scaled = self.scale_and_format(prey)
        predator_scaled = self.scale_and_format(predator)

        # Construct LLMTIME-compatible text format
        formatted_text = "; ".join([f"{prey_scaled[i]},{predator_scaled[i]}" for i in range(num_steps)])

        return formatted_text

    def tokenize_input(self, text):
        """
        Tokenizes the formatted text using the Qwen2.5 tokenizer.

        Args:
            text (str): Preprocessed time-series text.

        Returns:
            torch.Tensor: Tokenized input tensor.
        """
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)["input_ids"]

    def preprocess_sample(self, sample_index, num_steps=50):
        """
        Prepares a time-series sample for model input.

        Steps:
        1. Formats the sequence using LLMTIME encoding.
        2. Tokenizes the formatted text using Qwen2.5.

        Args:
            sample_index (int): Index of the sample.
            num_steps (int): Number of time steps to include.

        Returns:
            tuple:
                - formatted_text (str): LLMTIME-encoded time series.
                - tokenized_tensor (torch.Tensor): Tokenized version of the formatted text.
        """
        text = self.format_input(sample_index, num_steps)
        tokenized = self.tokenize_input(text)
        return text, tokenized


if __name__ == "__main__":
    preprocessor = LLMTIMEPreprocessor()

    # Select two sample indices
    sample_index_1 = 0
    sample_index_2 = 1

    # Load original data (before preprocessing)
    original_1 = preprocessor.trajectories[sample_index_1, :5, :]  # First 5 time steps
    original_2 = preprocessor.trajectories[sample_index_2, :5, :]

    # Preprocess samples
    raw_text_1, tokenized_seq_1 = preprocessor.preprocess_sample(sample_index_1, num_steps=5)
    raw_text_2, tokenized_seq_2 = preprocessor.preprocess_sample(sample_index_2, num_steps=5)

    print("\n================== Example 1 ==================")
    print("📌 Original Sequence (First 5 time steps):")
    for i, (prey, pred) in enumerate(original_1):
        print(f"   Step {i+1}: ({prey:.3f}, {pred:.3f})")

    print("\n📝 Preprocessed Sequence (LLMTIME Format):")
    print("  ", raw_text_1)

    print("\n🔢 Tokenized Sequence (Qwen Token IDs):")
    print("  ", tokenized_seq_1[0].tolist())

    print("\n================== Example 2 ==================")
    print("📌 Original Sequence (First 5 time steps):")
    for i, (prey, pred) in enumerate(original_2):
        print(f"   Step {i+1}: ({prey:.3f}, {pred:.3f})")

    print("\n📝 Preprocessed Sequence (LLMTIME Format):")
    print("  ", raw_text_2)

    print("\n🔢 Tokenized Sequence (Qwen Token IDs):")
    print("  ", tokenized_seq_2[0].tolist())


