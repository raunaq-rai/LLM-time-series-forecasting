import re
import numpy as np
import h5py
from transformers import AutoTokenizer

class LLMTIMEPostprocessor:
    """
    Postprocessor for converting Qwen2.5 tokenized outputs back into numerical time-series data.

    The postprocessing pipeline includes:
    1. **Decoding** the tokenized output back into text.
    2. **Cleaning** the decoded text to fix spacing and formatting inconsistencies.
    3. **Converting** the cleaned text into numerical arrays.
    4. **Rescaling** the numerical data back to its original range using the stored scaling factor.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for the Qwen2.5 model.
        scale_factor (float): Computed scaling factor to revert data to its original scale.
    """

    def __init__(self, file_path="lotka_volterra_data.h5", model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the postprocessor with the tokenizer and retrieves the original dataset scaling factor.

        Args:
            file_path (str, optional): Path to the dataset file for retrieving scaling information. Default is "lotka_volterra_data.h5".
            model_name (str, optional): Name of the Qwen2.5 tokenizer model. Default is "Qwen/Qwen2.5-0.5B-Instruct".

        Example:
            >>> postprocessor = LLMTIMEPostprocessor()
            >>> print(postprocessor.scale_factor)  # Example scaling factor
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load dataset to compute the original scale factor
        data, _ = self.load_data(file_path)
        self.scale_factor = self.compute_scale_factor(data)

    def load_data(self, file_path="lotka_volterra_data.h5"):
        """
        Loads the original predator-prey dataset to determine the scale factor.

        Args:
            file_path (str): Path to the dataset file.

        Returns:
            tuple:
                - np.ndarray: The dataset containing numerical time-series data (shape: (1000, 100, 2)).
                - np.ndarray: The corresponding time points (shape: (100,)).

        Example:
            >>> data, time_points = postprocessor.load_data("lotka_volterra_data.h5")
            >>> print(data.shape)  # (1000, 100, 2)
            >>> print(time_points.shape)  # (100,)
        """
        with h5py.File(file_path, "r") as f:
            trajectories = f["trajectories"][:]  # Shape (1000, 100, 2)
            time_points = f["time"][:]  # Shape (100,)

        return trajectories, time_points

    def compute_scale_factor(self, data, target_range=(0, 10)):
        """
        Computes the scaling factor used in preprocessing to revert numerical values to their original range.

        Args:
            data (np.ndarray): The dataset from which to compute scaling.
            target_range (tuple, optional): The desired range of values. Default is (0, 10).

        Returns:
            float: Scaling factor to rescale the processed values back to their original range.

        Example:
            >>> scale_factor = postprocessor.compute_scale_factor(data)
            >>> print(scale_factor)  # Example output: 5.2
        """
        min_val, max_val = np.min(data), np.max(data)
        scale_factor = target_range[1] / max(abs(min_val), abs(max_val))  # Scale to target range
        return scale_factor

    def clean_decoded_text(self, decoded_text):
        """
        Cleans the decoded text by correcting formatting inconsistencies.

        Steps:
        - Fixes incorrect decimal spacing (e.g., "1 .23" → "1.23").
        - Ensures correct spacing around commas and semicolons.

        Args:
            decoded_text (str): The raw decoded text output from the tokenizer.

        Returns:
            str: Cleaned and properly formatted text.

        Example:
            >>> raw_text = "1 .23 , 0 .567 ; 2.34 , 1.89"
            >>> cleaned_text = postprocessor.clean_decoded_text(raw_text)
            >>> print(cleaned_text)
            "1.23,0.567;2.34,1.89"
        """
        print(f"🔍 Before Cleaning: {decoded_text}")

        # Fix spacing issues around decimal points
        cleaned_text = re.sub(r'(\d)\s+\.', r'\1.', decoded_text)  # Fix "1 .23" -> "1.23"
        cleaned_text = re.sub(r'\.\s+(\d)', r'.\1', cleaned_text)  # Fix ". 23" -> ".23"

        # Ensure proper comma and semicolon formatting
        cleaned_text = re.sub(r'\s*,\s*', ',', cleaned_text)  # Remove spaces around commas
        cleaned_text = re.sub(r'\s*;\s*', ';', cleaned_text)  # Remove spaces around semicolons

        print(f"🛠️ After Cleaning: {cleaned_text}")
        return cleaned_text

    def decode_sequence(self, tokenized_output):
        """
        Converts tokenized output back into numerical arrays, ensuring consistent shape.
        """
        decoded_text = self.tokenizer.decode(tokenized_output, skip_special_tokens=True)
        print(f"📝 Decoded Raw Text: {decoded_text}")

        cleaned_text = self.clean_decoded_text(decoded_text)
        print(f"🛠️ Cleaned Text: {cleaned_text}")

        try:
            # Convert text back to numerical format
            numeric_values = [list(map(float, timestep.split(','))) for timestep in cleaned_text.split(';')]
            numeric_values = np.array(numeric_values)

            # Reverse the scaling applied in preprocessing
            rescaled_values = numeric_values / self.scale_factor

            # **Ensure the output shape is always (1, T, V)**
            return rescaled_values[np.newaxis, :, :]
        except ValueError as e:
            print(f"🚨 Decoding Error: {e}")
            print(f"🔍 Original Decoded Text: {decoded_text}")
            print(f"🛠️ Cleaned Text: {cleaned_text}")
            return None


if __name__ == "__main__":
    postprocessor = LLMTIMEPostprocessor()
    
    # Example tokenized sequence (replace with actual generated tokens)
    example_tokenized = [[16, 13, 17, 18, 11, 15, 13, 20, 21, 22, 26, 17, 13, 18, 19, 11, 16, 13, 23, 24]]  
    
    decoded_output = postprocessor.decode_sequence(example_tokenized[0])
    print("\n🔹 Decoded Example Sequence:")
    print(decoded_output)
