import re
import numpy as np
import h5py
from transformers import AutoTokenizer

class LLMTIMEPostprocessor:
    def __init__(self, file_path="lotka_volterra_data.h5", model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the postprocessor with the tokenizer and loads the original dataset for rescaling.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the dataset to determine the original scale factor
        data, _ = self.load_data(file_path)
        self.scale_factor = self.compute_scale_factor(data)

    def load_data(self, file_path="lotka_volterra_data.h5"):
        """
        Loads the predator-prey dataset.
        Args:
            file_path (str): Path to the dataset.
        Returns:
            np.array: Time series data.
        """
        with h5py.File(file_path, "r") as f:
            trajectories = f["trajectories"][:]  # Shape (1000, 100, 2)
            time_points = f["time"][:]  # Shape (100,)

        return trajectories, time_points

    def compute_scale_factor(self, data, target_range=(0, 10)):
        """
        Determines the scaling factor used in preprocessing.
        """
        min_val, max_val = np.min(data), np.max(data)
        scale_factor = target_range[1] / max(abs(min_val), abs(max_val))  # Scale to 0-10 range
        return scale_factor

    def clean_decoded_text(self, decoded_text):
        """
        Cleans the decoded text by fixing spacing and formatting issues.
        """
        print(f"🔍 Before Cleaning: {decoded_text}")

        # Ensure correct decimal formatting
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

            # Ensure the output matches the original shape (1, T, V)
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
