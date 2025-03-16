import h5py
import numpy as np
from transformers import AutoTokenizer

class LLMTIMEPreprocessor:
    """
    Preprocesses time-series data for input into the Qwen2.5-Instruct model.
    
    The preprocessing pipeline includes:
    1. **Loading** numerical time-series data from `lotka_volterra_data.h5`.
    2. **Computing a scaling factor** to normalize values into a suitable range (0-10).
    3. **Formatting** the data as structured text, where:
        - Different variables at the same timestep are separated by commas `,`
        - Different timesteps are separated by semicolons `;`
    4. **Tokenizing** the formatted text using the Qwen2.5 tokenizer.
    
    Attributes:
        decimal_places (int): The number of decimal places to round the scaled values.
        tokenizer (AutoTokenizer): Tokenizer from the Qwen2.5 model.
        scale_factor (float): Computed scaling factor to normalize the data.
    """

    def __init__(self, file_path="lotka_volterra_data.h5", decimal_places=2, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the preprocessor with the tokenizer and computes the scaling factor.

        Args:
            file_path (str, optional): Path to the dataset file. Default is "lotka_volterra_data.h5".
            decimal_places (int, optional): Number of decimal places to round values to. Default is 2.
            model_name (str, optional): Name of the Qwen2.5 tokenizer model. Default is "Qwen/Qwen2.5-0.5B-Instruct".

        Example:
            >>> preprocessor = LLMTIMEPreprocessor()
            >>> print(preprocessor.scale_factor)  # Example scaling factor
        """
        self.decimal_places = decimal_places
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the dataset to determine an appropriate scale factor
        data, _ = self.load_data(file_path)
        self.scale_factor = self.compute_scale_factor(data)

    def load_data(self, file_path="lotka_volterra_data.h5"):
        """
        Loads the predator-prey time-series dataset from an HDF5 file.

        Args:
            file_path (str): Path to the dataset file.

        Returns:
            tuple:
                - np.ndarray: The dataset containing numerical time-series data (shape: (1000, 100, 2)).
                - np.ndarray: The corresponding time points (shape: (100,)).

        Example:
            >>> data, time_points = preprocessor.load_data("lotka_volterra_data.h5")
            >>> print(data.shape)  # (1000, 100, 2)
            >>> print(time_points.shape)  # (100,)
        """
        with h5py.File(file_path, "r") as f:
            trajectories = f["trajectories"][:]  # Shape (1000, 100, 2)
            time_points = f["time"][:]  # Shape (100,)

        return trajectories, time_points

    def compute_scale_factor(self, data, target_range=(0, 10)):
        """
        Computes a scaling factor to normalize the data within a defined range.

        Args:
            data (np.ndarray): The dataset from which to compute scaling.
            target_range (tuple, optional): The desired range of values. Default is (0, 10).

        Returns:
            float: Scaling factor to ensure values are within the specified range.

        Example:
            >>> scale_factor = preprocessor.compute_scale_factor(data)
            >>> print(scale_factor)  # Example output: 5.2
        """
        min_val, max_val = np.min(data), np.max(data)
        scale_factor = target_range[1] / max(abs(min_val), abs(max_val))  # Scale to target range
        return scale_factor

    def scale_and_format(self, data):
        """
        Scales and formats the time-series data into structured text format.

        Steps:
        - Each value is scaled using the computed `scale_factor`.
        - The values are rounded to `decimal_places`.
        - The structured format follows:
            - Variables at the same timestep are separated by commas `,`.
            - Different timesteps are separated by semicolons `;`.

        Args:
            data (np.ndarray): Time-series data to be scaled and formatted.

        Returns:
            list[str]: List of formatted sequences as strings.

        Example:
            >>> sample_data = np.array([[[0.5, 1.5], [0.6, 1.4], [0.7, 1.3]]])
            >>> formatted = preprocessor.scale_and_format(sample_data)
            >>> print(formatted)
            ["2.5,7.5;3.0,7.0;3.5,6.5"]
        """
        scaled_data = np.round(data * self.scale_factor, self.decimal_places)
        formatted_sequences = [
            ";".join([",".join(map(str, timestep)) for timestep in sequence])
            for sequence in scaled_data
        ]
        return formatted_sequences

    def tokenize(self, sequences):
        """
        Tokenizes the formatted time-series sequences using the Qwen2.5 tokenizer.

        Args:
            sequences (list[str]): List of formatted sequences.

        Returns:
            list[list[int]]: Tokenized sequences represented as lists of integer token IDs.

        Example:
            >>> formatted_sequence = ["2.5,7.5;3.0,7.0;3.5,6.5"]
            >>> tokenized = preprocessor.tokenize(formatted_sequence)
            >>> print(tokenized)
            [[16, 23, 11, 24, 17, 26, 18, 21, 11, 22, 24, 26, 20, 19, 11, 21, 22]]
        """
        tokenized_sequences = [self.tokenizer(seq, return_tensors="pt")["input_ids"].tolist()[0] for seq in sequences]
        return tokenized_sequences

    def preprocess(self, file_path="lotka_volterra_data.h5", sample_size=2):
        """
        Full preprocessing pipeline that:
        1. Loads time-series data from the dataset.
        2. Scales and formats a subset of the data.
        3. Tokenizes the formatted sequences.

        Args:
            file_path (str, optional): Path to the dataset file. Default is "lotka_volterra_data.h5".
            sample_size (int, optional): Number of sequences to preprocess. Default is 2.

        Returns:
            tuple:
                - list[str]: Formatted sequences.
                - list[list[int]]: Tokenized sequences.

        Example:
            >>> formatted, tokenized = preprocessor.preprocess(sample_size=1)
            >>> print("Formatted:", formatted[0])
            >>> print("Tokenized:", tokenized[0])
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
