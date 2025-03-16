import torch
import numpy as np
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocessor import LLMTIMEPreprocessor
from load_qwen import load_qwen
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress tensor-related warnings
warnings.filterwarnings("ignore", category=UserWarning)

class QwenEvaluator:
    def __init__(self):
        """
        Initializes the evaluator with the Qwen model.
        """
        self.model, self.tokenizer = load_qwen()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        # Fix tokenizer padding issue
        self.tokenizer.padding_side = "left"

    def extract_numeric_values(self, text):
        """
        Extracts numeric values from model output while ignoring text.
        Args:
            text (str): Generated text output from model.
        Returns:
            list: List of extracted numeric values.
        """
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)  # Extracts floats & integers
        return list(map(float, numbers)) if numbers else [0.0]  # Default to 0.0 if empty

    def generate_forecasts(self, tokenized_inputs, max_new_tokens=10):
        """
        Generates predictions using the Qwen model.
        Args:
            tokenized_inputs (list): Tokenized input sequences.
            max_new_tokens (int): Number of tokens to generate.
        Returns:
            list: Extracted numeric values.
        """
        input_ids = self.tokenizer(tokenized_inputs, return_tensors="pt", padding=True)["input_ids"]
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=max_new_tokens, attention_mask=torch.ones_like(input_ids))

        # Decode generated tokens into text
        predicted_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        # Extract only numerical values
        return [self.extract_numeric_values(text) for text in predicted_texts]

    def pad_or_truncate(self, sequences, target_length):
        """
        Pads or truncates sequences to a fixed length.
        Args:
            sequences (list of lists): List of numerical sequences.
            target_length (int): Length to pad or truncate to.
        Returns:
            list of lists: Sequences with uniform length.
        """
        return [
            seq[:target_length] + [0] * (target_length - len(seq))
            for seq in sequences
        ]

    def evaluate(self, sample_size=2):
        """
        Evaluates the Qwen model on tokenized data.
        Computes MSE and MAE using numerical sequences.
        """
        preprocessor = LLMTIMEPreprocessor()
        formatted_sequences, tokenized_sequences = preprocessor.preprocess(sample_size=sample_size)

        # Get predictions as numerical values
        predicted_values = self.generate_forecasts(formatted_sequences)

        # Convert formatted sequences back to numerical values for comparison
        ground_truth_values = [list(map(float, seq.replace(";", ",").split(","))) for seq in formatted_sequences]

        # Find the max sequence length
        max_length = max(max(len(seq) for seq in ground_truth_values), max(len(seq) for seq in predicted_values))

        # Ensure all sequences are the same length
        ground_truth_values = self.pad_or_truncate(ground_truth_values, max_length)
        predicted_values = self.pad_or_truncate(predicted_values, max_length)

        # Compute metrics
        mse = mean_squared_error(ground_truth_values, predicted_values)
        mae = mean_absolute_error(ground_truth_values, predicted_values)

        print(f"\n📊 **Evaluation Metrics:**")
        print(f"⚠️ Mean Squared Error (MSE): {mse:.4f}")
        print(f"⚠️ Mean Absolute Error (MAE): {mae:.4f}")

if __name__ == "__main__":
    evaluator = QwenEvaluator()
    evaluator.evaluate()
