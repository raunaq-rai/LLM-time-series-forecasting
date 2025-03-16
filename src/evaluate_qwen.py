import torch
import numpy as np
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocessor import LLMTIMEPreprocessor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress tensor-related warnings
warnings.filterwarnings("ignore", category=UserWarning)


class QwenEvaluator:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the evaluator with the untrained Qwen model.
        Loads the pre-trained model and tokenizer, sets up the device.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
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
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)  # Extract floats & integers
        numeric_values = []
        for num in numbers:
            try:
                numeric_values.append(float(num))
            except ValueError:
                print(f"⚠️ Warning: Unexpected non-numeric value encountered in model output: {num}")
        return numeric_values if numeric_values else [0.0]  # Default to [0.0] if empty

    def generate_forecasts(self, text_inputs, max_new_tokens=10):
        """
        Generates predictions using the Qwen model.
        Args:
            text_inputs (list[str]): List of raw text sequences.
            max_new_tokens (int): Number of tokens to generate.
        Returns:
            list: Extracted numeric values.
        """
        # ✅ Add instruction to test if it changes behavior
        instruction = "Given the following time series, predict the next values: "
        modified_inputs = [instruction + seq for seq in text_inputs]

        input_ids = self.tokenizer(modified_inputs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=max_new_tokens, attention_mask=torch.ones_like(input_ids))

        # Decode generated tokens into text
        predicted_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        print(f"\n🔍 Debug: Raw Model Output = {predicted_texts}")  # Debugging print

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

    def preprocess_ground_truth(self, formatted_sequences):
        """
        Converts formatted sequences to numerical values while handling "X" placeholders.
        Args:
            formatted_sequences (list[str]): Formatted time-series sequences.
        Returns:
            list[list[float]]: Cleaned numerical sequences.
        """
        cleaned_sequences = []
        for seq in formatted_sequences:
            numeric_values = []
            for value in seq.replace(";", ",").split(","):
                try:
                    numeric_values.append(float(value))
                except ValueError:
                    print(f"⚠️ Warning: Encountered 'X' or invalid value in ground truth: {value}, replacing with 0.0")
                    numeric_values.append(0.0)  # Replace 'X' with 0.0
            cleaned_sequences.append(numeric_values)

        return cleaned_sequences

    def evaluate(self, sample_size=5):
        """
        Evaluates the untrained Qwen model on tokenized data.
        Computes MSE, MAE, and R² score.

        - Preprocesses sample time series data.
        - Passes tokenized sequences through the Qwen model.
        - Extracts numerical values from generated text.
        - Compares predictions to ground truth.
        """
        preprocessor = LLMTIMEPreprocessor()
        formatted_sequences, tokenized_sequences = preprocessor.preprocess(sample_size=sample_size)

        print(f"\n✅ Debug: Formatted Sequences = {formatted_sequences}")  # Debugging print

        # ✅ Replace "X" values before converting to float
        ground_truth_values = self.preprocess_ground_truth(formatted_sequences)

        # Get predictions as numerical values
        predicted_values = self.generate_forecasts(formatted_sequences)

        # Ensure predicted values match valid ground truth length
        predicted_values = [
            pred[:len(gt)] if len(pred) >= len(gt) else pred + [0] * (len(gt) - len(pred))
            for pred, gt in zip(predicted_values, ground_truth_values)
        ]

        # Find the max sequence length
        max_length = max(len(seq) for seq in ground_truth_values)

        # Ensure all sequences are the same length
        ground_truth_values = self.pad_or_truncate(ground_truth_values, max_length)
        predicted_values = self.pad_or_truncate(predicted_values, max_length)

        # Compute metrics
        mse = mean_squared_error(ground_truth_values, predicted_values)
        mae = mean_absolute_error(ground_truth_values, predicted_values)
        r2 = r2_score(ground_truth_values, predicted_values)

        # 🔍 Print Ground Truth vs. Predictions for Analysis
        for i in range(sample_size):
            print(f"\n **Example {i+1}:**")
            print(f" **Ground Truth:** {ground_truth_values[i][:10]} ...")
            print(f" **Model Prediction:** {predicted_values[i][:10]} ...")

        print(f"\n **Evaluation Metrics:**")
        print(f" Mean Squared Error (MSE): {mse:.4f}")
        print(f" Mean Absolute Error (MAE): {mae:.4f}")
        print(f" R² Score: {r2:.4f}")


if __name__ == "__main__":
    evaluator = QwenEvaluator()
    evaluator.evaluate()
