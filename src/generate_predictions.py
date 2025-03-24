import torch
import re
import numpy as np
from preprocessor import LLMTIMEPreprocessor
from load_qwen import load_qwen_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class QwenForecaster:
    """
    Evaluates Qwen2.5's ability to forecast time series data using the first half of the dataset tokens to predict the second half.
    """

    def __init__(self):
        """
        Loads the Qwen2.5 model and tokenizer using load_qwen.py.
        """
        print("📌 Loading untrained model via load_qwen.py")
        self.tokenizer, self.model, self.device = load_qwen_model()
        self.model.eval()  # Set model to evaluation mode

    def generate_prediction(self, tokenized_input, max_new_tokens):
        """
        Uses the model to generate predictions based on tokenized input.

        Args:
            tokenized_input (torch.Tensor): Tokenized input sequence.
            max_new_tokens (int): Number of new tokens to generate.

        Returns:
            list[int]: Generated token sequence from the model.
        """
        tokenized_input = tokenized_input.to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                tokenized_input, 
                max_new_tokens=max_new_tokens
            )

        # Extract only the newly generated tokens
        new_tokens = output_tokens[:, tokenized_input.shape[1]:]

        # Ensure exactly `max_new_tokens` are returned
        return new_tokens[0][:max_new_tokens]  # Trim extra tokens if necessary

    def decode_prediction(self, token_sequence):
        """
        Decodes a sequence of tokens into a numerical string.

        Args:
            token_sequence (list[int]): Token IDs.

        Returns:
            str: Decoded sequence.
        """
        return self.tokenizer.decode(token_sequence, skip_special_tokens=True)

    def extract_numbers(self, text):
        """
        Extracts numerical values from the generated model output.

        Args:
            text (str): Model output text.

        Returns:
            list[float]: List of extracted floating-point numbers.
        """
        extracted_numbers = [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", text)]
        return extracted_numbers

    def evaluate(self, true_values, predicted_values):
        """
        Computes performance metrics comparing true vs. predicted values.

        Args:
            true_values (list[float]): Ground truth values.
            predicted_values (list[float]): Model-predicted values.

        Returns:
            tuple: (mse, mae, r2) Mean Squared Error, Mean Absolute Error, R² Score.
        """
        # Ensure predicted_values has exactly 50 elements
        predicted_values = predicted_values[:50]  # Trim if too long
        if len(predicted_values) < 50:
            predicted_values += [float('nan')] * (50 - len(predicted_values))  # Pad if too short

        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)

        return mse, mae, r2


if __name__ == "__main__":
    preprocessor = LLMTIMEPreprocessor()
    forecaster = QwenForecaster()

    # Select a sample to process
    sample_index = 0

    # Use all 100 time steps as input
    full_text = preprocessor.format_input(sample_index, num_steps=100)
    tokenized_full = preprocessor.tokenize_input(full_text)

    print("\n📝 Full Preprocessed Input (100 Steps):\n", full_text)
    print("\n🔢 Full Tokenized Sequence (Variable Length):\n", tokenized_full.tolist())

    # Dynamically determine the halfway point
    num_tokens = tokenized_full.shape[1]
    half_point = num_tokens // 2

    # Split into first half (input) and second half (target)
    first_half_tokens = tokenized_full[:, :half_point]
    second_half_tokens = tokenized_full[:, half_point:]

    print(f"\n🔢 Splitting at {half_point} tokens...")
    print("\n🔢 First Half of Tokens (Used as Input):\n", first_half_tokens.tolist())
    print("\n🔢 Second Half of Tokens (Ground Truth for Evaluation):\n", second_half_tokens.tolist())

    # Generate new tokens using first half as input
    generated_tokens = forecaster.generate_prediction(
        first_half_tokens, 
        max_new_tokens=second_half_tokens.shape[1]
    )

    print("\n🔢 Generated Tokenized Sequence:\n", generated_tokens.tolist())

    # Decode the tokenized prediction into text
    decoded_output = forecaster.decode_prediction(generated_tokens)
    print("\n🔍 Model Output (Decoded Text):\n", decoded_output)

    # Extract numerical predictions from decoded text
    predicted_values = forecaster.extract_numbers(decoded_output)
    print("\n🔮 Extracted Predicted Values:\n", predicted_values)

    # Extract the true numerical values corresponding to the second half of the time series
    # Ensure `true_values` is only 50 values (matching `predicted_values`)
    true_values = preprocessor.trajectories[sample_index, 50:100, 0].tolist()

    print("\n✅ True Last 50 Values (Ensured Correct Length):\n", true_values)

    # Evaluate model performance
    mse, mae, r2 = forecaster.evaluate(true_values, predicted_values)

    print("\n📊 Evaluation Metrics:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R² Score: {r2:.4f}")
