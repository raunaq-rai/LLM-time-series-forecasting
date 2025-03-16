import torch
import re
from transformers import AutoTokenizer
from preprocessor import LotkaVolterraPreprocessor
from load_qwen import load_qwen_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class QwenForecaster:
    """
    Evaluates Qwen2.5's ability to forecast time series.
    """

    def __init__(self):
        """
        Loads the Qwen2.5 model and tokenizer using load_qwen.py.
        """
        print("📌 Loading model via load_qwen.py")
        self.tokenizer, self.model, self.device = load_qwen_model()

    def generate_prediction(self, input_text):
        """
        Uses the model to generate predictions.
        
        Args:
            input_text (str): The formatted input text sequence.

        Returns:
            str: Generated output sequence from the model.
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)["input_ids"]
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=40,  # Generate 40 tokens for prediction
                do_sample=True,
                top_k=30,
                temperature=1.6,
                repetition_penalty=1.2,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    @staticmethod
    def extract_numbers(text):
        """
        Extracts numerical values from the generated model output.

        Args:
            text (str): Model output text.

        Returns:
            list[float]: List of extracted floating-point numbers.
        """
        return [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", text)]

    def evaluate(self, true_values, predicted_values):
        """
        Computes performance metrics comparing true vs. predicted values.

        Args:
            true_values (list[float]): Ground truth values.
            predicted_values (list[float]): Model-predicted values.

        Returns:
            tuple: (mse, mae, r2) Mean Squared Error, Mean Absolute Error, R² Score.
        """
        predicted_values = predicted_values[:len(true_values)]
        
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)
        
        return mse, mae, r2

if __name__ == "__main__":
    preprocessor = LotkaVolterraPreprocessor()
    forecaster = QwenForecaster()

    # Preprocess and tokenize a sample
    sample_index = 0
    raw_text, tokenized_seq = preprocessor.preprocess_sample(sample_index)

    print("\n📝 Preprocessed Text:\n", raw_text)
    print("\n🔢 Tokenized Sequence (as integers):\n", tokenized_seq.tolist())

    # Generate prediction
    generated_text = forecaster.generate_prediction(raw_text)
    print("\n🔍 Raw Model Output:\n", generated_text)

    # Extract numerical values
    predicted_values = forecaster.extract_numbers(generated_text)
    true_values = preprocessor.trajectories[sample_index, 20:30, :].flatten()

    # Evaluate performance
    mse, mae, r2 = forecaster.evaluate(true_values, predicted_values)
    print("\n📊 Evaluation Metrics:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R² Score: {r2:.4f}")
