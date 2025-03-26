import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from load_qwen import load_qwen_model  # Ensure this file correctly loads the untrained model

class LLMTIMEPreprocessor:
    """Preprocesses Lotka-Volterra time-series data for Qwen2.5-Instruct."""

    def __init__(self, decimal_places=2):
        self.decimal_places = decimal_places
        self.model, self.tokenizer = load_qwen_model()  # Unpacking only two values
        self.scale_factor = None  # Initialize scale_factor as None

    def auto_scale_factor(self, prey, predator):
        """Computes an appropriate scale factor for stability based on the specific sample."""
        return max(np.percentile(prey, 95), np.percentile(predator, 95)) / 10

    def scale_and_format(self, values):
        """Scales and rounds values to a fixed precision."""
        return [f"{x:.{self.decimal_places}f}" for x in np.round(values / self.scale_factor, self.decimal_places)]

    def format_input(self, prey, predator, num_steps=50):
        """Formats time-series data into LLMTIME structured text."""
        self.scale_factor = self.auto_scale_factor(prey, predator)
        return ";".join([f"{p},{q}" for p, q in zip(self.scale_and_format(prey[:num_steps]), self.scale_and_format(predator[:num_steps]))])

    def tokenize_input(self, text):
        """Tokenizes formatted text using Qwen2.5 tokenizer."""
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)["input_ids"]

    def preprocess_sample(self, prey, predator, num_steps=50):
        """Formats and tokenizes a time-series sample, returning scale factor."""
        text = self.format_input(prey, predator, num_steps)
        tokenized = self.tokenize_input(text)
        return text, tokenized, self.scale_factor  # Now returning 3 values


if __name__ == "__main__":
    # Example usage
    prey = np.array([2.92, 2.27, 2.09, 2.20, 2.53, 3.08, 3.88, 4.93, 6.24, 7.75])
    predator = np.array([3.20, 2.39, 1.73, 1.25, 0.92, 0.71, 0.57, 0.50, 0.47, 0.50])
    num_steps = 10

    preprocessor = LLMTIMEPreprocessor()
    print("\n🔄 Preprocessing Sample...")
    raw_text, tokenized_seq, scale_factor = preprocessor.preprocess_sample(prey, predator, num_steps)
    print("\n📝 Formatted Input Text:\n", raw_text)
    print("\n🔢 Tokenized Sequence:\n", tokenized_seq.tolist())
    print("\n🔢 Scale Factor:\n", scale_factor)
