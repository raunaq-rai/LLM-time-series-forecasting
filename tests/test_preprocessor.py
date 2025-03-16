import pytest
import numpy as np
from preprocessor import LotkaVolterraPreprocessor
from load_qwen import load_qwen_model

@pytest.fixture(scope="module")
def preprocessor():
    """Fixture to initialize the preprocessor."""
    return LotkaVolterraPreprocessor()

@pytest.fixture(scope="module")
def tokenizer():
    """Fixture to load the tokenizer from load_qwen.py"""
    tokenizer, _, _ = load_qwen_model()
    return tokenizer

def test_load_dataset(preprocessor):
    """Test if the dataset loads correctly."""
    assert preprocessor.trajectories is not None, "Failed to load trajectories!"
    assert preprocessor.time_points is not None, "Failed to load time points!"
    assert preprocessor.trajectories.shape == (1000, 100, 2), "Unexpected shape for trajectories!"
    assert preprocessor.time_points.shape == (100,), "Unexpected shape for time points!"

def test_format_input(preprocessor):
    """Test formatting of a single sample."""
    sample_index = 0
    num_steps = 20
    formatted_text = preprocessor.format_input(sample_index, num_steps)
    
    assert isinstance(formatted_text, str), "Formatted input is not a string!"
    assert "Time-Series Data" in formatted_text, "Formatted text does not contain expected structure!"
    
    num_values = formatted_text.count(";") + 1  # Count time steps
    assert num_values == num_steps, f"Expected {num_steps} values, got {num_values}!"

def test_tokenization(preprocessor, tokenizer):
    """Test tokenization process using the tokenizer from load_qwen.py."""
    sample_index = 0
    num_steps = 20

    # Get preprocessed text
    input_text = preprocessor.format_input(sample_index, num_steps)

    # Tokenize using the correct tokenizer
    tokenized_output = tokenizer(input_text, return_tensors="pt")["input_ids"]

    assert tokenized_output is not None, "Tokenization failed!"
    assert tokenized_output.shape[1] > 0, "Tokenized sequence is empty!"

def test_detokenization(preprocessor, tokenizer):
    """Test if tokenized input can be correctly decoded back."""
    sample_index = 0
    num_steps = 20

    # Get preprocessed text
    input_text = preprocessor.format_input(sample_index, num_steps)

    # Tokenize and detokenize using the correct tokenizer
    tokenized_output = tokenizer(input_text, return_tensors="pt")["input_ids"]
    decoded_text = tokenizer.decode(tokenized_output[0], skip_special_tokens=True)

    # Check if the decoded text is similar to the original
    assert isinstance(decoded_text, str), "Decoded output is not a string!"
    assert "Time-Series Data" in decoded_text, "Decoded text does not match expected structure!"

if __name__ == "__main__":
    pytest.main()
