import pytest
import numpy as np
from preprocessor import LLMTIMEPreprocessor

@pytest.fixture
def preprocessor():
    """Fixture to initialize LLMTIMEPreprocessor once for all tests."""
    return LLMTIMEPreprocessor()

def test_dataset_loading(preprocessor):
    """Test if dataset loads correctly."""
    assert preprocessor.trajectories.shape == (1000, 100, 2), "Dataset shape is incorrect!"
    assert preprocessor.time_points.shape == (100,), "Time points shape is incorrect!"

def test_auto_scale_factor(preprocessor):
    """Test that scale factor is a reasonable positive value."""
    assert preprocessor.scale_factor > 0, "Scale factor should be positive!"

def test_scaling(preprocessor):
    """Test scaling of a known value."""
    sample_value = np.array([2.5, 5.0])
    scaled_values = preprocessor.scale_and_format(sample_value)
    
    # Expected scaling result
    expected_values = [f"{x / preprocessor.scale_factor:.{preprocessor.decimal_places}f}" for x in sample_value]
    assert scaled_values == expected_values, f"Scaling incorrect: {scaled_values} != {expected_values}"

def test_formatting(preprocessor):
    """Test LLMTIME formatting output."""
    sample_index = 0
    formatted_text = preprocessor.format_input(sample_index, num_steps=5)
    
    # Ensure correct structure (each timestep separated by ';', each variable by ',')
    assert ";" in formatted_text and "," in formatted_text, "Incorrect LLMTIME formatting!"

def test_tokenization(preprocessor):
    """Test if tokenization works correctly."""
    sample_text = "0.23,1.02;0.31,0.87;0.41,0.72"
    tokenized = preprocessor.tokenize_input(sample_text)

    assert tokenized is not None, "Tokenization returned None!"
    assert tokenized.shape[1] > 0, "Tokenized output should not be empty!"

def test_preprocess_sample(preprocessor):
    """Test full sample preprocessing."""
    sample_index = 0
    text, tokenized = preprocessor.preprocess_sample(sample_index, num_steps=5)
    
    assert isinstance(text, str), "Formatted text should be a string!"
    assert tokenized is not None, "Tokenization failed!"
    assert tokenized.shape[1] > 0, "Tokenized output should not be empty!"


