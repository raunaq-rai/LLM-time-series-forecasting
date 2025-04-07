import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.append(os.path.abspath("src"))

from preprocessor import LLMTIMEPreprocessor



@pytest.fixture
def mock_qwen():
    with patch("preprocessor.load_qwen_model") as mock_loader:
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": np.array([[1, 2, 3]])}
        mock_loader.return_value = ("mock_model", mock_tokenizer)
        yield mock_loader


def test_auto_scale_factor(mock_qwen):
    preprocessor = LLMTIMEPreprocessor()
    prey = np.array([2.0, 4.0, 6.0])
    predator = np.array([1.0, 5.0, 3.0])
    scale = preprocessor.auto_scale_factor(prey, predator)
    assert scale == pytest.approx(0.1 * max(np.percentile(prey, 95), np.percentile(predator, 95)))


def test_scale_and_format(mock_qwen):
    preprocessor = LLMTIMEPreprocessor(decimal_places=1)
    preprocessor.scale_factor = 2.0
    values = np.array([2.0, 4.0, 6.0])
    scaled = preprocessor.scale_and_format(values)
    assert scaled == ["1.0", "2.0", "3.0"]


def test_format_input(mock_qwen):
    preprocessor = LLMTIMEPreprocessor(decimal_places=1)
    prey = np.array([10.0, 20.0, 30.0])
    predator = np.array([5.0, 15.0, 25.0])
    formatted = preprocessor.format_input(prey, predator, num_steps=3)
    assert "," in formatted and ";" in formatted
    assert formatted.count(";") == 2  # 3 pairs → 2 semicolons


def test_tokenize_input(mock_qwen):
    preprocessor = LLMTIMEPreprocessor()
    dummy_text = "1.0,2.0;3.0,4.0"
    preprocessor.tokenizer = MagicMock(return_value={"input_ids": np.array([[42, 43, 44]])})
    tokens = preprocessor.tokenize_input(dummy_text)
    assert tokens.shape == (1, 3)


def test_preprocess_sample(mock_qwen):
    preprocessor = LLMTIMEPreprocessor()
    prey = np.array([2.0, 4.0, 6.0])
    predator = np.array([1.0, 5.0, 3.0])
    preprocessor.tokenizer = MagicMock(return_value={"input_ids": np.array([[11, 22, 33]])})
    text, tokens, scale = preprocessor.preprocess_sample(prey, predator, num_steps=3)

    assert isinstance(text, str)
    assert tokens.shape == (1, 3)
    assert scale > 0

