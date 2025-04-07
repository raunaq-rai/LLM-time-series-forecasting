import sys
import os
sys.path.append(os.path.abspath("src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from compute_flops import (
    count_tokens_for_series,
    compute_average_input_tokens,
    compute_total_training_flops,
    compute_total_eval_flops,
    estimate_max_training_steps
)


@pytest.fixture
def dummy_series():
    return np.sin(np.linspace(0, 10, 100))


@pytest.fixture
def dummy_dataset():
    mock = MagicMock()
    mock.trajectories = [np.ones(100)] * 10
    return mock


@pytest.fixture
def dummy_pipeline():
    mock_pipeline = MagicMock()
    mock_pipeline.trunc_string.return_value = "0.1; 0.2; 0.3;"
    mock_pipeline.tokenizer.return_value = {"input_ids": np.zeros((1, 50))}
    return mock_pipeline


def test_count_tokens_for_series(dummy_series):
    mock_pipeline = MagicMock()
    mock_pipeline.trunc_string.return_value = "0.1; 0.2; 0.3;"
    mock_pipeline.tokenizer.return_value = {"input_ids": np.zeros((1, 42))}
    result = count_tokens_for_series(mock_pipeline, dummy_series, input_fraction=0.7)
    assert result == 42

    result = count_tokens_for_series(mock_pipeline, dummy_series, input_fraction=0.7, context_length=30)
    assert result == 30


@patch("compute_flops.PredictionPipeline")
def test_compute_average_input_tokens(mock_pipeline_class, dummy_dataset):
    pipeline_instance = MagicMock()
    pipeline_instance.trunc_string.return_value = "0.1; 0.2; 0.3;"
    pipeline_instance.tokenizer.return_value = {"input_ids": np.zeros((1, 60))}
    mock_pipeline_class.return_value = pipeline_instance

    avg = compute_average_input_tokens(dummy_dataset, num_series=5)
    assert avg == 60


def test_compute_total_training_flops():
    total = compute_total_training_flops(avg_token_count=512, lora_rank=4, batch_size=2, training_steps=100)
    assert total > 0


def test_compute_total_eval_flops():
    total = compute_total_eval_flops(avg_token_count=512, eval_series_count=50, lora_rank=4)
    assert total > 0


@patch("compute_flops.PredictionPipeline")
@patch("compute_flops.TrajectoryDataset")
def test_estimate_max_training_steps(mock_dataset_class, mock_pipeline_class):
    # Mock dataset and pipeline behaviour
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.trajectories = [np.ones(100)] * 10
    mock_dataset_class.return_value = mock_dataset_instance

    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.trunc_string.return_value = "0.1; 0.2; 0.3;"
    mock_pipeline_instance.tokenizer.return_value = {"input_ids": np.zeros((1, 128))}
    mock_pipeline_class.return_value = mock_pipeline_instance

    steps = estimate_max_training_steps(
        data_path="dummy.h5",
        input_fraction=0.7,
        lora_rank=4,
        batch_size=4,
        flop_budget=1e15,
        train_series_count=10,
        eval_series_count=5,
        context_length=128
    )
    assert isinstance(steps, int)
    assert steps > 0


if __name__ == "__main__":
    pytest.main([__file__])

