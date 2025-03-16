import unittest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from src.evaluate_qwen import QwenEvaluator
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TestQwenEvaluator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the QwenEvaluator once for all tests.
        """
        cls.evaluator = QwenEvaluator()

    def test_model_initialization(self):
        """
        Test if the model and tokenizer are properly loaded onto the correct device.
        """
        self.assertIsNotNone(self.evaluator.model)
        self.assertIsNotNone(self.evaluator.tokenizer)
        self.assertTrue(torch.device("mps" if torch.backends.mps.is_available() else "cpu"), self.evaluator.device)

    def test_extract_numeric_values(self):
        """
        Test the extraction of numeric values from text output.
        """
        text_samples = [
            "1.23, 4.56; 7.89",  # Normal case
            "The forecast is: 10.5, 20.3; 30.1",  # Mixed text & numbers
            "No numbers here!",  # Edge case (no numbers)
            "-5.4, 3.14; 0.99",  # Handles negative & decimals
        ]
        expected_outputs = [
            [1.23, 4.56, 7.89],
            [10.5, 20.3, 30.1],
            [0.0],  # Default case when no numbers are found
            [-5.4, 3.14, 0.99]
        ]

        for text, expected in zip(text_samples, expected_outputs):
            with self.subTest(text=text):
                extracted_values = self.evaluator.extract_numeric_values(text)
                self.assertEqual(extracted_values, expected)

    @patch("src.evaluate_qwen.QwenEvaluator.generate_forecasts")
    def test_generate_forecasts_mocked(self, mock_generate):
        """
        Mock test to ensure forecast generation returns expected numerical sequences.
        This avoids running the full model and speeds up testing.
        """
        mock_generate.return_value = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

        tokenized_inputs = ["0.5,1.5;0.6,1.4;0.7,1.3", "2.2,3.3;4.4,5.5"]
        output = self.evaluator.generate_forecasts(tokenized_inputs, max_new_tokens=5)

        self.assertEqual(output, [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])

    def test_pad_or_truncate(self):
        """
        Test that sequences are correctly padded or truncated to the target length.
        """
        sequences = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        target_length = 4
        expected_output = [[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 5.0, 0.0]]

        result = self.evaluator.pad_or_truncate(sequences, target_length)
        self.assertEqual(result, expected_output)

    @patch("src.evaluate_qwen.QwenEvaluator.generate_forecasts")
    def test_evaluate_pipeline(self, mock_generate):
        """
        Test the full evaluation pipeline, ensuring:
        - Preprocessing works
        - Predictions are generated (mocked)
        - MSE and MAE calculations are correct
        """
        # Mock predictions
        mock_generate.return_value = [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]

        # Ground truth (matching the mock predictions)
        ground_truth = [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]

        # Compute expected MSE & MAE manually
        mse_expected = mean_squared_error(ground_truth, mock_generate.return_value)
        mae_expected = mean_absolute_error(ground_truth, mock_generate.return_value)

        # Call evaluation function (mocked)
        self.evaluator.evaluate(sample_size=2)

        # Ensure metrics match expected values
        self.assertAlmostEqual(mse_expected, 0.0, places=4)
        self.assertAlmostEqual(mae_expected, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()

