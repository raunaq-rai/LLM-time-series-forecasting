import unittest
import numpy as np
from src.preprocessor import LLMTIMEPreprocessor


class TestLLMTIMEPreprocessor(unittest.TestCase):
    """
    Unit tests for the LLMTIMEPreprocessor class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the preprocessor once for all tests.
        """
        cls.preprocessor = LLMTIMEPreprocessor(file_path="lotka_volterra_data.h5")

    def test_preprocess_scale_format(self):
        """
        Test if scaling and formatting work as expected.
        Ensures:
        - The formatted output is a list.
        - The format correctly separates time steps (;) and variables (,).
        """
        sample_data = np.array([
            [[0.5, 1.5], [0.6, 1.4], [0.7, 1.3]]
        ])  # Shape (1, 3, 2)

        formatted = self.preprocessor.scale_and_format(sample_data)
        
        self.assertIsInstance(formatted, list)
        self.assertEqual(len(formatted), 1)  # One sequence
        self.assertIn(";", formatted[0])  # Timestamps separated
        self.assertIn(",", formatted[0])  # Variables separated

    def test_preprocess_tokenization(self):
        """
        Test if tokenization works correctly.
        Ensures:
        - The output is a list.
        - It contains tokenized integer values.
        """
        sample_sequence = ["0.5,1.5;0.6,1.4;0.7,1.3"]
        tokenized = self.preprocessor.tokenize(sample_sequence)

        self.assertIsInstance(tokenized, list)
        self.assertGreater(len(tokenized[0]), 0)  # Ensure non-empty tokens
        self.assertIsInstance(tokenized[0], list)  # List of token sequences

    def test_preprocess_pipeline(self):
        """
        Test if the full preprocessing pipeline works correctly.
        - Checks formatted output.
        - Ensures tokenized output is not empty.
        """
        formatted, tokenized = self.preprocessor.preprocess(sample_size=1)

        self.assertIsInstance(formatted, list)
        self.assertIsInstance(tokenized, list)
        self.assertGreater(len(tokenized[0]), 0)


if __name__ == "__main__":
    unittest.main()

