import unittest
import numpy as np
from src.preprocessor import LLMTIMEPreprocessor
from src.postprocess import LLMTIMEPostprocessor


class TestLLMTIME(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the preprocessor and postprocessor once for all tests.
        """
        cls.preprocessor = LLMTIMEPreprocessor(file_path="lotka_volterra_data.h5")
        cls.postprocessor = LLMTIMEPostprocessor(file_path="lotka_volterra_data.h5")

    def test_preprocess_scale_format(self):
        """
        Test if scaling and formatting work as expected.
        """
        sample_data = np.array([
            [[0.5, 1.5], [0.6, 1.4], [0.7, 1.3]]
        ])  # Shape (1, 3, 2)

        formatted = self.preprocessor.scale_and_format(sample_data)
        self.assertIsInstance(formatted, list)
        self.assertEqual(len(formatted), 1)  # Only one sequence
        self.assertTrue(";" in formatted[0])  # Ensures timestamps are separated
        self.assertTrue("," in formatted[0])  # Ensures variables are separated

    def test_preprocess_tokenization(self):
        """
        Test if tokenization works correctly.
        """
        sample_sequence = ["0.5,1.5;0.6,1.4;0.7,1.3"]
        tokenized = self.preprocessor.tokenize(sample_sequence)
        self.assertIsInstance(tokenized, list)
        self.assertGreater(len(tokenized[0]), 0)  # Tokenized output should not be empty
        self.assertIsInstance(tokenized[0], list)  # Tokenized output should be a list of integers

    def test_postprocess_clean_text(self):
        """
        Test if cleaning function correctly fixes spacing issues.
        """
        raw_text = "1 .23 , 0 .567 ; 2.34 , 1.89"
        cleaned_text = self.postprocessor.clean_decoded_text(raw_text)
        self.assertEqual(cleaned_text, "1.23,0.567;2.34,1.89")

    def test_postprocess_decode_sequence(self):
        """
        Test if decoding works correctly.
        """
        sample_tokenized = [[16, 13, 17, 18, 11, 15, 13, 20, 21, 22, 26, 17, 13, 18, 19, 11, 16, 13, 23, 24]]
        decoded = self.postprocessor.decode_sequence(sample_tokenized[0])
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.shape[1], 2)  # Should have two variables per timestep

    def test_round_trip_consistency(self):
        """
        Test if a sample sequence remains consistent through preprocess → postprocess.
        """
        sample_data = np.array([
            [[0.5, 1.5], [0.6, 1.4], [0.7, 1.3]]
        ])  # Shape (1, 3, 2)

        # Preprocess
        formatted = self.preprocessor.scale_and_format(sample_data)
        tokenized = self.preprocessor.tokenize(formatted)

        # Postprocess
        decoded = self.postprocessor.decode_sequence(tokenized[0])

        # Ensure dimensions match
        self.assertEqual(decoded.shape, sample_data.shape)

        # Ensure values are approximately the same after scaling correction
        np.testing.assert_almost_equal(decoded, sample_data, decimal=2)


if __name__ == "__main__":
    unittest.main()

