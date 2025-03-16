import unittest
import numpy as np
from src.preprocessor import LLMTIMEPreprocessor
from src.postprocess import LLMTIMEPostprocessor


class TestLLMTIME(unittest.TestCase):
    """
    Unit test suite for LLMTIMEPreprocessor and LLMTIMEPostprocessor.

    These tests validate the preprocessing and postprocessing steps of a
    numerical time-series dataset formatted for use with Qwen2.5-Instruct.
    The test suite includes:
    - Validation of scaling and formatting.
    - Verification of tokenization.
    - Ensuring postprocessing correctly cleans formatted sequences.
    - Round-trip consistency checks to verify that preprocessing and 
      postprocessing are reversible.
    """

    @classmethod
    def setUpClass(cls):
        """
        Class-level setup method. Instantiates the LLMTIMEPreprocessor 
        and LLMTIMEPostprocessor once for efficiency.
        
        This method ensures that:
        - The dataset (`lotka_volterra_data.h5`) is loaded only once.
        - Preprocessing and postprocessing objects share the same scale factor.
        """
        cls.preprocessor = LLMTIMEPreprocessor(file_path="lotka_volterra_data.h5")
        cls.postprocessor = LLMTIMEPostprocessor(file_path="lotka_volterra_data.h5")

    def test_preprocess_scale_format(self):
        """
        Test if the scaling and formatting process correctly converts numerical 
        data into LLMTIME-compatible string representations.

        This function checks:
        - The output type is a list.
        - The number of formatted sequences matches the input.
        - The output strings contain correct delimiters (commas for variables, 
          semicolons for time steps).

        Expected behavior:
        - Input: `np.array([[[0.5, 1.5], [0.6, 1.4], [0.7, 1.3]]])`
        - Expected output format: `"0.5,1.5;0.6,1.4;0.7,1.3"`
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
        Test whether tokenization converts formatted sequences into integer 
        token lists as expected.

        This function ensures:
        - Tokenized output is a list.
        - The list contains non-empty sequences.
        - Each sequence is represented as a list of token integers.

        Expected behavior:
        - Input: `"0.5,1.5;0.6,1.4;0.7,1.3"`
        - Output: A list of integers corresponding to tokenized values.
        """
        sample_sequence = ["0.5,1.5;0.6,1.4;0.7,1.3"]
        tokenized = self.preprocessor.tokenize(sample_sequence)
        self.assertIsInstance(tokenized, list)
        self.assertGreater(len(tokenized[0]), 0)  # Tokenized output should not be empty
        self.assertIsInstance(tokenized[0], list)  # Tokenized output should be a list of integers

    def test_postprocess_clean_text(self):
        """
        Validate the postprocessing function that cleans decoded text.

        This function ensures:
        - Extra spaces around decimal points are removed.
        - Unwanted spaces around commas and semicolons are fixed.

        Expected behavior:
        - Input: `"1 .23 , 0 .567 ; 2.34 , 1.89"`
        - Expected Output: `"1.23,0.567;2.34,1.89"`
        """
        raw_text = "1 .23 , 0 .567 ; 2.34 , 1.89"
        cleaned_text = self.postprocessor.clean_decoded_text(raw_text)
        self.assertEqual(cleaned_text, "1.23,0.567;2.34,1.89")

    def test_postprocess_decode_sequence(self):
        """
        Validate the postprocessing function that converts tokenized sequences 
        back into numerical arrays.

        This test ensures:
        - The function correctly reconstructs a numerical array.
        - The output maintains the original variable count per timestep.

        Expected behavior:
        - Input: `[16, 13, 17, 18, 11, 15, 13, 20, 21, 22, 26, 17, 13, 18, 19, 11, 16, 13, 23, 24]`
        - Output: `[[1.23, 0.567], [2.34, 1.89]]` (rescaled accordingly).
        """
        sample_tokenized = [[16, 13, 17, 18, 11, 15, 13, 20, 21, 22, 26, 17, 13, 18, 19, 11, 16, 13, 23, 24]]
        decoded = self.postprocessor.decode_sequence(sample_tokenized[0])
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.shape[1], 2)  # Should have two variables per timestep

    def test_round_trip_consistency(self):
        """
        Ensure that a time series data sample remains numerically consistent 
        after a full preprocess → postprocess cycle.

        This test follows these steps:
        1. Scale and format a sample time series using preprocessing.
        2. Tokenize the formatted output.
        3. Decode the tokenized sequence back into numerical values.
        4. Validate that:
           - The decoded shape matches the input shape.
           - The numerical values are close to the original ones before preprocessing.

        Expected behavior:
        - If preprocessing and postprocessing are correct, the final output should 
          closely resemble the original input (within a small tolerance due to scaling).
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
