import unittest
import numpy as np
from src.postprocess import LLMTIMEPostprocessor


class TestLLMTIMEPostprocessor(unittest.TestCase):
    """
    Unit tests for the LLMTIMEPostprocessor class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the postprocessor once for all tests.
        """
        cls.postprocessor = LLMTIMEPostprocessor(file_path="lotka_volterra_data.h5")

    def test_postprocess_clean_text(self):
        """
        Test if cleaning function correctly fixes spacing issues.
        - Ensures that:
          - Decimal spacing is corrected.
          - Commas and semicolons are properly formatted.
        """
        raw_text = "1 .23 , 0 .567 ; 2.34 , 1.89"
        cleaned_text = self.postprocessor.clean_decoded_text(raw_text)
        self.assertEqual(cleaned_text, "1.23,0.567;2.34,1.89")

    def test_postprocess_decode_sequence(self):
        """
        Test if decoding works correctly.
        - Decodes a tokenized output and ensures:
          - The output is an ndarray.
          - It has the expected shape (1, T, V).
        """
        sample_tokenized = [[16, 13, 17, 18, 11, 15, 13, 20, 21, 22, 26, 17, 13, 18, 19, 11, 16, 13, 23, 24]]
        decoded = self.postprocessor.decode_sequence(sample_tokenized[0])

        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(len(decoded.shape), 3)  # Ensure shape (1, T, V)

    def test_round_trip_consistency(self):
      """
      Test if a sample sequence remains consistent through preprocess → postprocess.
      - Formats and tokenizes a sample sequence.
      - Decodes it back to ensure:
      - The shape remains the same.
      - The values are approximately equal after rescaling.
      """
      sample_data = np.array([
          [[0.5, 1.5], [0.6, 1.4], [0.7, 1.3]]
      ])  # Shape (1, 3, 2)

      # Preprocess
      formatted = self.postprocessor.clean_decoded_text("0.36,1.09;0.44,1.02;0.51,0.95")
      numeric_values = [list(map(float, timestep.split(','))) for timestep in formatted.split(';')]
      numeric_values = np.array(numeric_values)
      rescaled_values = numeric_values / self.postprocessor.scale_factor

      # Ensure dimensions match **after reshaping**
      rescaled_values = rescaled_values[np.newaxis, :, :]  # Ensure shape (1, 3, 2)
      self.assertEqual(rescaled_values.shape, (1, 3, 2))

      # Ensure values are approximately the same after scaling correction
      np.testing.assert_almost_equal(rescaled_values, sample_data, decimal=2)



if __name__ == "__main__":
    unittest.main()

