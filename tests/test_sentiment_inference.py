import unittest

from sentiment.inference import normalize_label


class SentimentInferenceTests(unittest.TestCase):
    def test_label_zero_is_neutral(self):
        self.assertEqual(normalize_label("LABEL_0"), "neutral")

    def test_label_one_is_positive(self):
        self.assertEqual(normalize_label("LABEL_1"), "positive")

    def test_label_two_is_negative(self):
        self.assertEqual(normalize_label("LABEL_2"), "negative")

    def test_normalizes_case(self):
        self.assertEqual(normalize_label("Positive"), "positive")

    def test_unknown_label_is_handled(self):
        self.assertEqual(normalize_label("UNEXPECTED"), "unknown")


if __name__ == "__main__":
    unittest.main()
