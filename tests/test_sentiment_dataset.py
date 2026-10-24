import unittest

from collectors.sentiment_dataset import (
    add_record_to_buckets,
    create_buckets,
    normalize_record,
)


class SentimentDatasetTests(unittest.TestCase):
    def test_normalizes_indonesian_record(self):
        record = {
            "text": "Halo @sobat https://example.com #BahasaIndonesia 😊",
            "label": "Positive",
            "language": "id",
            "source": "test",
            "domain": "social",
        }

        result = normalize_record(record)

        self.assertIsNotNone(result)
        self.assertEqual(result["language"], "id")
        self.assertEqual(result["label"], "positive")
        self.assertIn("USER", result["text"])
        self.assertIn("URL", result["text"])

    def test_rejects_unsupported_language(self):
        self.assertIsNone(
            normalize_record({
                "text": "This is a valid example sentence.",
                "label": "positive",
                "language": "fr",
            })
        )

    def test_rejects_unsupported_label(self):
        self.assertIsNone(
            normalize_record({
                "text": "This is a valid example sentence.",
                "label": "mixed",
                "language": "en",
            })
        )

    def test_rejects_empty_text(self):
        self.assertIsNone(
            normalize_record({
                "text": "",
                "label": "positive",
                "language": "en",
            })
        )

    def test_respects_per_group_limit(self):
        buckets = create_buckets()
        record = {
            "text": "Pelayanan maskapai ini sangat baik.",
            "label": "positive",
            "language": "id",
        }

        self.assertTrue(add_record_to_buckets(buckets, record, 1))
        self.assertFalse(add_record_to_buckets(buckets, record, 1))
        self.assertEqual(len(buckets[("id", "positive")]), 1)

    def test_keeps_language_label_groups_separate(self):
        buckets = create_buckets()

        english = {
            "text": "The airline service was excellent.",
            "label": "positive",
            "language": "en",
        }
        indonesian = {
            "text": "Pelayanan maskapai ini sangat baik.",
            "label": "positive",
            "language": "id",
        }

        self.assertTrue(add_record_to_buckets(buckets, english))
        self.assertTrue(add_record_to_buckets(buckets, indonesian))
        self.assertEqual(len(buckets[("en", "positive")]), 1)
        self.assertEqual(len(buckets[("id", "positive")]), 1)


if __name__ == "__main__":
    unittest.main()
