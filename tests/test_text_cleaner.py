import unittest

from preprocessing.text_cleaner import clean_text, detect_language


class TextCleanerTests(unittest.TestCase):
    def test_replaces_url_and_mention(self):
        result = clean_text(
            "Hai @sobat! Lihat https://example.com/page #BahasaIndonesia 😊"
        )
        self.assertIn("USER", result)
        self.assertIn("URL", result)
        self.assertIn("BahasaIndonesia", result)
        self.assertIn("😊", result)
        self.assertNotIn("https://", result)

    def test_removes_html_and_decodes_entities(self):
        result = clean_text("<p>Halo&nbsp; dunia!</p>")
        self.assertEqual(result, "Halo dunia!")

    def test_collapses_whitespace_and_control_characters(self):
        result = clean_text("Halo   dunia!\x00\nApa kabar?")
        self.assertEqual(result, "Halo dunia! Apa kabar?")

    def test_handles_empty_or_missing_text(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")

    def test_optional_lowercase(self):
        self.assertEqual(clean_text("Halo DUNIA", lowercase=True), "halo dunia")

    def test_short_language_detection_returns_unknown(self):
        self.assertEqual(detect_language("OK"), "unknown")


if __name__ == "__main__":
    unittest.main()
