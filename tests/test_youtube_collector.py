import unittest

from collectors.youtube_collector import normalize_search_payload


class YouTubeCollectorTests(unittest.TestCase):
    def test_normalizes_video_metadata(self):
        payload = {
            "items": [{
                "id": {"kind": "youtube#video", "videoId": "abc123"},
                "snippet": {
                    "title": "Contoh video",
                    "description": "Video untuk pengujian",
                    "channelTitle": "Channel Demo",
                    "publishedAt": "2026-10-24T10:00:00Z",
                },
            }]
        }

        result = normalize_search_payload(payload)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["platform"], "youtube")
        self.assertEqual(result[0]["video_id"], "abc123")
        self.assertEqual(result[0]["title"], "Contoh video")
        self.assertIn("youtube.com/watch?v=abc123", result[0]["url"])

    def test_ignores_non_video_items(self):
        payload = {
            "items": [
                {"id": {"channelId": "channel1"}, "snippet": {}},
                {"id": {"playlistId": "playlist1"}, "snippet": {}},
            ]
        }
        self.assertEqual(normalize_search_payload(payload), [])

    def test_handles_empty_results(self):
        self.assertEqual(normalize_search_payload({"items": []}), [])

    def test_handles_missing_items(self):
        self.assertEqual(normalize_search_payload({}), [])


if __name__ == "__main__":
    unittest.main()
