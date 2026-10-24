"""Collect public YouTube video metadata using YouTube Data API v3."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_URL = "https://www.googleapis.com/youtube/v3/search"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def normalize_search_payload(payload):
    """Normalize YouTube search results without retaining channel identifiers."""
    records = []

    for item in payload.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})

        if not video_id:
            continue

        records.append({
            "platform": "youtube",
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        })

    return records


def search_videos(query, max_results=10, language=None):
    """Search public videos. Requires YOUTUBE_API_KEY in project .env."""
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY belum tersedia. Isi file .env di folder proyek."
        )

    if not query.strip():
        raise ValueError("Query pencarian tidak boleh kosong.")

    if not 1 <= max_results <= 50:
        raise ValueError("max_results harus antara 1 dan 50.")

    params = {
        "key": api_key,
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "safeSearch": "moderate",
    }

    if language:
        params["relevanceLanguage"] = language

    response = requests.get(API_URL, params=params, timeout=(10, 45))

    if not response.ok:
        try:
            detail = response.json().get("error", {}).get("message", "")
        except ValueError:
            detail = response.text[:300]

        raise RuntimeError(
            f"YouTube API HTTP {response.status_code}: {detail}"
        )

    payload = response.json()

    # Preserve the raw API response for reproducibility and auditing.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RAW_DIR / f"youtube_search_{stamp}.json"

    raw_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    records = normalize_search_payload(payload)

    print(f"Video ditemukan : {len(records)}")
    print(f"Raw JSON         : {raw_path}")

    for record in records:
        print(
            f"- {record['title'][:100]} "
            f"[{record['published_at']}]"
        )

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Collect public YouTube video metadata."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    try:
        search_videos(
            query=args.query,
            max_results=args.max_results,
            language=args.language,
        )
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
