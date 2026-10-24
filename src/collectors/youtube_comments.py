"""Collect a bounded sample of public YouTube comments."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
INPUT_CSV = ROOT / "data" / "processed" / "youtube_videos.csv"
OUTPUT_CSV = ROOT / "data" / "processed" / "youtube_comments.csv"
RAW_DIR = ROOT / "data" / "raw"


def main():
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: YOUTUBE_API_KEY tidak ditemukan di .env")

    if not INPUT_CSV.exists():
        raise SystemExit(f"ERROR: File video tidak ditemukan: {INPUT_CSV}")

    videos = pd.read_csv(INPUT_CSV).fillna("")
    if not {"video_id", "title"}.issubset(videos.columns):
        raise SystemExit("ERROR: CSV video harus memiliki video_id dan title.")

    videos = videos.drop_duplicates(subset=["video_id"]).head(10)

    session = requests.Session()
    records = []
    raw_responses = []

    for number, video in enumerate(videos.to_dict("records"), start=1):
        video_id = str(video["video_id"]).strip()
        title = str(video["title"])

        if not video_id:
            continue

        params = {
            "key": api_key,
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 20,
            "textFormat": "plainText",
            "order": "time",
        }

        try:
            response = session.get(API_URL, params=params, timeout=(10, 40))

            if not response.ok:
                try:
                    error = response.json().get("error", {})
                    reasons = [
                        item.get("reason", "")
                        for item in error.get("errors", [])
                    ]
                    message = error.get("message", "")
                except ValueError:
                    reasons = []
                    message = response.text[:200]

                print(
                    f"[{number}/{len(videos)}] Dilewati: {title[:65]} "
                    f"| HTTP {response.status_code} "
                    f"| {', '.join(reasons) or message[:120]}"
                )
                continue

            payload = response.json()
            items = payload.get("items", [])

            for item in items:
                top = item.get("snippet", {}).get("topLevelComment", {})
                snippet = top.get("snippet", {})
                text = snippet.get("textOriginal") or snippet.get("textDisplay") or ""

                if not text.strip():
                    continue

                record = {
                    "comment_id": top.get("id", item.get("id", "")),
                    "video_id": video_id,
                    "video_title": title,
                    "text": text.strip(),
                    "published_at": snippet.get("publishedAt", ""),
                    "like_count": snippet.get("likeCount", 0),
                    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                }

                records.append(record)

            # Store only the fields needed for analysis; exclude commenter names/IDs.
            raw_responses.append({
                "video_id": video_id,
                "video_title": title,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "comments": [
                    row for row in records if row["video_id"] == video_id
                ],
            })

            print(
                f"[{number}/{len(videos)}] "
                f"{len(items)} komentar | {title[:65]}"
            )

        except requests.RequestException as exc:
            print(
                f"[{number}/{len(videos)}] Gangguan jaringan untuk "
                f"{title[:65]}: {type(exc).__name__}"
            )

    # Deduplicate comments in case the input contains repeated videos.
    unique = {row["comment_id"]: row for row in records if row["comment_id"]}
    records = list(unique.values())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_CSV.write_text("", encoding="utf-8") if not records else None

    if records:
        pd.DataFrame(records).to_csv(
            OUTPUT_CSV, index=False, encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=[
            "comment_id", "video_id", "video_title", "text",
            "published_at", "like_count", "collected_at_utc",
        ]).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    raw_path = RAW_DIR / "youtube_comments_sample.json"
    raw_path.write_text(
        json.dumps(raw_responses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RINGKASAN ===")
    print("Video diperiksa:", len(videos))
    print("Komentar tersimpan:", len(records))
    print("CSV:", OUTPUT_CSV)
    print("JSON:", raw_path)


if __name__ == "__main__":
    main()
