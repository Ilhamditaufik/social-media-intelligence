import csv
import json
from pathlib import Path

from collectors.youtube_collector import normalize_search_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "youtube_videos.csv"

FIELDS = [
    "platform",
    "video_id",
    "title",
    "description",
    "channel_title",
    "published_at",
    "collected_at",
    "url",
    "raw_file",
]


def main():
    files = sorted(RAW_DIR.glob("youtube_search_*.json"))

    if not files:
        raise SystemExit("Belum ada file JSON YouTube di data/raw.")

    records_by_id = {}

    for path in files:
        try:
            with path.open(encoding="utf-8") as file:
                payload = json.load(file)

            records = normalize_search_payload(payload)

            for record in records:
                record["raw_file"] = path.name
                records_by_id[record["video_id"]] = record

            print(f"{path.name}: {len(records)} video")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"File dilewati: {path.name} ({exc})")

    records = list(records_by_id.values())

    if not records:
        raise SystemExit("Tidak ditemukan video valid untuk CSV.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nTotal file JSON: {len(files)}")
    print(f"Total video unik: {len(records)}")
    print(f"CSV tersimpan: {OUTPUT}")


if __name__ == "__main__":
    main()
