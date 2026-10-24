"""Controlled multilingual sentiment dataset sampling."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from preprocessing.text_cleaner import clean_text

DATASET_ID = "clapAI/MultiLingualSentiment"
LANGUAGES = ("id", "en")
LABELS = ("negative", "neutral", "positive")
FIELDS = ("text", "label", "language", "source", "domain")


def normalize_record(record):
    """Validate and normalize one dataset record."""
    text = clean_text(record.get("text"))
    language = str(record.get("language") or "").strip().lower()
    label = str(record.get("label") or "").strip().lower()

    if not text or language not in LANGUAGES or label not in LABELS:
        return None

    return {
        "text": text,
        "label": label,
        "language": language,
        "source": str(record.get("source") or ""),
        "domain": str(record.get("domain") or ""),
    }


def create_buckets():
    """Create a separate sample bucket for each language-label pair."""
    return {
        (language, label): []
        for language in LANGUAGES
        for label in LABELS
    }


def add_record_to_buckets(buckets, record, per_group=25):
    """Add a valid record without exceeding its group limit."""
    if per_group < 1:
        raise ValueError("per_group must be at least 1")

    item = normalize_record(record)
    if item is None:
        return False

    key = (item["language"], item["label"])
    if len(buckets[key]) >= per_group:
        return False

    buckets[key].append(item)
    return True


def collect_sample(
    per_group=25,
    max_scan=100_000,
    buffer_size=10_000,
    seed=42,
    output="data/external/multilingual_sentiment_sample.csv",
):
    """Stream the dataset and save a bounded ID/EN sentiment sample."""
    if per_group < 1 or max_scan < 1 or buffer_size < 1:
        raise ValueError("Sampling limits must be positive")

    dataset = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=buffer_size)

    buckets = create_buckets()
    scanned = 0

    for record in dataset:
        scanned += 1
        add_record_to_buckets(buckets, record, per_group)

        if scanned % 10_000 == 0:
            print(f"Scanned {scanned:,} rows...")

        if all(len(rows) >= per_group for rows in buckets.values()):
            break

        if scanned >= max_scan:
            break

    records = [
        item
        for key in buckets
        for item in buckets[key]
    ]

    if not records:
        raise RuntimeError(
            "No valid Indonesian or English records found. "
            "Check dataset access and the dataset schema."
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nRows scanned: {scanned:,}")
    print(f"Records saved: {len(records):,}")
    print(f"CSV: {output_path}")

    print("\nLanguage-label distribution:")
    counts = Counter(
        (item["language"], item["label"])
        for item in records
    )

    for language in LANGUAGES:
        for label in LABELS:
            print(f"{language:>2} | {label:<8} | {counts[(language, label)]}")

    missing = [
        f"{language}/{label}"
        for (language, label), rows in buckets.items()
        if len(rows) == 0
    ]
    if missing:
        print("Missing groups:", ", ".join(missing))

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Sample multilingual sentiment data using streaming."
    )
    parser.add_argument("--per-group", type=int, default=25)
    parser.add_argument("--max-scan", type=int, default=100_000)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument(
        "--output",
        default="data/external/multilingual_sentiment_sample.csv",
    )
    args = parser.parse_args()

    collect_sample(
        per_group=args.per_group,
        max_scan=args.max_scan,
        buffer_size=args.buffer_size,
        output=args.output,
    )


if __name__ == "__main__":
    main()
