import requests

BASE_URL = "https://datasets-server.huggingface.co"
DATASET_ID = "clapAI/MultiLingualSentiment"
LANGUAGES = ("id", "en")
LABELS = ("negative", "neutral", "positive")


def main():
    session = requests.Session()

    print("1. Memeriksa split dataset...")
    response = session.get(
        f"{BASE_URL}/splits",
        params={"dataset": DATASET_ID},
        timeout=45,
    )
    response.raise_for_status()

    splits = response.json().get("splits", [])
    for split in splits:
        print(f"  {split['config']} / {split['split']}")

    train_split = next(
        (item for item in splits if item["split"] == "train"),
        None,
    )
    if train_split is None:
        raise RuntimeError("Split train tidak ditemukan.")

    config = train_split["config"]
    print(f"\n2. Memeriksa ketersediaan label pada config: {config}")

    for language in LANGUAGES:
        for label in LABELS:
            where = (
                f'"language"=\'{language}\' '
                f'AND "label"=\'{label}\''
            )

            result = session.get(
                f"{BASE_URL}/filter",
                params={
                    "dataset": DATASET_ID,
                    "config": config,
                    "split": "train",
                    "where": where,
                    "offset": 0,
                    "length": 1,
                },
                timeout=60,
            )

            if not result.ok:
                print(
                    f"{language} | {label:<8} | "
                    f"API ERROR {result.status_code}: "
                    f"{result.text[:250]}"
                )
                continue

            payload = result.json()
            count = payload.get("num_rows_total", "unknown")
            print(f"{language} | {label:<8} | {count} baris tersedia")

    print("\nPemeriksaan selesai.")


if __name__ == "__main__":
    main()
