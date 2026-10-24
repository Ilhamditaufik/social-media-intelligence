"""Run transformer sentiment analysis on collected YouTube metadata."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sentiment.inference import predict_sentiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "youtube_videos.csv"
OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "youtube_videos_sentiment.csv"
)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"CSV YouTube tidak ditemukan: {INPUT_CSV}\n"
            "Jalankan youtube_to_csv.py terlebih dahulu."
        )

    df = pd.read_csv(INPUT_CSV)

    for column in ("title", "description"):
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("").astype(str)

    if df.empty:
        raise ValueError("CSV YouTube belum berisi video.")

    results = []
    total = len(df)

    for index, row in df.iterrows():
        # Analyze the title and description, not comments.
        text = f"{row['title']}. {row['description']}".strip()

        try:
            prediction = predict_sentiment(text)

            results.append({
                "sentiment": prediction["label"],
                "confidence": prediction["confidence"],
                "sentiment_model": prediction["model"],
                "analyzed_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "analysis_status": "success",
            })

            print(
                f"[{index + 1}/{total}] "
                f"{prediction['label'].upper()} "
                f"({prediction['confidence']:.3f}) — "
                f"{row['title'][:90]}"
            )

        except Exception as exc:
            results.append({
                "sentiment": "unknown",
                "confidence": None,
                "sentiment_model": "",
                "analyzed_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "analysis_status": f"error: {type(exc).__name__}",
            })

            print(f"[{index + 1}/{total}] Gagal: {exc}")

    result_df = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(results)],
        axis=1,
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n=== RINGKASAN ANALISIS ===")
    print("Total video:", len(result_df))
    print("Berhasil:", int((result_df["analysis_status"] == "success").sum()))
    print("Gagal:", int((result_df["analysis_status"] != "success").sum()))
    print("\nDistribusi sentimen:")
    print(result_df["sentiment"].value_counts().to_string())
    print("\nCSV tersimpan:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
