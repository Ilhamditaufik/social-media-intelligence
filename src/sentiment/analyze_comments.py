"""Analyze collected YouTube comments with the transformer model."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sentiment.inference import predict_sentiment

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = ROOT / "data" / "processed" / "youtube_comments.csv"
OUTPUT_CSV = (
    ROOT / "data" / "processed" / "youtube_comments_sentiment.csv"
)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV).fillna("")

    if "text" not in df.columns:
        raise ValueError("Kolom text tidak ditemukan di CSV komentar.")

    results = []
    total = len(df)

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        text = str(row["text"]).strip()

        if not text:
            results.append({
                "sentiment": "unknown",
                "confidence": None,
                "sentiment_model": "",
                "analysis_status": "empty_text",
            })
            continue

        try:
            prediction = predict_sentiment(text)

            results.append({
                "sentiment": prediction["label"],
                "confidence": prediction["confidence"],
                "sentiment_model": prediction["model"],
                "analysis_status": "success",
            })

            print(
                f"[{position}/{total}] "
                f"{prediction['label'].upper()} "
                f"({prediction['confidence']:.3f}) | "
                f"{text[:75]}"
            )

        except Exception as exc:
            results.append({
                "sentiment": "unknown",
                "confidence": None,
                "sentiment_model": "",
                "analysis_status": f"error:{type(exc).__name__}",
            })

            print(
                f"[{position}/{total}] Gagal: "
                f"{type(exc).__name__}: {exc}"
            )

    result_df = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(results)],
        axis=1,
    )

    result_df["analyzed_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    successful = int((result_df["analysis_status"] == "success").sum())

    print("\n=== RINGKASAN SENTIMEN KOMENTAR ===")
    print("Total komentar:", len(result_df))
    print("Berhasil dianalisis:", successful)
    print("Gagal/teks kosong:", len(result_df) - successful)

    print("\nDistribusi prediksi:")
    print(
        result_df["sentiment"]
        .value_counts()
        .to_string()
    )

    print("\nDistribusi per video:")
    print(
        result_df.groupby(
            ["video_title", "sentiment"]
        ).size().to_string()
    )

    print("\nCSV tersimpan:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
