from pathlib import Path
import json
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN


def main():
    base_dir = Path.cwd()
    input_file = base_dir / "data/processed/youtube_comments_sentiment.csv"
    output_dir = base_dir / "data/processed"
    report_dir = base_dir / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {input_file}"
        )

    df = pd.read_csv(input_file, low_memory=False)

    # Cari kolom teks komentar.
    candidates = [
        "text", "comment_text", "comment", "clean_text",
        "cleaned_text", "content", "message"
    ]

    text_col = next(
        (col for col in candidates if col in df.columns),
        None
    )

    if text_col is None:
        raise ValueError(
            "Kolom teks komentar tidak ditemukan. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    # Bersihkan teks kosong dan simpan indeks baris asal.
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    valid_df = df[df[text_col].str.len() > 0].copy()
    valid_df.reset_index(drop=True, inplace=True)

    texts = valid_df[text_col].tolist()

    if len(texts) < 10:
        raise ValueError(
            f"Komentar valid hanya {len(texts)}. "
            "Diperlukan minimal 10 komentar untuk eksplorasi topik."
        )

    print(f"Dataset: {input_file}")
    print(f"Jumlah komentar valid: {len(texts)}")
    print("Memuat model embedding multilingual di CPU...")

    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )

    embeddings = embedding_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    umap_model = UMAP(
        n_neighbors=min(15, len(texts) - 1),
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    cluster_model = HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        metric="euclidean",
        prediction_data=True
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w+\b"
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",
        verbose=True
    )

    print("Menemukan topik dari komentar...")
    topics, probabilities = topic_model.fit_transform(
        texts, embeddings
    )

    valid_df["topic_id"] = topics

    if probabilities is not None:
        if getattr(probabilities, "ndim", 1) == 1:
            valid_df["topic_probability"] = probabilities
        else:
            valid_df["topic_probability"] = probabilities.max(axis=1)
    else:
        valid_df["topic_probability"] = None

    topic_info = topic_model.get_topic_info()

    output_comments = output_dir / "youtube_comments_topics.csv"
    output_topics = output_dir / "youtube_topics_summary.csv"
    output_report = report_dir / "topic_model_report.json"

    valid_df.to_csv(output_comments, index=False)
    topic_info.to_csv(output_topics, index=False)

    report = {
        "input_file": str(input_file),
        "text_column": text_col,
        "total_input_rows": int(len(df)),
        "valid_comments_analyzed": int(len(valid_df)),
        "topic_count_excluding_outliers": int(
            (topic_info["Topic"] != -1).sum()
        ),
        "outlier_comments": int((valid_df["topic_id"] == -1).sum()),
        "topics": topic_info.where(
            pd.notna(topic_info), None
        ).to_dict(orient="records")
    }

    with output_report.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, default=str)

    print("\n=== PEMODELAN TOPIK SELESAI ===")
    print(f"Komentar berlabel topik : {output_comments}")
    print(f"Ringkasan topik         : {output_topics}")
    print(f"Laporan JSON            : {output_report}")
    print("\nRingkasan topik:")
    print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))
    print(
        "\nCatatan: topik -1 berarti komentar tidak masuk "
        "ke klaster topik tertentu."
    )


if __name__ == "__main__":
    main()
