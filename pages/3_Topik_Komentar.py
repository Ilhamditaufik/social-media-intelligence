from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Topic Sentiment Intelligence",
    page_icon="🔎",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

COMMENTS_FILE = DATA_DIR / "youtube_comments_topics.csv"
TOPICS_FILE = DATA_DIR / "youtube_topics_summary.csv"
SUMMARY_FILE = DATA_DIR / "youtube_topic_sentiment_summary.csv"

SENTIMENT_ORDER = ["Positif", "Netral", "Negatif"]


@st.cache_data
def load_data():
    comments = pd.read_csv(COMMENTS_FILE, low_memory=False)
    topics = pd.read_csv(TOPICS_FILE, low_memory=False)
    return comments, topics


def find_sentiment_column(columns):
    candidates = [
        "sentiment",
        "predicted_sentiment",
        "sentiment_label",
        "label_sentiment",
        "prediction",
        "predicted_label",
        "label",
    ]

    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    for column in columns:
        name = str(column).strip().lower()
        if (
            "sentiment" in name
            and not any(
                word in name
                for word in ["confidence", "score", "probability"]
            )
        ):
            return column

    return None


def normalize_sentiment(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"positive", "positif", "pos", "1"}:
        return "Positif"

    if value in {"neutral", "netral", "neu", "0"}:
        return "Netral"

    if value in {"negative", "negatif", "neg", "-1"}:
        return "Negatif"

    if "positive" in value or "positif" in value:
        return "Positif"

    if "negative" in value or "negatif" in value:
        return "Negatif"

    if "neutral" in value or "netral" in value:
        return "Netral"

    return None


st.title("🔎 Topic Sentiment Intelligence")
st.caption(
    "Menggabungkan penemuan topik BERTopic dengan prediksi sentimen "
    "komentar YouTube."
)

if not COMMENTS_FILE.exists() or not TOPICS_FILE.exists():
    st.error(
        "File hasil BERTopic belum ditemukan. "
        "Jalankan pemodelan topik terlebih dahulu."
    )
    st.stop()

try:
    comments, topics = load_data()
except Exception as exc:
    st.error(f"Gagal membaca dataset: {exc}")
    st.stop()

required_columns = {"Topic", "Count", "Name"}

if not required_columns.issubset(topics.columns):
    st.error(
        "Kolom Topic, Count, dan Name belum lengkap pada "
        "youtube_topics_summary.csv."
    )
    st.stop()

if "topic_id" not in comments.columns:
    st.error("Kolom topic_id tidak ditemukan pada data komentar.")
    st.stop()

sentiment_col = find_sentiment_column(comments.columns)

if sentiment_col is None:
    st.error(
        "Kolom sentimen tidak terdeteksi pada dataset komentar. "
        "Kolom yang tersedia:"
    )
    st.write(comments.columns.tolist())
    st.stop()

comments["topic_id"] = pd.to_numeric(
    comments["topic_id"], errors="coerce"
)
comments = comments.dropna(subset=["topic_id"]).copy()
comments["topic_id"] = comments["topic_id"].astype(int)

comments["sentiment_display"] = comments[sentiment_col].apply(
    normalize_sentiment
)

topic_map = {
    int(row["Topic"]): str(row["Name"])
    for _, row in topics.dropna(subset=["Topic"]).iterrows()
}

comments["topic_name"] = comments["topic_id"].apply(
    lambda topic_id: (
        "Outlier (-1)"
        if topic_id == -1
        else topic_map.get(topic_id, f"Topik {topic_id}")
    )
)

valid_sentiments = comments[
    comments["sentiment_display"].isin(SENTIMENT_ORDER)
].copy()

total = len(comments)
classified = len(valid_sentiments)
unmapped = total - classified

counts = valid_sentiments["sentiment_display"].value_counts()

positive_count = int(counts.get("Positif", 0))
neutral_count = int(counts.get("Netral", 0))
negative_count = int(counts.get("Negatif", 0))

st.subheader("Ringkasan Sentimen")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Komentar", f"{total:,}")
c2.metric("Positif", f"{positive_count:,}")
c3.metric("Netral", f"{neutral_count:,}")
c4.metric("Negatif", f"{negative_count:,}")

if unmapped:
    st.warning(
        f"{unmapped} komentar tidak memiliki label sentimen "
        "yang dikenali. Komentar tersebut tidak dimasukkan "
        "ke dalam grafik sentimen."
    )

st.divider()
st.subheader("Sentimen per Topik")

if valid_sentiments.empty:
    st.warning(
        "Belum ada label sentimen yang berhasil dikenali. "
        f"Kolom yang terdeteksi: {sentiment_col}"
    )
    st.stop()

summary = (
    valid_sentiments
    .groupby(["topic_id", "topic_name", "sentiment_display"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    .reset_index()
)

summary["Total"] = summary[SENTIMENT_ORDER].sum(axis=1)
summary = summary.sort_values("Total", ascending=False)

chart_data = summary.set_index("topic_name")[SENTIMENT_ORDER].copy()
chart_data.columns = ["Positif", "Netral", "Negatif"]

st.bar_chart(chart_data)

st.caption(
    "Grafik menunjukkan jumlah komentar dengan label sentimen "
    "tertentu pada setiap topik. Ini bukan ukuran persentase "
    "opini masyarakat."
)

st.subheader("Tabel Sentimen per Topik")

display_summary = summary[
    ["topic_id", "topic_name", "Positif", "Netral", "Negatif", "Total"]
].rename(
    columns={
        "topic_id": "ID Topik",
        "topic_name": "Nama Topik",
    }
)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)

try:
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    st.success(
        "Ringkasan berhasil disimpan ke "
        "data/processed/youtube_topic_sentiment_summary.csv"
    )
except OSError as exc:
    st.warning(f"Ringkasan belum dapat disimpan: {exc}")

st.divider()
st.subheader("Eksplorasi Komentar")

topic_options = sorted(comments["topic_id"].unique().tolist())

def format_topic(topic_id):
    if topic_id == -1:
        return "Outlier (-1)"
    return f"Topik {topic_id}: {topic_map.get(topic_id, 'Belum bernama')}"

selected_topic = st.selectbox(
    "Pilih topik",
    topic_options,
    format_func=format_topic
)

filtered = comments[comments["topic_id"] == selected_topic].copy()

text_candidates = [
    "text",
    "comment_text",
    "comment",
    "clean_text",
    "cleaned_text",
    "content",
    "message",
]

text_col = next(
    (column for column in text_candidates if column in filtered.columns),
    None
)

preferred_columns = []

if text_col:
    preferred_columns.append(text_col)

preferred_columns.extend([
    sentiment_col,
    "sentiment_display",
    "confidence",
    "topic_id",
    "topic_probability",
])

display_columns = list(dict.fromkeys(
    column for column in preferred_columns
    if column in filtered.columns
))

st.caption(f"{len(filtered)} komentar ditemukan pada topik ini.")

st.dataframe(
    filtered[display_columns] if display_columns else filtered,
    use_container_width=True,
    hide_index=True
)

st.info(
    "Interpretasi tetap perlu memeriksa isi komentar. "
    "Nama topik berupa kata-kata representatif, sedangkan "
    "label sentimen merupakan prediksi model. Ukuran dataset "
    "yang kecil membatasi generalisasi hasil."
)
