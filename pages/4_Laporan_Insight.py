from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Laporan Insight",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

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
        "sentiment_display",
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
        if "sentiment" in name and not any(
            word in name for word in ["score", "confidence", "probability"]
        ):
            return column

    return None


def normalize_sentiment(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"positive", "positif", "pos", "label_1", "1"}:
        return "Positif"

    if value in {"neutral", "netral", "neu", "label_0", "0"}:
        return "Netral"

    if value in {"negative", "negatif", "neg", "label_2", "-1"}:
        return "Negatif"

    if "positive" in value or "positif" in value:
        return "Positif"

    if "negative" in value or "negatif" in value:
        return "Negatif"

    if "neutral" in value or "netral" in value:
        return "Netral"

    return None


def build_summary(comments, topic_map):
    rows = []

    for topic_id, group in comments.groupby("topic_id", sort=True):
        counts = (
            group["sentiment_display"]
            .value_counts()
            .reindex(SENTIMENT_ORDER, fill_value=0)
        )

        topic_id = int(topic_id)
        total = int(counts.sum())

        dominant = (
            str(counts.idxmax())
            if total > 0
            else "Tidak tersedia"
        )

        rows.append({
            "ID Topik": topic_id,
            "Nama Topik": (
                "Outlier"
                if topic_id == -1
                else topic_map.get(topic_id, f"Topik {topic_id}")
            ),
            "Total Berlabel": total,
            "Positif": int(counts["Positif"]),
            "Netral": int(counts["Netral"]),
            "Negatif": int(counts["Negatif"]),
            "Sentimen Dominan": dominant,
            "Persentase Positif (%)": (
                round(100 * counts["Positif"] / total, 2)
                if total else 0.0
            ),
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "Total Berlabel", ascending=False
        ).reset_index(drop=True)

    return result


st.title("📊 Laporan Insight Media Sosial")
st.caption(
    "Ringkasan otomatis dari prediksi sentimen dan pengelompokan "
    "topik komentar YouTube."
)

if not COMMENTS_FILE.exists() or not TOPICS_FILE.exists():
    st.error(
        "Data hasil BERTopic belum lengkap. Pastikan file berikut tersedia: "
        "youtube_comments_topics.csv dan youtube_topics_summary.csv."
    )
    st.stop()

try:
    comments, topics = load_data()
except Exception as exc:
    st.error(f"Gagal membaca data: {exc}")
    st.stop()

if "topic_id" not in comments.columns:
    st.error("Kolom topic_id tidak ditemukan.")
    st.stop()

if not {"Topic", "Name"}.issubset(topics.columns):
    st.error("Kolom Topic dan Name tidak ditemukan di ringkasan topik.")
    st.stop()

sentiment_col = find_sentiment_column(comments.columns)

if sentiment_col is None:
    st.error(
        "Kolom sentimen tidak terdeteksi. Kolom dataset: "
        + ", ".join(map(str, comments.columns))
    )
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

total_comments = len(comments)
outliers = int((comments["topic_id"] == -1).sum())
topic_count = len({
    int(value) for value in comments["topic_id"] if value != -1
})

classified = comments[
    comments["sentiment_display"].isin(SENTIMENT_ORDER)
].copy()

unclassified_count = total_comments - len(classified)

sentiment_counts = (
    classified["sentiment_display"]
    .value_counts()
    .reindex(SENTIMENT_ORDER, fill_value=0)
)

summary = build_summary(classified, topic_map)

generated_at = datetime.now().astimezone().isoformat(
    timespec="seconds"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Komentar", f"{total_comments:,}")
col2.metric("Komentar Berlabel", f"{len(classified):,}")
col3.metric("Jumlah Topik", topic_count)
col4.metric("Outlier", f"{outliers:,}")

st.subheader("Distribusi Sentimen")

sentiment_chart = pd.DataFrame({
    "Sentimen": SENTIMENT_ORDER,
    "Jumlah": [
        int(sentiment_counts[label])
        for label in SENTIMENT_ORDER
    ],
}).set_index("Sentimen")

st.bar_chart(sentiment_chart, y="Jumlah")

if unclassified_count:
    st.warning(
        f"{unclassified_count} komentar tidak mempunyai label sentimen "
        "yang dikenali dan tidak disertakan dalam perhitungan sentimen."
    )

st.subheader("Insight Utama")

if summary.empty:
    st.warning("Belum ada data dengan label sentimen yang dikenali.")
else:
    regular_topics = summary[summary["ID Topik"] != -1]
    if not regular_topics.empty:
        top_topic = regular_topics.iloc[0]
        st.write(
            f"Topik dengan komentar berlabel terbanyak: "
            f"**{top_topic['Nama Topik']}** "
            f"({int(top_topic['Total Berlabel'])} komentar)."
        )

    if len(classified):
        top_sentiment = str(sentiment_counts.idxmax())
        percentage = round(
            100 * int(sentiment_counts[top_sentiment]) / len(classified),
            2
        )
        st.write(
            f"Sentimen yang paling banyak terdeteksi: **{top_sentiment}**, "
            f"sebanyak {int(sentiment_counts[top_sentiment])} komentar "
            f"({percentage}% dari komentar berlabel)."
        )

st.subheader("Rincian Sentimen per Topik")
st.dataframe(summary, use_container_width=True, hide_index=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)

topic_records = summary.to_dict(orient="records")

report_data = {
    "generated_at": generated_at,
    "source_file": str(COMMENTS_FILE.relative_to(BASE_DIR)),
    "total_comments": int(total_comments),
    "sentiment_classified_comments": int(len(classified)),
    "unrecognized_sentiment_comments": int(unclassified_count),
    "topic_count_excluding_outliers": int(topic_count),
    "outlier_comments": int(outliers),
    "sentiment_distribution": {
        label: int(sentiment_counts[label])
        for label in SENTIMENT_ORDER
    },
    "topic_summary": topic_records,
    "limitations": [
        "Sentimen merupakan prediksi otomatis, bukan anotasi manusia.",
        "Nama topik adalah representasi kata, bukan kesimpulan semantik final.",
        "Sampel komentar yang kecil dan berasal dari video yang dipilih "
        "tidak mewakili seluruh opini masyarakat.",
        "Komentar merupakan data publik, tetapi hasil ini tetap perlu "
        "digunakan secara bertanggung jawab."
    ],
}

markdown_lines = [
    "# Laporan Insight Media Sosial",
    "",
    f"Waktu pembuatan: {generated_at}",
    "",
    "## Ringkasan",
    f"- Total komentar yang dianalisis: {total_comments}",
    f"- Komentar dengan sentimen dikenali: {len(classified)}",
    f"- Komentar tanpa label sentimen dikenali: {unclassified_count}",
    f"- Jumlah topik selain outlier: {topic_count}",
    f"- Komentar outlier BERTopic: {outliers}",
    "",
    "## Distribusi Sentimen",
]

for label in SENTIMENT_ORDER:
    markdown_lines.append(
        f"- {label}: {int(sentiment_counts[label])}"
    )

markdown_lines += [
    "",
    "## Sentimen per Topik",
    "",
    "| ID | Nama Topik | Total | Positif | Netral | Negatif | Dominan |",
    "|---:|---|---:|---:|---:|---:|---|",
]

for _, row in summary.iterrows():
    safe_name = str(row["Nama Topik"]).replace("|", "/")
    markdown_lines.append(
        f"| {int(row['ID Topik'])} | {safe_name} "
        f"| {int(row['Total Berlabel'])} "
        f"| {int(row['Positif'])} "
        f"| {int(row['Netral'])} "
        f"| {int(row['Negatif'])} "
        f"| {row['Sentimen Dominan']} |"
    )

markdown_lines += [
    "",
    "## Batasan Interpretasi",
    "",
    "- Label sentimen merupakan prediksi model otomatis.",
    "- Nama topik perlu ditafsirkan dan diperiksa secara manual.",
    "- Sampel yang kecil tidak cukup untuk mewakili seluruh masyarakat.",
    "- Jumlah komentar per topik tidak dengan sendirinya menunjukkan "
    "tingkat kepentingan atau kebenaran suatu opini.",
    "",
]

report_markdown = "\n".join(markdown_lines)
report_json = json.dumps(
    report_data,
    ensure_ascii=False,
    indent=2,
    default=str
)

st.subheader("Unduh Laporan")

d1, d2, d3 = st.columns(3)

with d1:
    st.download_button(
        "Unduh laporan Markdown",
        data=report_markdown,
        file_name="social_media_insight_report.md",
        mime="text/markdown",
        use_container_width=True
    )

with d2:
    st.download_button(
        "Unduh laporan JSON",
        data=report_json,
        file_name="social_media_insight_report.json",
        mime="application/json",
        use_container_width=True
    )

with d3:
    st.download_button(
        "Unduh ringkasan CSV",
        data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="youtube_topic_sentiment_summary.csv",
        mime="text/csv",
        use_container_width=True
    )

# Simpan salinan laporan untuk dokumentasi proyek.
(REPORT_DIR / "social_media_insight_report.md").write_text(
    report_markdown,
    encoding="utf-8"
)

(REPORT_DIR / "social_media_insight_report.json").write_text(
    report_json,
    encoding="utf-8"
)

st.caption(
    "Laporan ini menyajikan hasil agregat. Angka sentimen bukan "
    "pengukuran opini masyarakat secara representatif."
)
