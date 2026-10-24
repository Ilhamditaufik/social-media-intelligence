from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data" / "external" / "multilingual_sentiment_sample.csv"

sys.path.insert(0, str(PROJECT_DIR / "src"))

from preprocessing.text_cleaner import clean_text
from sentiment.inference import predict_sentiment

st.set_page_config(
    page_title="Nexvia | Social Media Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Social Media Intelligence")
st.caption(
    "Nexvia Technologies · DS-03 · "
    "Multilingual Sentiment Analysis Prototype"
)


@st.cache_data
def load_dataset():
    if not CSV_PATH.exists():
        return pd.DataFrame(
            columns=["text", "label", "language", "source", "domain"]
        )

    df = pd.read_csv(CSV_PATH)
    required = {"text", "label", "language", "source", "domain"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {sorted(missing)}")

    for col in required:
        df[col] = df[col].fillna("").astype(str)

    df["label"] = df["label"].str.lower()
    df["language"] = df["language"].str.lower()

    return df[df["text"].str.strip().ne("")].copy()


try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Gagal membaca dataset: {exc}")
    df = pd.DataFrame()

overview_tab, dataset_tab, ai_tab = st.tabs(
    ["🏠 Ringkasan", "🗂️ Dataset", "🧠 AI Sentiment"]
)

with overview_tab:
    a, b, c, d = st.columns(4)

    a.metric("Teks tersedia", len(df))
    b.metric(
        "Bahasa",
        df["language"].nunique() if not df.empty else 0,
    )
    c.metric(
        "Label",
        df["label"].nunique() if not df.empty else 0,
    )
    d.metric(
        "Model AI",
        "Transformer",
    )

    st.subheader("Distribusi sentimen")

    if not df.empty:
        sentiment_counts = (
            df["label"]
            .value_counts()
            .reindex(
                ["positive", "neutral", "negative"],
                fill_value=0,
            )
        )
        st.bar_chart(sentiment_counts)

        st.subheader("Distribusi bahasa")
        st.bar_chart(df["language"].value_counts())

        if "neutral" not in set(df["label"]):
            st.warning(
                "Sampel CSV belum memiliki label netral. "
                "Model transformer tetap mendukung prediksi netral, "
                "tetapi sampel ini belum cukup untuk evaluasi model."
            )
    else:
        st.info(
            "CSV sampel belum tersedia. Fitur prediksi AI tetap "
            "dapat dicoba."
        )

    st.caption(
        "Statistik di atas menggambarkan sampel lokal, "
        "bukan keseluruhan percakapan media sosial."
    )


with dataset_tab:
    st.subheader("Jelajahi dataset")

    if df.empty:
        st.info("Belum ada dataset yang dapat ditampilkan.")
    else:
        f1, f2 = st.columns(2)

        language_options = ["Semua"] + sorted(df["language"].unique())
        label_options = ["Semua"] + sorted(df["label"].unique())

        language = f1.selectbox("Bahasa", language_options)
        label = f2.selectbox("Sentimen", label_options)

        view = df.copy()

        if language != "Semua":
            view = view[view["language"] == language]

        if label != "Semua":
            view = view[view["label"] == label]

        st.write(f"Menampilkan **{len(view)}** baris.")

        st.dataframe(
            view[["text", "language", "label", "source", "domain"]],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Unduh CSV",
            data=view.to_csv(index=False).encode("utf-8-sig"),
            file_name="sentiment_sample_filtered.csv",
            mime="text/csv",
        )


with ai_tab:
    st.subheader("Analisis sentimen menggunakan Transformer")

    st.write(
        "Masukkan teks berbahasa Indonesia atau Inggris untuk "
        "mendapatkan prediksi sentimen dari model Hugging Face."
    )

    st.info(
        "Model: nahiar/sentiment-analysis-v2. "
        "Inference berjalan lokal melalui Python Ubuntu."
    )

    with st.form("sentiment_form"):
        user_text = st.text_area(
            "Teks yang ingin dianalisis",
            placeholder=(
                "Contoh: Pelayanannya ramah, makanannya enak, "
                "dan harganya terjangkau."
            ),
            height=130,
        )

        submitted = st.form_submit_button(
            "🔍 Analisis Sentimen",
            type="primary",
        )

    if submitted:
        if not clean_text(user_text):
            st.error("Masukkan teks terlebih dahulu.")
        else:
            try:
                with st.spinner("Model sedang menganalisis teks..."):
                    result = predict_sentiment(user_text)

                labels = {
                    "positive": "😊 POSITIVE",
                    "neutral": "😐 NEUTRAL",
                    "negative": "😞 NEGATIVE",
                }

                st.divider()
                left, right = st.columns(2)

                left.metric(
                    "Prediksi sentimen",
                    labels.get(
                        result["label"],
                        result["label"].upper(),
                    ),
                )

                right.metric(
                    "Skor confidence model",
                    f"{result['confidence']:.2%}",
                )

                st.write("**Teks setelah preprocessing**")
                st.write(result["text"])

                st.caption(f"Model: {result['model']}")
                st.caption(
                    "Confidence bukan jaminan bahwa prediksi benar. "
                    "Model perlu dievaluasi pada dataset berlabel "
                    "yang terpisah sebelum digunakan untuk keputusan nyata."
                )

            except Exception as exc:
                st.error(f"Prediksi gagal: {exc}")

st.divider()
st.caption(
    "DS-03 Prototype · Data sampel lokal · "
    "Integrasi YouTube melalui halaman terpisah · "
    "Reddit API dan evaluasi model masih dalam pengembangan"
)
