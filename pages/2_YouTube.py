from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
VIDEO_CSV = ROOT / "data" / "processed" / "youtube_videos_sentiment.csv"
COMMENT_CSV = ROOT / "data" / "processed" / "youtube_comments_sentiment.csv"

st.set_page_config(
    page_title="YouTube Intelligence | DS-03",
    page_icon="▶️",
    layout="wide",
)

st.title("▶️ YouTube Intelligence")
st.caption(
    "DS-03 · YouTube Data API v3 · Transformer Sentiment Analysis"
)


@st.cache_data
def read_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def clean_columns(df):
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna("")
    return df


videos = clean_columns(read_csv(str(VIDEO_CSV)))
comments = clean_columns(read_csv(str(COMMENT_CSV)))

if not videos.empty and "sentiment" not in videos.columns:
    videos["sentiment"] = "not_analyzed"

if not comments.empty and "sentiment" not in comments.columns:
    comments["sentiment"] = "not_analyzed"

tab_overview, tab_videos, tab_comments = st.tabs(
    ["📊 Ringkasan", "🎬 Video", "💬 Komentar Penonton"]
)

with tab_overview:
    st.subheader("Ringkasan platform")

    v1, v2, v3, v4 = st.columns(4)

    v1.metric("Video terkumpul", len(videos))
    v2.metric("Komentar terkumpul", len(comments))

    if not videos.empty:
        v2v = videos["sentiment"].isin(
            ["positive", "neutral", "negative"]
        ).sum()
    else:
        v2v = 0

    if not comments.empty:
        c2v = comments["sentiment"].isin(
            ["positive", "neutral", "negative"]
        ).sum()
    else:
        c2v = 0

    v3.metric("Video dianalisis", int(v2v))
    v4.metric("Komentar dianalisis", int(c2v))

    left, right = st.columns(2)

    with left:
        st.markdown("### Sentimen metadata video")
        if not videos.empty:
            counts = (
                videos["sentiment"]
                .value_counts()
                .reindex(
                    ["positive", "neutral", "negative"],
                    fill_value=0,
                )
            )
            st.bar_chart(counts)
        else:
            st.info("Data analisis video belum tersedia.")

    with right:
        st.markdown("### Sentimen komentar")
        if not comments.empty:
            counts = (
                comments["sentiment"]
                .value_counts()
                .reindex(
                    ["positive", "neutral", "negative"],
                    fill_value=0,
                )
            )
            st.bar_chart(counts)
        else:
            st.info("Data analisis komentar belum tersedia.")

    st.info(
        "Prediksi metadata video dan prediksi komentar adalah dua hal berbeda. "
        "Distribusi yang ditampilkan hanya mewakili data yang berhasil "
        "dikumpulkan, bukan seluruh YouTube."
    )

with tab_videos:
    st.subheader("Data video dan prediksi sentimen")

    if videos.empty:
        st.warning("File youtube_videos_sentiment.csv belum tersedia.")
    else:
        query = st.text_input(
            "Cari video",
            key="video_search",
            placeholder="Kata kunci judul atau channel",
        )

        sentiment_values = ["Semua"] + sorted(
            videos["sentiment"].dropna().astype(str).unique().tolist()
        )
        selected = st.selectbox(
            "Filter sentimen video",
            sentiment_values,
            key="video_sentiment_filter",
        )

        view = videos.copy()

        if query:
            title = view.get(
                "title", pd.Series("", index=view.index)
            ).astype(str)
            channel = view.get(
                "channel_title", pd.Series("", index=view.index)
            ).astype(str)

            mask = (
                title.str.contains(query, case=False, regex=False)
                | channel.str.contains(query, case=False, regex=False)
            )
            view = view[mask]

        if selected != "Semua":
            view = view[view["sentiment"] == selected]

        cols = [
            column for column in [
                "title", "channel_title", "published_at",
                "sentiment", "confidence", "url"
            ] if column in view.columns
        ]

        display = view[cols].copy()

        if "confidence" in display.columns:
            display["confidence"] = pd.to_numeric(
                display["confidence"], errors="coerce"
            ).round(3)

        st.write(f"Menampilkan **{len(display)}** video.")
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Unduh CSV video",
            data=view.to_csv(index=False).encode("utf-8-sig"),
            file_name="youtube_videos_sentiment.csv",
            mime="text/csv",
            key="download_videos",
        )

with tab_comments:
    st.subheader("Analisis sentimen komentar publik")

    if comments.empty:
        st.warning(
            "File youtube_comments_sentiment.csv belum tersedia. "
            "Jalankan modul analisis komentar terlebih dahulu."
        )
    else:
        allowed = {"positive", "neutral", "negative"}
        valid = comments[
            comments["sentiment"].astype(str).str.lower().isin(allowed)
        ].copy()

        f1, f2 = st.columns(2)

        video_titles = ["Semua video"] + sorted(
            x for x in valid.get(
                "video_title", pd.Series(dtype=str)
            ).astype(str).unique() if x.strip()
        )

        chosen_video = f1.selectbox(
            "Pilih video",
            video_titles,
            key="comment_video_filter",
        )

        comment_labels = ["Semua"] + sorted(
            valid["sentiment"].astype(str).unique().tolist()
        )

        chosen_label = f2.selectbox(
            "Pilih sentimen",
            comment_labels,
            key="comment_sentiment_filter",
        )

        view = valid.copy()

        if chosen_video != "Semua video":
            view = view[view["video_title"] == chosen_video]

        if chosen_label != "Semua":
            view = view[view["sentiment"] == chosen_label]

        counts = (
            view["sentiment"]
            .value_counts()
            .reindex(
                ["positive", "neutral", "negative"],
                fill_value=0,
            )
        )

        st.bar_chart(counts)
        st.write(f"Menampilkan **{len(view)}** komentar.")

        display_cols = [
            column for column in [
                "video_title", "text", "sentiment",
                "confidence", "like_count", "published_at"
            ] if column in view.columns
        ]

        display = view[display_cols].copy()

        if "confidence" in display.columns:
            display["confidence"] = pd.to_numeric(
                display["confidence"], errors="coerce"
            ).round(3)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Unduh CSV komentar",
            data=view.to_csv(index=False).encode("utf-8-sig"),
            file_name="youtube_comments_sentiment_filtered.csv",
            mime="text/csv",
            key="download_comments",
        )

        st.caption(
            "Sentimen merupakan prediksi model pada teks komentar. "
            "Hasil tidak membuktikan opini seluruh penonton dan belum "
            "memiliki evaluasi akurasi independen."
        )

st.divider()
st.caption(
    "DS-03 Prototype · YouTube API · Transformer inference · "
    "Local dataset · Model evaluation masih diperlukan"
)
