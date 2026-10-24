"""Multilingual sentiment inference with a Hugging Face transformer."""

from functools import lru_cache

from transformers import pipeline

from preprocessing.text_cleaner import clean_text

MODEL_ID = "nahiar/sentiment-analysis-v2"

LABEL_MAP = {
    "LABEL_0": "neutral",
    "LABEL_1": "positive",
    "LABEL_2": "negative",
    "NEUTRAL": "neutral",
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "POSITIF": "positive",
    "NEGATIF": "negative",
    "NETRAL": "neutral",
}


def normalize_label(label: str) -> str:
    """Normalize the model's output label."""
    return LABEL_MAP.get(str(label).strip().upper(), "unknown")


@lru_cache(maxsize=1)
def get_classifier():
    """Load the model once and reuse it for subsequent predictions."""
    return pipeline(
        task="text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device=-1,
    )


def predict_sentiment(text: str) -> dict:
    """Predict sentiment and return its confidence score."""
    cleaned = clean_text(text)

    if not cleaned:
        raise ValueError("Teks tidak boleh kosong.")

    classifier = get_classifier()
    output = classifier(
        cleaned,
        truncation=True,
        max_length=256,
    )

    if isinstance(output, list) and output:
        output = output[0]

    label = normalize_label(output["label"])

    return {
        "text": cleaned,
        "label": label,
        "confidence": float(output["score"]),
        "model": MODEL_ID,
    }
