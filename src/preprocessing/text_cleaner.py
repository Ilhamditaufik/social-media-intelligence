"""Text preprocessing utilities for multilingual social media data."""

import html
import re
import unicodedata

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

# Keep language detection reproducible where possible.
DetectorFactory.seed = 0

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<![\w.])@[A-Za-z0-9_]{1,50}")
HTML_TAG_PATTERN = re.compile(r"<[^>]*>")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#([\w]+)", re.UNICODE)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str | None, lowercase: bool = False) -> str:
    """Clean social media text while preserving multilingual characters and emoji.

    URLs become URL, account mentions become USER, and hashtag words are kept.
    """
    if not isinstance(text, str) or not text:
        return ""

    text = unicodedata.normalize("NFC", html.unescape(text))
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" URL ", text)
    text = MENTION_PATTERN.sub(" USER ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = CONTROL_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def detect_language(text: str | None) -> str:
    """Return a language code, or 'unknown' when text is too short or unclear."""
    cleaned = clean_text(text)
    words = re.findall(r"[^\W\d_]+", cleaned, flags=re.UNICODE)
    letter_count = sum(len(word) for word in words)

    # Short posts are too unreliable for language detection.
    if len(words) < 3 or letter_count < 8:
        return "unknown"

    try:
        return detect(cleaned)
    except LangDetectException:
        return "unknown"
