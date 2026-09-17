"""
preprocessing.py
-----------------
Text cleaning and preprocessing utilities for the NLP Sentiment
Analysis project (ICT 305 - Assessment 2).

Pipeline steps implemented:
    1. Lower-casing
    2. Removal of URLs, @mentions, hashtags symbols, HTML entities
    3. Removal of punctuation / non-alphabetic characters
    4. Tokenisation (NLTK)
    5. Stop-word removal (NLTK English stop-word list)
    6. Lemmatisation (WordNet Lemmatiser)
"""

import re
import nltk

# Ensure required NLTK corpora are available (downloads only if missing).
for resource in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(
            f"corpora/{resource}" if resource in
            ("stopwords", "wordnet", "omw-1.4") else f"tokenizers/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
# Keep negation words - they flip sentiment and must NOT be removed.
NEGATIONS = {"no", "not", "nor", "never", "none", "n't", "cannot"}
STOPWORDS = STOPWORDS - NEGATIONS

LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_SYMBOL_PATTERN = re.compile(r"#")
HTML_ENTITY_PATTERN = re.compile(r"&\w+;")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply regex-based cleaning steps to a raw text string."""
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HTML_ENTITY_PATTERN.sub(" ", text)
    text = HASHTAG_SYMBOL_PATTERN.sub("", text)          # keep the word, drop '#'
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list:
    """Tokenise, remove stop-words, and lemmatise a cleaned text string."""
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenise -> stop-word removal -> lemmatise.
    Returns a single space-joined string ready for vectorisation."""
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "I LOVE this product!! :) Check it out http://example.com @friend #amazing"
    print("Raw     :", sample)
    print("Cleaned :", clean_text(sample))
    print("Final   :", preprocess(sample))
