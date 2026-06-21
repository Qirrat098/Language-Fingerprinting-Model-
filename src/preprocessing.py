"""
preprocessing.py

This file cleans text before it is converted into TF-IDF features.

Important:
We remove punctuation and stopwords only for the TF-IDF text branch.
Later, stylometric features will still use the original raw text because
punctuation style is part of an author's language fingerprint.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


def download_nltk_resources():
    """
    Downloads required NLTK resources.
    Run automatically before text preprocessing.
    """

    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            print(f"Could not download NLTK resource: {resource}")


def clean_text(text):
    """
    Basic cleaning:
    - Converts text to lowercase
    - Removes extra spaces
    - Removes numbers
    - Keeps letters and punctuation for now

    Parameters:
        text (str): Input text

    Returns:
        str: Cleaned text
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    """
    Splits text into word tokens using NLTK.

    Example:
        "Hello, how are you?"
        -> ["Hello", ",", "how", "are", "you", "?"]
    """

    return word_tokenize(text)


def remove_stopwords_and_punctuation(tokens):
    """
    Removes:
    - English stopwords such as 'the', 'is', 'and'
    - punctuation symbols
    - non-alphabetic tokens

    Parameters:
        tokens (list): List of word tokens

    Returns:
        list: Important cleaned tokens only
    """

    stop_words = set(stopwords.words("english"))

    cleaned_tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return cleaned_tokens


def preprocess_for_tfidf(text):
    """
    Full preprocessing pipeline for TF-IDF.

    Steps:
    1. Clean the text
    2. Tokenize it
    3. Remove stopwords and punctuation
    4. Join remaining words back into one string

    Example:
        Input:
        "The report is ready, and it looks great!"

        Output:
        "report ready looks great"
    """

    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords_and_punctuation(tokens)

    return " ".join(filtered_tokens)


def get_sentences(text):
    """
    Splits text into sentences.

    This will be useful later for features such as:
    - average sentence length
    - sentence count
    """

    if not isinstance(text, str):
        text = str(text)

    return sent_tokenize(text)