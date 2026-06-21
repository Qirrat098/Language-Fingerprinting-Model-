

import re
import string

from src.preprocessing import get_sentences


def get_words(text):
    """
    Extracts alphabetic words from text.

    Example:
    'Hello, world! It is 2026.'
    -> ['Hello', 'world', 'It', 'is']
    """

    if not isinstance(text, str):
        text = str(text)

    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def extract_stylometric_features(text):
    """
    Extracts numerical writing-style features from a single text sample.

    Returns:
        dict: A dictionary containing stylometric measurements.
    """

    if not isinstance(text, str):
        text = str(text)

    words = get_words(text)
    sentences = get_sentences(text)

    total_characters = max(len(text), 1)
    total_words = max(len(words), 1)
    total_sentences = max(len(sentences), 1)

    unique_words = set(word.lower() for word in words)

    punctuation_count = sum(
        1 for character in text
        if character in string.punctuation
    )

    uppercase_count = sum(
        1 for character in text
        if character.isupper()
    )

    digit_count = sum(
        1 for character in text
        if character.isdigit()
    )

    average_word_length = (
        sum(len(word) for word in words) / total_words
        if words else 0
    )

    features = {
        "average_sentence_length": total_words / total_sentences,
        "average_word_length": average_word_length,
        "vocabulary_richness": len(unique_words) / total_words,
        "punctuation_frequency": punctuation_count / total_characters,
        "comma_frequency": text.count(",") / total_characters,
        "period_frequency": text.count(".") / total_characters,
        "exclamation_frequency": text.count("!") / total_characters,
        "question_frequency": text.count("?") / total_characters,
        "semicolon_frequency": text.count(";") / total_characters,
        "colon_frequency": text.count(":") / total_characters,
        "uppercase_ratio": uppercase_count / total_characters,
        "digit_ratio": digit_count / total_characters,
    }

    return features