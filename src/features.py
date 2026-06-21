"""
features.py

This file combines three types of features:

1. Word TF-IDF:
   Learns important words and word pairs.

2. Character TF-IDF:
   Learns smaller writing patterns such as punctuation habits,
   spelling style, contractions, and character sequences.

3. Stylometric features:
   Learns writing-style measurements such as sentence length,
   word length, punctuation frequency, and capitalization.
"""

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import preprocess_for_tfidf
from src.stylometry import extract_stylometric_features


# These names must match the feature names returned by stylometry.py
STYLOMETRIC_FEATURE_NAMES = [
    "average_sentence_length",
    "average_word_length",
    "vocabulary_richness",
    "punctuation_frequency",
    "comma_frequency",
    "period_frequency",
    "exclamation_frequency",
    "question_frequency",
    "semicolon_frequency",
    "colon_frequency",
    "uppercase_ratio",
    "digit_ratio",
]


class StylometricTransformer(BaseEstimator, TransformerMixin):
    """
    Converts raw text into numerical stylometric feature vectors.

    Input:
        A list, NumPy array, or Pandas Series containing text samples.

    Output:
        A sparse matrix where each row represents one text sample.
    """

    def fit(self, X, y=None):
        """
        No training is needed for manually calculated stylometric features.
        This method is required so it works inside Scikit-learn pipelines.
        """
        return self

    def transform(self, X):
        """
        Extract stylometric features for every text sample.
        """

        feature_rows = []

        for text in X:
            feature_dictionary = extract_stylometric_features(text)

            row = [
                feature_dictionary[feature_name]
                for feature_name in STYLOMETRIC_FEATURE_NAMES
            ]

            feature_rows.append(row)

        # Convert the normal numeric array into sparse matrix format.
        # Sparse matrices work efficiently with TF-IDF matrices.
        return csr_matrix(np.array(feature_rows, dtype=float))

    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of stylometric features.
        """
        return np.array(STYLOMETRIC_FEATURE_NAMES)


def build_feature_union():
    """
    Creates and returns one combined feature extractor.

    It combines:
    - Word TF-IDF
    - Character TF-IDF
    - Stylometric numeric features

    This returned object will later be connected to the SVM model.
    """

    # ---------------------------------------------------------
    # 1. Word-level TF-IDF
    # ---------------------------------------------------------
    # Captures individual words and 2-word phrases.
    # Example:
    # "final report"
    # "hidden purpose"
    # "public version"
    word_tfidf = TfidfVectorizer(
        preprocessor=preprocess_for_tfidf,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
        max_features=8000,
        sublinear_tf=True,
    )

    # ---------------------------------------------------------
    # 2. Character-level TF-IDF
    # ---------------------------------------------------------
    # Captures character sequences.
    # Example:
    # "ing", "tion", "!!", "; ", "thou"
    #
    # This helps recognize punctuation and small writing habits.
    character_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=10000,
        sublinear_tf=True,
    )

    # ---------------------------------------------------------
    # 3. Stylometric feature pipeline
    # ---------------------------------------------------------
    # StandardScaler makes numeric style features comparable in scale.
    # with_mean=False is needed because we are working with sparse matrices.
    stylometry_pipeline = Pipeline(
        steps=[
            ("stylometry", StylometricTransformer()),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    # `Any` is used here only to prevent a false Pylance warning.
    # TfidfVectorizer works correctly inside FeatureUnion at runtime.
    transformers: Any = [
        ("word_tfidf", word_tfidf),
        ("character_tfidf", character_tfidf),
        ("stylometry", stylometry_pipeline),
    ]

    # Combine all three feature groups into one large feature matrix.
    all_features = FeatureUnion(
        transformer_list=transformers
    )

    return all_features