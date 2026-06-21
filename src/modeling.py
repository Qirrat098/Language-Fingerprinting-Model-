"""
modeling.py

This file creates the complete machine learning pipeline.

Pipeline:
Raw text
    ↓
TF-IDF + Character TF-IDF + Stylometric Features
    ↓
Linear SVM Classifier
    ↓
Predicted Author
"""

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.features import build_feature_union


def build_author_identification_model():
    """
    Creates the full author-identification machine learning pipeline.

    The model combines:
    - Word TF-IDF features
    - Character TF-IDF features
    - Stylometric features
    - Linear Support Vector Machine classifier

    Returns:
        Pipeline: Complete Scikit-learn machine learning pipeline
    """

    model_pipeline = Pipeline(
        steps=[
            (
                "features",
                build_feature_union(),
            ),
            (
                "classifier",
                SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return model_pipeline