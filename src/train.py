"""
train.py

This file:
1. Loads dataset.csv
2. Splits data into training and testing sets
3. Trains the SVM author-identification model
4. Saves the trained model using pickle
5. Saves the test split for evaluation later
"""

from pathlib import Path
from collections import Counter
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.preprocessing import download_nltk_resources
from src.modeling import build_author_identification_model


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"

MODELS_FOLDER = PROJECT_ROOT / "models"
OUTPUTS_FOLDER = PROJECT_ROOT / "outputs"

MODEL_PATH = MODELS_FOLDER / "author_model.pkl"
TEST_DATA_PATH = MODELS_FOLDER / "test_data.pkl"


def load_dataset():
    """
    Loads the CSV dataset and validates it.

    Required columns:
    - author
    - text

    Returns:
        X: Text samples
        y: Author labels
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATASET_PATH}\n\n"
            "Make sure dataset.csv is inside the data folder."
        )

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {"author", "text"}

    if not required_columns.issubset(dataset.columns):
        raise ValueError(
            "Your dataset.csv must contain exactly these columns:\n"
            "author,text"
        )

    # Keep only useful columns.
    dataset = dataset[["author", "text"]]

    # Remove missing rows.
    dataset = dataset.dropna()

    # Convert all values into clean strings.
    dataset["author"] = dataset["author"].astype(str).str.strip()
    dataset["text"] = dataset["text"].astype(str).str.strip()

    # Remove blank author/text rows.
    dataset = dataset[
        (dataset["author"] != "") &
        (dataset["text"] != "")
    ]

    author_counts = Counter(dataset["author"])

    if len(author_counts) < 2:
        raise ValueError(
            "At least two different authors are needed to train a classifier."
        )

    print("\nDataset Summary")
    print("-" * 50)
    print(f"Total samples: {len(dataset)}")
    print(f"Number of authors: {len(author_counts)}")

    print("\nSamples per author:")

    for author, count in sorted(author_counts.items()):
        print(f"- {author}: {count}")

    return dataset["text"], dataset["author"]


def main():
    """
    Runs the complete training process.
    """

    print("Preparing NLTK resources...")
    download_nltk_resources()

    # Create model/output folders if they do not exist.
    MODELS_FOLDER.mkdir(exist_ok=True)
    OUTPUTS_FOLDER.mkdir(exist_ok=True)

    # Load the labeled dataset.
    X, y = load_dataset()

    # Split data:
    # 75% training data
    # 25% testing data
    #
    # stratify=y ensures every author appears proportionally
    # in both the training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("\nTraining/Test Split")
    print("-" * 50)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nBuilding SVM model...")
    model = build_author_identification_model()

    print("Training model. Please wait...")
    model.fit(X_train, y_train)

    # Make predictions on unseen test data.
    predictions = model.predict(X_test)

    # Calculate basic accuracy.
    accuracy = accuracy_score(y_test, predictions)

    print("\nTraining Complete")
    print("-" * 50)
    print(f"Test Accuracy: {accuracy:.2%}")

    # Save trained model plus helpful metadata.
    model_bundle = {
        "model": model,
        "authors": sorted(y.unique().tolist()),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": float(accuracy),
    }

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model_bundle, model_file)

    # Save test data for the next evaluation step.
    test_data_bundle = {
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
    }

    with open(TEST_DATA_PATH, "wb") as test_file:
        pickle.dump(test_data_bundle, test_file)

    print("\nFiles Saved Successfully")
    print("-" * 50)
    print(f"Trained model: {MODEL_PATH}")
    print(f"Test data: {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()