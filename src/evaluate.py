"""
evaluate.py

This file evaluates the trained author-identification model.

It generates:
- Accuracy score
- Classification report
- Confusion matrix image
- JSON evaluation summary
"""

from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_FOLDER = PROJECT_ROOT / "models"
OUTPUTS_FOLDER = PROJECT_ROOT / "outputs"

MODEL_PATH = MODELS_FOLDER / "author_model.pkl"
TEST_DATA_PATH = MODELS_FOLDER / "test_data.pkl"

CONFUSION_MATRIX_PATH = OUTPUTS_FOLDER / "confusion_matrix.png"
REPORT_PATH = OUTPUTS_FOLDER / "classification_report.txt"
RESULTS_PATH = OUTPUTS_FOLDER / "evaluation_results.json"


def main():
    """
    Loads the trained model and saved test data,
    then evaluates model performance.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run this first:\n"
            "python -m src.train"
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Test data file not found. Run this first:\n"
            "python -m src.train"
        )

    OUTPUTS_FOLDER.mkdir(exist_ok=True)

    print("Loading trained model...")

    with open(MODEL_PATH, "rb") as model_file:
        model_bundle = pickle.load(model_file)

    model = model_bundle["model"]

    print("Loading test data...")

    with open(TEST_DATA_PATH, "rb") as test_file:
        test_data = pickle.load(test_file)

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    print("Generating predictions...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    authors = sorted(model_bundle["authors"])

    report_text = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    report_dictionary = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    print("\nEvaluation Results")
    print("-" * 55)
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report")
    print("-" * 55)
    print(report_text)

    # Save readable classification report.
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    # Create confusion matrix.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=authors,
    )

    figure, axis = plt.subplots(figsize=(12, 9))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=authors,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=False,
    )

    axis.set_title("Language Fingerprinting: Confusion Matrix")
    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Save results as JSON for future Streamlit display.
    results = {
        "accuracy": float(accuracy),
        "training_samples": model_bundle["training_samples"],
        "test_samples": model_bundle["test_samples"],
        "authors": authors,
        "classification_report": report_dictionary,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=4)

    print("\nFiles Generated Successfully")
    print("-" * 55)
    print(f"Classification report: {REPORT_PATH}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"Evaluation summary: {RESULTS_PATH}")


if __name__ == "__main__":
    main()