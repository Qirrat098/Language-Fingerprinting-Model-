"""
predict.py

This file loads the trained Language Fingerprinting model
and predicts the most likely author of an anonymous text.

Output:
- Predicted author
- Confidence score
- Ranking of all possible authors
"""

from pathlib import Path
import pickle


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_FOLDER = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_FOLDER / "author_model.pkl"


def load_saved_model():
    """
    Loads the saved author-identification model.

    Returns:
        dict: Model bundle containing the model and metadata.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model file was not found.\n\n"
            "Please train the model first using:\n"
            "python -m src.train"
        )

    with open(MODEL_PATH, "rb") as model_file:
        model_bundle = pickle.load(model_file)

    return model_bundle


def predict_author(text):
    """
    Predicts the most likely author for an anonymous text.

    Parameters:
        text (str): Anonymous text entered by the user.

    Returns:
        dict: Prediction results, confidence, and ranked author probabilities.
    """

    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    text = text.strip()

    if len(text) < 20:
        raise ValueError(
            "Please enter a longer text sample. "
            "At least 20 characters are required."
        )

    model_bundle = load_saved_model()

    model = model_bundle["model"]

    # Predict the most likely author.
    predicted_author = model.predict([text])[0]

    # Get probability/confidence for every author.
    probabilities = model.predict_proba([text])[0]

    # Get class names in the same order as probabilities.
    authors = model.classes_

    ranked_predictions = []

    for author, probability in zip(authors, probabilities):
        ranked_predictions.append(
            {
                "author": str(author),
                "confidence": float(probability),
            }
        )

    # Sort from highest confidence to lowest confidence.
    ranked_predictions = sorted(
        ranked_predictions,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    return {
        "predicted_author": str(predicted_author),
        "confidence": float(ranked_predictions[0]["confidence"]),
        "ranked_predictions": ranked_predictions,
        "model_information": {
            "training_samples": model_bundle["training_samples"],
            "test_samples": model_bundle["test_samples"],
            "test_accuracy": model_bundle["test_accuracy"],
        },
    }


def main():
    """
    Runs an interactive command-line prediction test.
    """

    print("=" * 60)
    print("LANGUAGE FINGERPRINTING SYSTEM")
    print("=" * 60)

    print("\nPaste an anonymous text below.")
    print("Press Enter twice when you finish.\n")

    lines = []

    while True:
        line = input()

        if line.strip() == "":
            break

        lines.append(line)

    anonymous_text = " ".join(lines)

    try:
        result = predict_author(anonymous_text)

        print("\nPrediction Result")
        print("-" * 60)

        print(f"Most Likely Author: {result['predicted_author']}")
        print(f"Estimated Confidence: {result['confidence']:.2%}")

        print("\nTop Author Rankings")
        print("-" * 60)

        for index, item in enumerate(result["ranked_predictions"], start=1):
            print(
                f"{index}. "
                f"{item['author']} "
                f"({item['confidence']:.2%})"
            )

        print("\nModel Information")
        print("-" * 60)

        print(
            f"Training Samples: "
            f"{result['model_information']['training_samples']}"
        )

        print(
            f"Test Accuracy: "
            f"{result['model_information']['test_accuracy']:.2%}"
        )

        print(
            "\nNote: Confidence is an ML estimate, not proof of identity."
        )

    except ValueError as error:
        print(f"\nInput Error: {error}")

    except FileNotFoundError as error:
        print(f"\nModel Error: {error}")

    except Exception as error:
        print(f"\nUnexpected Error: {error}")


if __name__ == "__main__":
    main()