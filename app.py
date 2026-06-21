"""
app.py

Streamlit interface for the Language Fingerprinting System.

Run using:
streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.predict import predict_author


# ---------------------------------------------------------
# Page settings
# ---------------------------------------------------------

st.set_page_config(
    page_title="Language Fingerprinting System",
    page_icon="🧬",
    layout="centered",
)


# ---------------------------------------------------------
# Page heading
# ---------------------------------------------------------

st.title("🧬 Language Fingerprinting System")

st.write(
    "This system predicts the most likely author of an anonymous text "
    "by analyzing language patterns, TF-IDF features, and stylometric features."
)

st.warning(
    "Important: This result is a machine-learning estimate, not proof of identity. "
    "It should only be used for educational, authorized, and forensic-support purposes."
)


# ---------------------------------------------------------
# User input
# ---------------------------------------------------------

anonymous_text = st.text_area(
    label="Enter Anonymous Text",
    placeholder=(
        "Paste a paragraph here. Longer text samples usually produce "
        "more meaningful predictions."
    ),
    height=220,
)


# ---------------------------------------------------------
# Prediction button
# ---------------------------------------------------------

if st.button("Predict Author", type="primary"):

    if not anonymous_text.strip():
        st.error("Please enter some text before predicting.")

    else:
        try:
            result = predict_author(anonymous_text)

            predicted_author = result["predicted_author"]
            confidence = result["confidence"]

            st.success(
                f"Most Likely Author: {predicted_author}"
            )

            st.metric(
                label="Estimated Confidence",
                value=f"{confidence:.2%}",
            )

            st.subheader("Author Probability Rankings")

            probability_data = pd.DataFrame(
                result["ranked_predictions"]
            )

            probability_data["confidence_percent"] = (
                probability_data["confidence"] * 100
            )

            probability_data = probability_data.rename(
                columns={
                    "author": "Author",
                    "confidence_percent": "Confidence (%)",
                }
            )

            display_table = probability_data[
                ["Author", "Confidence (%)"]
            ].copy()

            display_table["Confidence (%)"] = display_table[
                "Confidence (%)"
            ].round(2)

            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True,
            )

            chart_data = probability_data.set_index("Author")[
                "Confidence (%)"
            ]

            st.bar_chart(chart_data)

            st.subheader("Model Information")

            model_info = result["model_information"]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Training Samples",
                model_info["training_samples"],
            )

            col2.metric(
                "Testing Samples",
                model_info["test_samples"],
            )

            col3.metric(
                "Controlled Test Accuracy",
                f"{model_info['test_accuracy']:.2%}",
            )

            st.info(
                "The confidence score shows how strongly the trained model "
                "prefers one author label over the other available labels. "
                "It is not proof that a person wrote the text."
            )

        except ValueError as error:
            st.error(str(error))

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                f"An unexpected error occurred: {error}"
            )


# ---------------------------------------------------------
# Sidebar explanation
# ---------------------------------------------------------

with st.sidebar:
    st.header("How It Works")

    st.write(
        """
        **1. TF-IDF Features**  
        Identifies meaningful words and phrases.

        **2. Character TF-IDF**  
        Identifies small patterns such as spelling, contractions,
        punctuation, and word endings.

        **3. Stylometric Features**  
        Measures sentence length, word length, vocabulary richness,
        punctuation frequency, capitalization, and digit usage.

        **4. SVM Classifier**  
        Predicts the most likely author class.
        """
    )

    st.divider()

    st.caption(
        "University Project: Language Fingerprinting System "
        "for Author Identification Using Machine Learning"
    )