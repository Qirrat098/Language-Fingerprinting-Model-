from preprocessing import download_nltk_resources, preprocess_for_tfidf


download_nltk_resources()

sample_text = "The report is ready, and it looks great! I checked it twice."

result = preprocess_for_tfidf(sample_text)

print("Original text:")
print(sample_text)

print("\nProcessed text:")
print(result)