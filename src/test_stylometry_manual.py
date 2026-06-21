from src.stylometry import extract_stylometric_features


sample_text = """
Hello! THIS is a short sentence, but it has punctuation.
Is it working correctly? I hope so!
"""

features = extract_stylometric_features(sample_text)

print("Stylometric Features:\n")

for feature_name, value in features.items():
    print(f"{feature_name}: {value:.4f}")