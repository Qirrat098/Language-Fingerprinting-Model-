from src.features import build_feature_union


sample_texts = [
    "The report was reviewed carefully, and the final decision remains pending.",
    "Well, the whole town got worried for nearly three minutes, which was impressive.",
    "Perhaps the room had changed, or perhaps it was only the memory of it.",
]


feature_extractor = build_feature_union()

feature_matrix = feature_extractor.fit_transform(sample_texts)

print("Feature matrix shape:")
print(feature_matrix.shape)

print("\nNumber of text samples:")
print(feature_matrix.shape[0])

print("\nNumber of combined features:")
print(feature_matrix.shape[1])