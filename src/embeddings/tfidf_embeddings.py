import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    def __init__(self, max_features=100):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True
        )

    def fit(self, sentences):
        self.vectorizer.fit(sentences)
        return self

    def transform(self, sentences):
        return self.vectorizer.transform(sentences).toarray()

    def fit_transform(self, sentences):
        return self.vectorizer.fit_transform(sentences).toarray()


# # Data Reading

def load_sentences(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = [item["sentence"] for item in data]
    ids = [item["id"] for item in data]

    return ids, sentences



# main function

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    json_path = os.path.join(BASE_DIR, "data", "type_c_dataset.json")

    ids, sentences = load_sentences(json_path)

    embedder = TfidfEmbedder(max_features=50)
    embeddings = embedder.fit_transform(sentences)

    print("TF-IDF shape:", embeddings.shape)
    print("Example:", embeddings[0][:10])

    # save
    output_path = os.path.join(BASE_DIR, "results", "metrics", "tfidf.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

    print("Saved to:", output_path)