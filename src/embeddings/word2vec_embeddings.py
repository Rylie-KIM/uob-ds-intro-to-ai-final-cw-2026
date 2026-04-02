import json
import os
import numpy as np
from gensim.models import Word2Vec


class Word2VecEmbedder:
    def __init__(self, vector_size=50):
        self.model = None
        self.vector_size = vector_size

    def tokenize(self, sentence):
        return sentence.lower().split()

    def fit(self, sentences):
        tokenized = [self.tokenize(s) for s in sentences]

        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=5,
            min_count=1,
            workers=4
        )
        return self

    def sentence_vector(self, sentence):
        tokens = self.tokenize(sentence)

        vectors = []
        for word in tokens:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])

        if len(vectors) == 0:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)

    def transform(self, sentences):
        return np.array([self.sentence_vector(s) for s in sentences])

    def fit_transform(self, sentences):
        self.fit(sentences)
        return self.transform(sentences)



# data reading

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

    embedder = Word2VecEmbedder(vector_size=50)
    embeddings = embedder.fit_transform(sentences)

    print("Word2Vec shape:", embeddings.shape)
    print("Example:", embeddings[0][:10])

    # 保存
    output_path = os.path.join(BASE_DIR, "results", "metrics", "word2vec.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

    print("Saved to:", output_path)