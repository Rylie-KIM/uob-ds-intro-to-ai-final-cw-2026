import os
import re
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TYPE_A = os.path.join(BASE_DIR, "sentences_a.csv")
TYPE_B = os.path.join(BASE_DIR, "sentences_b.csv")
TYPE_C = os.path.join(BASE_DIR, "sentences_c.csv")

OUTPUT_CSV = os.path.join(BASE_DIR, "fake_glove_embeddings.csv")
OUTPUT_REPORT = os.path.join(BASE_DIR, "fake_glove_report.txt")

DIM = 100

word_vectors = {}

def get_word_vector(word):
    if word not in word_vectors:
        word_vectors[word] = np.random.normal(size=(DIM,))
    return word_vectors[word]


def normalize(text):
    text = str(text)
    text = text.replace("X", "cross")
    text = text.replace("O", "nought")
    return text.lower()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize(text))


def sentence_to_vec(sentence):
    tokens = tokenize(sentence)

    if len(tokens) == 0:
        return np.zeros(DIM)

    vecs = [get_word_vector(t) for t in tokens]
    return np.mean(vecs, axis=0)

def load_data():
    df_a = pd.read_csv(TYPE_A)
    df_b = pd.read_csv(TYPE_B)
    df_c = pd.read_csv(TYPE_C)

    df_a = pd.DataFrame({
        "type": "A",
        "sentence": df_a["label"]
    })

    df_b = pd.DataFrame({
        "type": "B",
        "sentence": df_b["sentence"]
    })

    df_c = pd.DataFrame({
        "type": "C",
        "sentence": df_c["sentence"]
    })

    return pd.concat([df_a, df_b, df_c], ignore_index=True)


def main():
    df = load_data()

    print(f"Total sentences: {len(df)}")

    embeddings = []

    for s in df["sentence"]:
        vec = sentence_to_vec(s)
        embeddings.append(vec)

    embeddings = np.vstack(embeddings)

    for i in range(DIM):
        df[f"dim_{i}"] = embeddings[:, i]

    df.to_csv(OUTPUT_CSV, index=False)

    sim = cosine_similarity(embeddings[:10], embeddings[:10])

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("Fake GloVe Test Report\n\n")
        f.write(f"Total samples: {len(df)}\n\n")
        f.write("Counts:\n")
        f.write(df["type"].value_counts().to_string())
        f.write("\n\nSimilarity matrix:\n")
        f.write(np.array2string(sim, precision=3))

    print("\nDone.")
    print("Saved:", OUTPUT_CSV)
    print("Saved:", OUTPUT_REPORT)


if __name__ == "__main__":
    main()