import json
import os
import re
from typing import List, Tuple

import numpy as np
from gensim import downloader as api
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = "data/type_c_dataset.json"
EMBEDDING_OUTPUT = "data/type_c_glove_embeddings.npy"
REPORT_OUTPUT = "data/type_c_glove_report.txt"


def load_dataset(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Please run your sentence generator first to create this file."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("Dataset is empty.")

    return data


def tokenize(text: str) -> List[str]:
    """
    Convert sentence to lowercase tokens.
    Keep only letters/numbers/apostrophes.
    """
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def sentence_to_glove_vector(sentence: str, glove_model, dim: int) -> np.ndarray:
    """
    Represent a sentence by averaging all word vectors.
    If no token is found in GloVe, return a zero vector.
    """
    tokens = tokenize(sentence)
    vectors = [glove_model[token] for token in tokens if token in glove_model]

    if not vectors:
        return np.zeros(dim, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def extract_labels_from_notation(notation: str) -> Tuple[int, int, int]:
    """
    From a tic-tac-toe notation string, derive simple labels for testing.

    Returns:
        x_count: number of X on the board
        o_count: number of O on the board
        center_class:
            0 -> center is empty
            1 -> center is X
            2 -> center is O
    """
    clean = notation.replace(" ", "")
    cells = list(clean)

    x_count = cells.count("X")
    o_count = cells.count("O")

    center = cells[4] if len(cells) >= 5 else "_"
    if center == "_":
        center_class = 0
    elif center == "X":
        center_class = 1
    else:
        center_class = 2

    return x_count, o_count, center_class


def build_feature_and_labels(data: List[dict], glove_model, dim: int):
    """
    Build:
      X = GloVe sentence embeddings
      y1 = x_count
      y2 = o_count
      y3 = center_class
    """
    X = []
    y_x_count = []
    y_o_count = []
    y_center = []

    for item in data:
        sentence = item["sentence"]
        notation = item["notation"]

        vec = sentence_to_glove_vector(sentence, glove_model, dim)
        x_count, o_count, center_class = extract_labels_from_notation(notation)

        X.append(vec)
        y_x_count.append(x_count)
        y_o_count.append(o_count)
        y_center.append(center_class)

    return (
        np.array(X, dtype=np.float32),
        np.array(y_x_count),
        np.array(y_o_count),
        np.array(y_center),
    )


def train_and_evaluate(X: np.ndarray, y: np.ndarray, label_name: str) -> str:
    """
    Train a simple Logistic Regression classifier on GloVe embeddings
    and return a text report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    clf = LogisticRegression(
        max_iter=2000,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    report = []
    report.append(f"===== Task: Predict {label_name} from GloVe sentence embeddings =====")
    report.append(f"Accuracy: {acc:.4f}")
    report.append("Classification Report:")
    report.append(classification_report(y_test, y_pred, digits=4))
    report.append("")

    return "\n".join(report)


def show_example_similarities(data: List[dict], X: np.ndarray, top_k: int = 3) -> str:
    """
    Show a few nearest-neighbor examples based on cosine similarity.
    This helps inspect whether semantically similar sentences are close
    in GloVe space.
    """
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    lines = []
    lines.append("===== Example nearest-neighbor sentence similarities =====")

    sample_indices = [0, min(10, len(data) - 1), min(50, len(data) - 1)]

    for idx in sample_indices:
        sims = []
        for j in range(len(data)):
            if j == idx:
                continue
            sim = cosine_similarity(X[idx], X[j])
            sims.append((j, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        top_matches = sims[:top_k]

        lines.append(f"\nQuery sentence [{idx}]: {data[idx]['sentence']}")
        lines.append(f"Notation: {data[idx]['notation']}")

        for rank, (j, sim) in enumerate(top_matches, start=1):
            lines.append(
                f"  Top {rank}: sim={sim:.4f} | sentence={data[j]['sentence']} | notation={data[j]['notation']}"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs("data", exist_ok=True)

    print("Loading dataset...")
    data = load_dataset(DATA_PATH)
    print(f"Loaded {len(data)} samples")

    print("Loading GloVe model (this may take a while the first time)...")
    glove_model = api.load("glove-wiki-gigaword-100")
    dim = glove_model.vector_size
    print(f"GloVe dimension: {dim}")

    print("Building sentence embeddings...")
    X, y_x_count, y_o_count, y_center = build_feature_and_labels(data, glove_model, dim)

    np.save(EMBEDDING_OUTPUT, X)
    print(f"Saved embeddings to: {EMBEDDING_OUTPUT}")
    print(f"Embedding shape: {X.shape}")

    print("Running evaluation...")
    report_parts = []

    report_parts.append(train_and_evaluate(X, y_x_count, "x_count"))
    report_parts.append(train_and_evaluate(X, y_o_count, "o_count"))
    report_parts.append(train_and_evaluate(X, y_center, "center_cell"))
    report_parts.append(show_example_similarities(data, X, top_k=3))

    full_report = "\n".join(report_parts)

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write(full_report)

    print("\n===== FINAL REPORT =====")
    print(full_report)
    print(f"\nSaved report to: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()