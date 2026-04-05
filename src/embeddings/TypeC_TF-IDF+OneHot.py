import csv
import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from type_c_core import parse_notation, POSITIONS



# Paths
SRC_ROOT = Path(__file__).resolve().parents[2]
TYPE_C_DATA_DIR = SRC_ROOT / "data" / "type-c"

INPUT_CSV = TYPE_C_DATA_DIR / "sentences_c.csv"
OUTPUT_FEATURES_NPZ = TYPE_C_DATA_DIR / "type_c_text_features.npz"
OUTPUT_META_PKL = TYPE_C_DATA_DIR / "type_c_text_features_meta.pkl"



# Position normalisation
POSITION_NORMALISATION = {
    # center
    "center": "center",
    "middle": "center",

    # top row
    "top left": "top_left",
    "upper left": "top_left",

    "top center": "top_center",
    "top middle": "top_center",

    "top right": "top_right",
    "upper right": "top_right",

    # middle row
    "middle left": "middle_left",
    "center left": "middle_left",

    "middle right": "middle_right",
    "center right": "middle_right",

    # bottom row
    "bottom left": "bottom_left",
    "lower left": "bottom_left",

    "bottom center": "bottom_center",
    "bottom middle": "bottom_center",

    "bottom right": "bottom_right",
    "lower right": "bottom_right",
}

CANONICAL_POSITIONS = {
    "TL": "top_left",
    "TM": "top_center",
    "TR": "top_right",
    "ML": "middle_left",
    "C": "center",
    "MR": "middle_right",
    "BL": "bottom_left",
    "BM": "bottom_center",
    "BR": "bottom_right",
}



# Utility: load csv
def load_sentence_dataset(csv_path: str | Path) -> List[Dict[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    required_cols = {"sentence_id", "sentence", "notation", "n_moves", "winner"}
    missing = required_cols - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    return rows



# Step 1: sentence normalisation
def canonicalise_positions(text: str) -> str:
    items = sorted(POSITION_NORMALISATION.items(), key=lambda x: len(x[0]), reverse=True)
    for old, new in items:
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def canonicalise_patterns(text: str) -> str:
    """
    Optional light normalisation for sentence templates.
    We keep this light to preserve useful structure while reducing variation.
    """
    replacements = {
        "there is x in": "x is in",
        "there is o in": "o is in",
        "has x": "contains_x",
        "has o": "contains_o",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalise_sentence(sentence: str) -> str:
    s = sentence.strip().lower()
    s = canonicalise_positions(s)
    s = canonicalise_patterns(s)

    # keep underscore tokens like top_left
    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s



# Step 2: TF-IDF
def build_tfidf_features(
    sentences: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,  # already lowercased manually
        token_pattern=r"(?u)\b[a-zA-Z0-9_]+\b",
    )
    X_tfidf = vectorizer.fit_transform(sentences)
    return X_tfidf.tocsr(), vectorizer



# Step 3: Board one-hot from notation
# 9 cells × 3 states = 27 dims     per-cell order = [X, O, EMPTY]

def notation_to_board_onehot(notation: str) -> np.ndarray:
    board = parse_notation(notation)
    vec = np.zeros(27, dtype=np.float32)

    for i, pos in enumerate(POSITIONS):
        offset = i * 3
        if pos in board.x:
            vec[offset + 0] = 1.0
        elif pos in board.o:
            vec[offset + 1] = 1.0
        else:
            vec[offset + 2] = 1.0

    return vec


def build_board_onehot_features(notations: List[str]) -> np.ndarray:
    return np.stack([notation_to_board_onehot(n) for n in notations], axis=0)


# Step 4: Optional extra structured features
def winner_onehot(winner: str) -> np.ndarray:
    """
    winner: "", "X", or "O"
    order = [X_win, O_win, no_winner]
    """
    vec = np.zeros(3, dtype=np.float32)
    winner = (winner or "").strip().upper()

    if winner == "X":
        vec[0] = 1.0
    elif winner == "O":
        vec[1] = 1.0
    else:
        vec[2] = 1.0
    return vec


def move_count_onehot(n_moves: int) -> np.ndarray:
    """
    0..9 moves
    """
    if not (0 <= n_moves <= 9):
        raise ValueError(f"n_moves must be in [0, 9], got {n_moves}")
    vec = np.zeros(10, dtype=np.float32)
    vec[n_moves] = 1.0
    return vec


def center_features(notation: str) -> np.ndarray:
    """
    order = [center_is_X, center_is_O, center_is_EMPTY]
    """
    board = parse_notation(notation)
    vec = np.zeros(3, dtype=np.float32)
    if "C" in board.x:
        vec[0] = 1.0
    elif "C" in board.o:
        vec[1] = 1.0
    else:
        vec[2] = 1.0
    return vec


def build_extra_structured_features(
    notations: List[str],
    n_moves_list: List[int],
    winners: List[str],
    use_center: bool = True,
    use_moves: bool = True,
    use_winner: bool = True,
) -> np.ndarray:
    rows = []

    for notation, n_moves, winner in zip(notations, n_moves_list, winners):
        parts = []

        if use_center:
            parts.append(center_features(notation))

        if use_moves:
            parts.append(move_count_onehot(n_moves))

        if use_winner:
            parts.append(winner_onehot(winner))

        if parts:
            rows.append(np.concatenate(parts, axis=0))
        else:
            rows.append(np.zeros(0, dtype=np.float32))

    return np.stack(rows, axis=0)


# Step 5: Concatenate
def concatenate_features(
    X_tfidf: sparse.csr_matrix,
    X_board: np.ndarray,
    X_extra: np.ndarray | None = None,
) -> sparse.csr_matrix:
    X_board_sparse = sparse.csr_matrix(X_board)

    if X_extra is not None and X_extra.shape[1] > 0:
        X_extra_sparse = sparse.csr_matrix(X_extra)
        X_all = sparse.hstack([X_tfidf, X_board_sparse, X_extra_sparse], format="csr")
    else:
        X_all = sparse.hstack([X_tfidf, X_board_sparse], format="csr")

    return X_all.tocsr()


# Feature names for debugging / analysis
def get_board_feature_names() -> List[str]:
    names = []
    for pos in POSITIONS:
        pos_name = CANONICAL_POSITIONS[pos]
        names.extend([
            f"{pos_name}_is_X",
            f"{pos_name}_is_O",
            f"{pos_name}_is_EMPTY",
        ])
    return names


def get_extra_feature_names(
    use_center: bool = True,
    use_moves: bool = True,
    use_winner: bool = True,
) -> List[str]:
    names = []

    if use_center:
        names.extend(["center_is_X", "center_is_O", "center_is_EMPTY"])

    if use_moves:
        names.extend([f"n_moves_{i}" for i in range(10)])

    if use_winner:
        names.extend(["winner_X", "winner_O", "winner_none"])

    return names


# Main pipeline
def build_type_c_text_features(
    input_csv: str | Path = INPUT_CSV,
    save_npz: str | Path = OUTPUT_FEATURES_NPZ,
    save_meta: str | Path = OUTPUT_META_PKL,
    use_extra_features: bool = True,
    use_center: bool = True,
    use_moves: bool = True,
    use_winner: bool = True,
):
    rows = load_sentence_dataset(input_csv)

    sentence_ids = [row["sentence_id"] for row in rows]
    raw_sentences = [row["sentence"] for row in rows]
    notations = [row["notation"] for row in rows]
    n_moves_list = [int(row["n_moves"]) for row in rows]
    winners = [row["winner"] for row in rows]

    # 1) normalise sentences
    norm_sentences = [normalise_sentence(s) for s in raw_sentences]

    # 2) TF-IDF
    X_tfidf, vectorizer = build_tfidf_features(norm_sentences)

    # 3) board one-hot
    X_board = build_board_onehot_features(notations)

    # 4) extra structured features
    X_extra = None
    extra_feature_names = []
    if use_extra_features:
        X_extra = build_extra_structured_features(
            notations=notations,
            n_moves_list=n_moves_list,
            winners=winners,
            use_center=use_center,
            use_moves=use_moves,
            use_winner=use_winner,
        )
        extra_feature_names = get_extra_feature_names(
            use_center=use_center,
            use_moves=use_moves,
            use_winner=use_winner,
        )

    # 5) concatenate
    X_all = concatenate_features(X_tfidf, X_board, X_extra)

    # 6) feature names
    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()
    board_feature_names = get_board_feature_names()
    all_feature_names = tfidf_feature_names + board_feature_names + extra_feature_names

    # 7) save sparse matrix
    save_npz = Path(save_npz)
    save_npz.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(save_npz, X_all)

    # 8) save metadata
    meta = {
        "sentence_ids": sentence_ids,
        "raw_sentences": raw_sentences,
        "normalised_sentences": norm_sentences,
        "notations": notations,
        "n_moves": n_moves_list,
        "winners": winners,
        "tfidf_vocab_size": X_tfidf.shape[1],
        "board_feature_dim": X_board.shape[1],
        "extra_feature_dim": 0 if X_extra is None else X_extra.shape[1],
        "total_feature_dim": X_all.shape[1],
        "feature_names": all_feature_names,
        "tfidf_feature_names": tfidf_feature_names,
        "board_feature_names": board_feature_names,
        "extra_feature_names": extra_feature_names,
        "vectorizer": vectorizer,
    }

    save_meta = Path(save_meta)
    with open(save_meta, "wb") as f:
        pickle.dump(meta, f)

    print("=" * 60)
    print("Type-C text features built successfully")
    print(f"Samples: {len(rows)}")
    print(f"TF-IDF dim: {X_tfidf.shape[1]}")
    print(f"Board one-hot dim: {X_board.shape[1]}")
    print(f"Extra feature dim: {0 if X_extra is None else X_extra.shape[1]}")
    print(f"Final fused dim: {X_all.shape[1]}")
    print(f"Saved sparse features to: {save_npz}")
    print(f"Saved metadata to: {save_meta}")
    print("=" * 60)

    return {
        "X_all": X_all,
        "X_tfidf": X_tfidf,
        "X_board": X_board,
        "X_extra": X_extra,
        "meta": meta,
    }


# Convenience loader
def load_saved_features(
    features_npz: str | Path = OUTPUT_FEATURES_NPZ,
    meta_pkl: str | Path = OUTPUT_META_PKL,
):
    X = sparse.load_npz(features_npz)
    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)
    return X, meta



# Script entry

if __name__ == "__main__":
    build_type_c_text_features(
        input_csv=INPUT_CSV,
        save_npz=OUTPUT_FEATURES_NPZ,
        save_meta=OUTPUT_META_PKL,
        use_extra_features=True,
        use_center=True,
        use_moves=True,
        use_winner=True,
    )
