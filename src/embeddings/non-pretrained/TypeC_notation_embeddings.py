import json
import os
import numpy as np


# Fixed chessboard sequence
# TL TM TR
# ML C  MR
# BL BM BR
POSITION_ORDER = ["TL", "TM", "TR", "ML", "C", "MR", "BL", "BM", "BR"]
POSITION_TO_INDEX = {pos: i for i, pos in enumerate(POSITION_ORDER)}


def parse_notation(notation: str) -> dict:
    """
    Parse notation like:
        'X:TL,TR O:TM'
    into:
        {'X': ['TL', 'TR'], 'O': ['TM']}
    """
    result = {"X": [], "O": []}

    notation = notation.strip()
    if not notation:
        return result

    parts = notation.split()
    for part in parts:
        if ":" not in part:
            continue

        player, cells = part.split(":", 1)
        player = player.strip().upper()

        if player not in {"X", "O"}:
            continue

        positions = [c.strip().upper() for c in cells.split(",") if c.strip()]
        result[player].extend(positions)

    return result


def notation_to_compact_vector(notation: str) -> np.ndarray:
    """
    9D compact encoding:
        empty = 0
        X = 1
        O = -1

    Example:
        'X:TL,TR O:TM'
        -> [1, -1, 1, 0, 0, 0, 0, 0, 0]
    """
    vec = np.zeros(9, dtype=np.int8)
    parsed = parse_notation(notation)

    for pos in parsed["X"]:
        if pos in POSITION_TO_INDEX:
            vec[POSITION_TO_INDEX[pos]] = 1

    for pos in parsed["O"]:
        if pos in POSITION_TO_INDEX:
            vec[POSITION_TO_INDEX[pos]] = -1

    return vec


def notation_to_onehot_vector(notation: str) -> np.ndarray:
    """
    27D one-hot encoding.
    Each cell uses 3 dims:
        [1, 0, 0] = empty
        [0, 1, 0] = X
        [0, 0, 1] = O

    Total dims = 9 * 3 = 27
    """
    board = ["empty"] * 9
    parsed = parse_notation(notation)

    for pos in parsed["X"]:
        if pos in POSITION_TO_INDEX:
            board[POSITION_TO_INDEX[pos]] = "X"

    for pos in parsed["O"]:
        if pos in POSITION_TO_INDEX:
            board[POSITION_TO_INDEX[pos]] = "O"

    onehot = []
    for cell in board:
        if cell == "empty":
            onehot.extend([1, 0, 0])
        elif cell == "X":
            onehot.extend([0, 1, 0])
        elif cell == "O":
            onehot.extend([0, 0, 1])

    return np.array(onehot, dtype=np.int8)


def encode_dataset(json_path: str, mode: str = "onehot"):
    """
    Read dataset JSON and convert notation to vectors.

    Args:
        json_path: path to dataset json
        mode: 'compact' or 'onehot'

    Returns:
        ids: list[int]
        notations: list[str]
        embeddings: np.ndarray
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []
    notations = []
    embeddings = []

    for item in data:
        sample_id = item["id"]
        notation = item["notation"]

        if mode == "compact":
            vec = notation_to_compact_vector(notation)
        elif mode == "onehot":
            vec = notation_to_onehot_vector(notation)
        else:
            raise ValueError("mode must be 'compact' or 'onehot'")

        ids.append(sample_id)
        notations.append(notation)
        embeddings.append(vec)

    embeddings = np.array(embeddings)
    return ids, notations, embeddings


def save_embeddings(output_path: str, embeddings: np.ndarray):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    json_path = os.path.join(BASE_DIR, "data", "type_c_dataset.json")

    # Optional: "compact" or "onehot"
    mode = "onehot"

    ids, notations, embeddings = encode_dataset(json_path, mode=mode)

    print("Mode:", mode)
    print("Embeddings shape:", embeddings.shape)

    for i in range(min(5, len(ids))):
        print(f"\nID: {ids[i]}")
        print("Notation:", notations[i])
        print("Embedding:", embeddings[i])

    output_path = f"../../results/metrics/typec_notation_{mode}.npy"
    save_embeddings(output_path, embeddings)
    print(f"\nSaved embeddings to: {output_path}")
    # =========================
    # Complex Sample Test
    # =========================
    print("\n===== Complex Sample Test =====")

    test_notation = "X:TL,TR O:TM"
    vec = notation_to_onehot_vector(test_notation)

    print("Notation:", test_notation)

    for i in range(9):
        cell = vec[i*3:(i+1)*3]
        print(POSITION_ORDER[i], cell)