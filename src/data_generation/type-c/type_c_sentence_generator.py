import csv
import random
from pathlib import Path

from type_c_core import generate_all_boards, to_notation, board_to_sentence, SEED


random.seed(SEED)


SRC_ROOT = Path(__file__).resolve().parents[2]
TYPE_C_DATA_DIR = SRC_ROOT / "data" / "type-c"
DEFAULT_OUTPUT_CSV = TYPE_C_DATA_DIR / "sentences_c.csv"


def generate_sentence_dataset(
    limit: int = 5269,
    output_csv: str = str(DEFAULT_OUTPUT_CSV)
) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    boards = generate_all_boards()
    print(f"Total enumerated valid boards: {len(boards)}")

    records = []
    for i, board in enumerate(boards[:limit]):
        sentence_id = f"c_{i}"
        winner = board.winner() or ""
        records.append({
            "sentence_id": sentence_id,
            "sentence": board_to_sentence(board),
            "notation": to_notation(board),
            "n_moves": len(board.x) + len(board.o),
            "winner": winner,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sentence_id", "sentence", "notation", "n_moves", "winner"],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} sentence samples")
    print(f"Sentences saved to: {output_path}")


if __name__ == "__main__":
    print(f"[type-c] Using SEED={SEED} for reproducibility")
    generate_sentence_dataset(limit=5269) # whole valid cases 