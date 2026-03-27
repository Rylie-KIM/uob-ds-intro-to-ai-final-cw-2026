import json
import os

from type_c_core import generate_all_boards, to_notation, board_to_sentence


def generate_sentence_dataset(
    limit: int = 5000,
    output_file: str = "data/type_c_dataset.json"
) -> None:
    os.makedirs("data", exist_ok=True)

    boards = generate_all_boards()
    print(f"Total enumerated valid boards: {len(boards)}")

    dataset = []
    for i, board in enumerate(boards[:limit], start=1):
        record = {
            "id": i,
            "notation": to_notation(board),
            "sentence": board_to_sentence(board),
            "image": f"type_c_{i}.png"
        }
        dataset.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(dataset)} sentence samples")
    print(f"Dataset saved to: {output_file}")


if __name__ == "__main__":
    generate_sentence_dataset(limit=5000)