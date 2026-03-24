

import json
import os

from type_c_core import random_board, to_notation, board_to_sentence, save_board_image


def generate_dataset(n_samples=100, image_size=500):
    image_dir = "data/type_c_images"
    dataset_file = "data/type_c_dataset.json"

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    dataset = []

    for i in range(1, n_samples + 1):
        board = random_board()

        image_filename = f"type_c_{i}.png"
        image_path = os.path.join(image_dir, image_filename)

        sentence = board_to_sentence(board)
        save_board_image(board, image_path, size=image_size)

        record = {
            "id": i,
            "notation": to_notation(board),
            "sentence": sentence,
            "image": image_filename
        }

        dataset.append(record)

        print(f"Generated sample {i}: {record}")

    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Dataset saved to: {dataset_file}")
    print(f"Images saved to: {image_dir}")


if __name__ == "__main__":
    # 开发测试先改成 5
    # 最终建议 100
    generate_dataset(n_samples=100, image_size=500)#最后100 500#5 100