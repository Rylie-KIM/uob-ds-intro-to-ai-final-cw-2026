import json
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw

from type_c_core import Board, parse_notation


def _rotate_translate(img: Image.Image, angle_deg: float, dx: int, dy: int) -> Image.Image:
    img = img.rotate(
        angle_deg,
        fillcolor=255,
        resample=Image.Resampling.BILINEAR
    )
    img = img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        fillcolor=255,
        resample=Image.Resampling.BILINEAR,
    )
    return img


def _blur_edges(arr: np.ndarray) -> np.ndarray:
    result = arr.copy()
    fg = arr > 0.5

    def _dilate(mask: np.ndarray) -> np.ndarray:
        return (
            np.roll(mask, 1, axis=0) | np.roll(mask, -1, axis=0) |
            np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
        )

    bg_adj = ~fg & _dilate(fg)
    fg_adj = fg & _dilate(~fg)

    result[bg_adj] = np.random.uniform(0.0, 0.5, bg_adj.sum())
    result[fg_adj] = np.random.uniform(0.5, 1.0, fg_adj.sum())

    return result


def render_board_array(board: Board, size: int = 500) -> np.ndarray:
    margin = size // 10
    grid_size = size - 2 * margin
    cell = grid_size // 3
    line_width = max(3, size // 100)
    pad = cell // 6

    canvas = np.full((size, size), 255.0)

    grid_img = Image.new("L", (size, size), color=255)
    draw_grid = ImageDraw.Draw(grid_img)

    for i in (1, 2):
        x = margin + i * cell
        y = margin + i * cell
        draw_grid.line([(x, margin), (x, margin + grid_size)], fill=0, width=line_width)
        draw_grid.line([(margin, y), (margin + grid_size, y)], fill=0, width=line_width)

    angle = random.uniform(-math.pi / 16, math.pi / 16) * 180 / math.pi
    dx, dy = random.randint(-3, 3), random.randint(-3, 3)
    grid_img = _rotate_translate(grid_img, angle, dx, dy)
    canvas = np.minimum(canvas, np.array(grid_img, dtype=float))

    board_grid = board.to_grid()

    for r in range(3):
        for c in range(3):
            symbol = board_grid[r][c]
            if symbol == " ":
                continue

            symbol_img = Image.new("L", (cell, cell), color=255)
            draw_symbol = ImageDraw.Draw(symbol_img)

            if symbol == "X":
                draw_symbol.line([(pad, pad), (cell - pad, cell - pad)], fill=0, width=line_width * 2)
                draw_symbol.line([(cell - pad, pad), (pad, cell - pad)], fill=0, width=line_width * 2)
            elif symbol == "O":
                draw_symbol.ellipse([(pad, pad), (cell - pad, cell - pad)], outline=0, width=line_width * 2)

            angle = random.uniform(-math.pi / 16, math.pi / 16) * 180 / math.pi
            dx, dy = random.randint(-3, 3), random.randint(-3, 3)
            symbol_img = _rotate_translate(symbol_img, angle, dx, dy)

            stamp = np.full((size, size), 255.0)
            py, px = margin + r * cell, margin + c * cell
            stamp[py:py + cell, px:px + cell] = np.array(symbol_img, dtype=float)
            canvas = np.minimum(canvas, stamp)

    arr = 1.0 - canvas / 255.0
    return _blur_edges(arr)


def save_board_image(board: Board, output_path: str, size: int = 500) -> None:
    arr = render_board_array(board, size=size)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img.save(output_path)


def generate_images_from_dataset(
    dataset_file: str = "data/type_c_dataset.json",
    image_dir: str = "data/type_c_images",
    image_size: int = 500
) -> None:
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    os.makedirs(image_dir, exist_ok=True)

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for record in dataset:
        board = parse_notation(record["notation"])
        image_filename = record.get("image", f"type_c_{record['id']}.png")
        image_path = os.path.join(image_dir, image_filename)

        save_board_image(board, image_path, size=image_size)
        print(f"Generated image for sample {record['id']}: {image_filename}")

    print(f"\nAll images saved to: {image_dir}")


if __name__ == "__main__":
    generate_images_from_dataset()