import csv
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from type_c_core import Board, parse_notation, SEED


random.seed(SEED)
np.random.seed(SEED)


SRC_ROOT = Path(__file__).resolve().parents[2]
TYPE_C_DATA_DIR = SRC_ROOT / "data" / "type-c"
TYPE_C_IMAGE_DIR = SRC_ROOT / "data" / "images" / "type-c"

INPUT_CSV = TYPE_C_DATA_DIR / "sentences_c.csv"
OUTPUT_DIR = TYPE_C_IMAGE_DIR
OUTPUT_MAP = TYPE_C_DATA_DIR / "image_map_c.csv"


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


def generate(
    input_csv: str = str(INPUT_CSV),
    output_dir: str = str(OUTPUT_DIR),
    output_map: str = str(OUTPUT_MAP),
    image_size: int = 500,
) -> None:
    input_path = Path(input_csv)
    output_dir_path = Path(output_dir)
    output_map_path = Path(output_map)

    if not input_path.exists():
        raise FileNotFoundError(f"Sentences file not found: {input_path}")

    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_map_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records = []
    for i, row in enumerate(rows):
        sentence_id = row["sentence_id"]
        notation = row.get("notation", "")
        if notation is None:
            notation = ""

        board = parse_notation(notation)
        filename = f"{sentence_id}.png"
        image_path = output_dir_path / filename

        save_board_image(board, str(image_path), size=image_size)
        records.append({"filename": filename, "sentence_id": sentence_id})

        if i % 500 == 0:
            print(f"  {i}/{len(rows)} processed...")

    with open(output_map_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "sentence_id"])
        writer.writeheader()
        writer.writerows(records)

    print(f"[type-c] {len(records)} images saved >> {output_dir_path}")
    print(f"[type-c] image map saved >> {output_map_path}")


if __name__ == "__main__":
    generate()