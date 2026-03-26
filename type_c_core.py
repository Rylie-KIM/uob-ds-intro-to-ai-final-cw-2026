
#imageimage

import math
import random
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# Position vocabulary
# ============================================================

POSITIONS = {
    "TL": (0, 0), "TM": (0, 1), "TR": (0, 2),
    "ML": (1, 0), "C":  (1, 1), "MR": (1, 2),
    "BL": (2, 0), "BM": (2, 1), "BR": (2, 2),
}

ALIASES = {
    "TC": "TM",
    "BC": "BM",
    "CTR": "C",
    "M": "C",
    "MM": "C",
}

POSITION_TO_TEXT = {
    "TL": "top left",
    "TM": "top center",
    "TR": "top right",
    "ML": "middle left",
    "C":  "center",
    "MR": "middle right",
    "BL": "bottom left",
    "BM": "bottom center",
    "BR": "bottom right",
}


def canonical_pos_name(pos: str) -> str:
    pos = pos.strip().upper()
    return ALIASES.get(pos, pos)


def resolve(pos: str) -> tuple[int, int]:
    pos = canonical_pos_name(pos)
    if pos not in POSITIONS:
        raise ValueError(f"Unknown position: {pos}")
    return POSITIONS[pos]


def sort_positions(pos_list: list[str]) -> list[str]:
    order = ["TL", "TM", "TR", "ML", "C", "MR", "BL", "BM", "BR"]
    index = {name: i for i, name in enumerate(order)}
    return sorted([canonical_pos_name(p) for p in pos_list], key=lambda p: index[p])


def join_naturally(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ============================================================
# Board
# ============================================================

@dataclass
class Board:
    x: list[str] = field(default_factory=list)
    o: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.x = sort_positions(self.x)
        self.o = sort_positions(self.o)
        self._validate()

    def _validate(self):
        x_coords = [resolve(p) for p in self.x]
        o_coords = [resolve(p) for p in self.o]
        all_coords = x_coords + o_coords

        if len(all_coords) != len(set(all_coords)):
            raise ValueError("Duplicate positions detected.")

        if not (len(self.x) == len(self.o) or len(self.x) == len(self.o) + 1):
            raise ValueError(
                f"Invalid move counts: X={len(self.x)}, O={len(self.o)}. "
                "X must have equal or one more move than O."
            )

    def to_grid(self) -> list[list[str]]:
        grid = [[" "] * 3 for _ in range(3)]
        for pos in self.x:
            r, c = resolve(pos)
            grid[r][c] = "X"
        for pos in self.o:
            r, c = resolve(pos)
            grid[r][c] = "O"
        return grid

    def winner(self) -> Optional[str]:
        grid = self.to_grid()
        lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]
        for line in lines:
            values = {grid[r][c] for r, c in line}
            if values == {"X"}:
                return "X"
            if values == {"O"}:
                return "O"
        return None

    def __str__(self) -> str:
        grid = self.to_grid()
        rows = []
        for i, row in enumerate(grid):
            rows.append(" " + " | ".join(row) + " ")
            if i < 2:
                rows.append("---+---+---")
        return "\n".join(rows)


def to_notation(board: Board) -> str:
    parts = []
    if board.x:
        parts.append("X:" + ",".join(board.x))
    if board.o:
        parts.append("O:" + ",".join(board.o))
    return " ".join(parts)


def _is_reachable(board: Board) -> bool:
    w = board.winner()

    if w == "X" and len(board.x) != len(board.o) + 1:
        return False
    if w == "O" and len(board.x) != len(board.o):
        return False

    last_player = "x" if len(board.x) > len(board.o) else "o"
    prev_x = board.x[:-1] if last_player == "x" else board.x
    prev_o = board.o[:-1] if last_player == "o" else board.o

    if prev_x or prev_o or len(board.x) + len(board.o) > 1:
        prev_board = Board(x=prev_x, o=prev_o)
        if prev_board.winner() is not None:
            return False

    return True


def random_board(num_moves: Optional[int] = None) -> Board:
    all_positions = list(POSITIONS.keys())

    while True:
        if num_moves is None:
            n = random.randint(0, 9)
        else:
            if not 0 <= num_moves <= 9:
                raise ValueError("num_moves must be between 0 and 9.")
            n = num_moves

        chosen = random.sample(all_positions, n)
        x_count = (n + 1) // 2
        x_pos = chosen[:x_count]
        o_pos = chosen[x_count:]

        board = Board(x=x_pos, o=o_pos)
        if _is_reachable(board):
            return board


# ============================================================
# Sentence generator (wenjia part)
# ============================================================

def board_to_sentence(board: Board) -> str:
    if not board.x and not board.o:
        return "The board is empty."

    x_text = join_naturally([POSITION_TO_TEXT[p] for p in board.x])
    o_text = join_naturally([POSITION_TO_TEXT[p] for p in board.o])

    parts = []

    if board.x:
        parts.append(f"X is in the {x_text}")

    if board.o:
        parts.append(f"O is in the {o_text}")

    return ". ".join(parts) + "."


# ============================================================
# Image generator (zhimao's part)
# ============================================================

def _rotate_translate(img: Image.Image, angle_deg: float, dx: int, dy: int) -> Image.Image:
    img = img.rotate(angle_deg, fillcolor=255, resample=Image.Resampling.BILINEAR)
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

    def _dilate(mask):
        return (
            np.roll(mask, 1, axis=0) | np.roll(mask, -1, axis=0)
            | np.roll(mask, 1, axis=1) | np.roll(mask, -1, axis=1)
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
    lw = max(3, size // 100)
    pad = cell // 6

    canvas = np.full((size, size), 255.0)

    grid_img = Image.new("L", (size, size), color=255)
    gd = ImageDraw.Draw(grid_img)
    for i in (1, 2):
        x = margin + i * cell
        gd.line([(x, margin), (x, margin + grid_size)], fill=0, width=lw)
        y = margin + i * cell
        gd.line([(margin, y), (margin + grid_size, y)], fill=0, width=lw)

    angle = random.uniform(-math.pi / 16, math.pi / 16) * 180 / math.pi
    dx, dy = random.randint(-3, 3), random.randint(-3, 3)
    grid_img = _rotate_translate(grid_img, angle, dx, dy)
    canvas = np.minimum(canvas, np.array(grid_img, dtype=float))

    board_grid = board.to_grid()
    for r in range(3):
        for c in range(3):
            sym = board_grid[r][c]
            if sym == " ":
                continue

            sym_img = Image.new("L", (cell, cell), color=255)
            sd = ImageDraw.Draw(sym_img)

            if sym == "X":
                sd.line([(pad, pad), (cell - pad, cell - pad)], fill=0, width=lw * 2)
                sd.line([(cell - pad, pad), (pad, cell - pad)], fill=0, width=lw * 2)
            elif sym == "O":
                sd.ellipse([(pad, pad), (cell - pad, cell - pad)], outline=0, width=lw * 2)

            angle = random.uniform(-math.pi / 16, math.pi / 16) * 180 / math.pi
            dx, dy = random.randint(-3, 3), random.randint(-3, 3)
            sym_img = _rotate_translate(sym_img, angle, dx, dy)

            stamp = np.full((size, size), 255.0)
            py, px = margin + r * cell, margin + c * cell
            stamp[py:py + cell, px:px + cell] = np.array(sym_img, dtype=float)
            canvas = np.minimum(canvas, stamp)

    arr = 1.0 - canvas / 255.0
    return _blur_edges(arr)


def save_board_image(board: Board, output_path: str, size: int = 500) -> None:
    arr = render_board_array(board, size=size)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img.save(output_path)