from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional
import random


SEED = 42
random.seed(SEED)

POSITIONS = [
    "TL", "TM", "TR",
    "ML", "C", "MR",
    "BL", "BM", "BR"
]

POSITION_TO_COORD = {
    "TL": (0, 0), "TM": (0, 1), "TR": (0, 2),
    "ML": (1, 0), "C":  (1, 1), "MR": (1, 2),
    "BL": (2, 0), "BM": (2, 1), "BR": (2, 2),
}

# Canonical position text
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

# Controlled randomness:
# keep some semantically meaningful variation in position expressions
POSITION_VARIANTS = {
    "TL": ["top left", "upper left"],
    "TM": ["top center", "top middle"],
    "TR": ["top right", "upper right"],
    "ML": ["middle left", "center left"],
    "C":  ["center", "middle"],
    "MR": ["middle right", "center right"],
    "BL": ["bottom left", "lower left"],
    "BM": ["bottom center", "bottom middle"],
    "BR": ["bottom right", "lower right"],
}

ALIASES = {
    "TC": "TM",
    "BC": "BM",
    "CTR": "C",
    "M": "C",
    "MM": "C",
}


def resolve_position(pos: str) -> str:
    name = pos.strip().upper()
    name = ALIASES.get(name, name)
    if name not in POSITION_TO_COORD:
        valid = ", ".join(POSITIONS)
        raise ValueError(f"Unknown position '{pos}'. Valid positions: {valid}")
    return name


def sort_positions(pos_list: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(POSITIONS)}
    return sorted(pos_list, key=lambda p: order[p])


def join_naturally(items: list[str]) -> list[str] | str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def sample_position_phrase(pos: str) -> str:
    pos = resolve_position(pos)
    return random.choice(POSITION_VARIANTS[pos])


@dataclass
class Board:
    x: list[str] = field(default_factory=list)
    o: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.x = sort_positions([resolve_position(p) for p in self.x])
        self.o = sort_positions([resolve_position(p) for p in self.o])
        self._validate()

    def _validate(self) -> None:
        occupied = self.x + self.o
        if len(occupied) != len(set(occupied)):
            raise ValueError("Duplicate positions detected across X and O.")

        if not is_valid_move_count(self):
            raise ValueError(
                f"Invalid move counts: X={len(self.x)}, O={len(self.o)}. "
                "X must have equal or one more move than O."
            )

    def to_grid(self) -> list[list[str]]:
        grid = [[" "] * 3 for _ in range(3)]
        for pos in self.x:
            r, c = POSITION_TO_COORD[pos]
            grid[r][c] = "X"
        for pos in self.o:
            r, c = POSITION_TO_COORD[pos]
            grid[r][c] = "O"
        return grid

    def winner(self) -> Optional[str]:
        g = self.to_grid()
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
            values = {g[r][c] for r, c in line}
            if values == {"X"}:
                return "X"
            if values == {"O"}:
                return "O"
        return None


def to_notation(board: Board) -> str:
    parts = []
    if board.x:
        parts.append("X:" + ",".join(board.x))
    if board.o:
        parts.append("O:" + ",".join(board.o))
    return " ".join(parts)


def parse_notation(notation: str) -> Board:
    x_positions = []
    o_positions = []

    notation = notation.strip()
    if not notation:
        return Board()

    parts = notation.split()
    for part in parts:
        if part.startswith("X:"):
            value = part[2:]
            if value:
                x_positions = [resolve_position(p) for p in value.split(",") if p.strip()]
        elif part.startswith("O:"):
            value = part[2:]
            if value:
                o_positions = [resolve_position(p) for p in value.split(",") if p.strip()]
        else:
            raise ValueError(
                f"Expected token starting with 'X:' or 'O:', got '{part}'."
            )

    return Board(x=x_positions, o=o_positions)


def board_to_sentence(board: Board) -> str:
    """
    Controlled-randomness sentence generator:
    - removes random player names
    - removes irrelevant verb variation
    - keeps semantically meaningful variation in position phrases
    - keeps sentence structure simple and learnable
    """
    if not board.x and not board.o:
        return "The board is empty."

    sentence_patterns = []

    if board.x and board.o:
        sentence_patterns = [
            "X is in {x_pos}; O is in {o_pos}.",
            "{x_pos} has X; {o_pos} has O.",
            "There is X in {x_pos} and O in {o_pos}.",
        ]
    elif board.x:
        sentence_patterns = [
            "X is in {x_pos}.",
            "{x_pos} has X.",
            "There is X in {x_pos}.",
        ]
    else:
        sentence_patterns = [
            "O is in {o_pos}.",
            "{o_pos} has O.",
            "There is O in {o_pos}.",
        ]

    x_pos = join_naturally([sample_position_phrase(p) for p in board.x]) if board.x else ""
    o_pos = join_naturally([sample_position_phrase(p) for p in board.o]) if board.o else ""

    template = random.choice(sentence_patterns)
    return template.format(x_pos=x_pos, o_pos=o_pos)


def is_valid_move_count(board: Board) -> bool:
    return len(board.x) == len(board.o) or len(board.x) == len(board.o) + 1


def is_valid_winner_state(board: Board) -> bool:
    winner = board.winner()

    if winner == "X" and len(board.x) != len(board.o) + 1:
        return False
    if winner == "O" and len(board.x) != len(board.o):
        return False

    return True


def is_reachable(board: Board) -> bool:
    if not is_valid_winner_state(board):
        return False

    last_player = "x" if len(board.x) > len(board.o) else "o"
    prev_x = board.x[:-1] if last_player == "x" else board.x
    prev_o = board.o[:-1] if last_player == "o" else board.o

    if prev_x or prev_o or (len(board.x) + len(board.o) > 1):
        prev_board = Board(x=prev_x, o=prev_o)
        if prev_board.winner() is not None:
            return False

    return True


def generate_all_boards() -> list[Board]:
    boards = []

    for n in range(10):
        x_count = (n + 1) // 2
        o_count = n // 2

        for occupied_cells in combinations(POSITIONS, n):
            for x_cells in combinations(occupied_cells, x_count):
                x_set = set(x_cells)
                o_cells = [cell for cell in occupied_cells if cell not in x_set]

                board = Board(x=list(x_cells), o=o_cells)

                if not is_valid_move_count(board):
                    continue
                if not is_reachable(board):
                    continue

                boards.append(board)

    return boards
