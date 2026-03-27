from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional


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


def sort_positions(pos_list: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(POSITIONS)}
    return sorted(pos_list, key=lambda p: order[p])


def join_naturally(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


@dataclass
class Board:
    x: list[str] = field(default_factory=list)
    o: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.x = sort_positions(self.x)
        self.o = sort_positions(self.o)

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
                x_positions = [p.strip() for p in value.split(",") if p.strip()]
        elif part.startswith("O:"):
            value = part[2:]
            if value:
                o_positions = [p.strip() for p in value.split(",") if p.strip()]

    return Board(x=x_positions, o=o_positions)


def board_to_sentence(board: Board) -> str:
    if not board.x and not board.o:
        return "The board is empty."

    parts = []

    if board.x:
        x_text = join_naturally([POSITION_TO_TEXT[p] for p in board.x])
        parts.append(f"X is in the {x_text}")

    if board.o:
        o_text = join_naturally([POSITION_TO_TEXT[p] for p in board.o])
        parts.append(f"O is in the {o_text}")

    return ". ".join(parts) + "."


def is_valid_move_count(board: Board) -> bool:
    return len(board.x) == len(board.o) or len(board.x) == len(board.o) + 1


def is_valid_winner_state(board: Board) -> bool:
    winner = board.winner()

    if winner == "X" and len(board.x) != len(board.o) + 1:
        return False
    if winner == "O" and len(board.x) != len(board.o):
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
                if not is_valid_winner_state(board):
                    continue

                boards.append(board)

    return boards