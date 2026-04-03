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

POSITION_PHRASES = {
    "C":  ["middle", "center"],
    "ML": ["middle row left", "left of centre"],
    "MR": ["middle row right", "right of centre"],
    "TM": ["top middle", "above the center"],
    "BM": ["bottom middle", "below the center"],
    "TL": ["top left corner", "up and left from center"],
    "TR": ["top right corner", "up and right from center"],
    "BL": ["bottom left corner", "down and left from center"],
    "BR": ["bottom right corner", "down and right from center"],
}

NAMES = ["Seoyeon", "Sujith", "Fergus", "Wenjia", "Zhenmao", "Kim", "Watson", "Goli", "SONG", "Wang"]

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


def join_naturally(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _phrase(pos: str) -> str:
    """Pick a random English phrase for a position."""
    pos = pos.strip().upper()
    pos = ALIASES.get(pos, pos)
    return random.choice(POSITION_PHRASES[pos])


def _pick_names(x_name: Optional[str], o_name: Optional[str]) -> tuple[str, str]:
    """Return (x_name, o_name), filling in random names as needed."""
    if x_name and o_name:
        return x_name, o_name
    available = [n for n in NAMES if n not in (x_name, o_name)]
    random.shuffle(available)
    if not x_name:
        x_name = available.pop()
    if not o_name:
        o_name = available.pop()
    return x_name, o_name


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


def board_to_sentence(board: Board, x_name: Optional[str] = None, o_name: Optional[str] = None) -> str:
    """
    Return an English sentence describing the board position.

    Names are chosen randomly from NAMES if not provided.
    Each position is described using one of its multiple alternative phrasings.
    Verbs and phrasing are randomized for variety.
    """
    if not board.x and not board.o:
        return "The board is empty."

    name_x, name_o = _pick_names(x_name, o_name)
    parts = []

    if board.x:
        x_phrases = join_naturally([_phrase(p) for p in board.x])
        verb = random.choice(["gone for", "taken"])
        parts.append(f"{name_x} is X and has {verb} {x_phrases}")
    else:
        parts.append(f"{name_x} is X and has not moved yet")

    if board.o:
        o_phrases = join_naturally([_phrase(p) for p in board.o])
        verb = random.choice(["gone for", "taken"])
        parts.append(f"{name_o} is O and has {verb} {o_phrases}")
    else:
        parts.append(f"{name_o} is O and has not moved yet")

    return "; ".join(parts) + "."


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