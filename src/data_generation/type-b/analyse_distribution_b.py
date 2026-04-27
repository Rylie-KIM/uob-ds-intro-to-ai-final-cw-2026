
import csv
import os
import sys
from collections import Counter
from pathlib import Path


_root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.config.paths import TYPE_B_SENTENCES as _CSV_PATH

# CSV_PATH = os.path.join(os.path.dirname(__file__), '../../data/type-b/sentences_b.csv')
CSV_PATH = str(_CSV_PATH)


def analyse(csv_path: str = CSV_PATH) -> None:
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    sizes   = Counter(r['sentence'].split()[0] for r in rows)
    colours = Counter(r['sentence'].split()[1] for r in rows)
    digits  = Counter(r['n_digits'] for r in rows)

    print(f"Total sentences : {total}")
    print()

    print("Size distribution:")
    for label, count in sorted(sizes.items()):
        print(f"  {label:<8} {count:>5}  ({count/total*100:.1f}%)")
    print()

    print("Colour distribution:")
    for label, count in sorted(colours.items()):
        print(f"  {label:<8} {count:>5}  ({count/total*100:.1f}%)")
    print()

    print("Digit length distribution:")
    for length, count in sorted(digits.items(), key=lambda x: int(x[0])):
        print(f"  {length}-digit  {count:>5}  ({count/total*100:.1f}%)")


if __name__ == '__main__':
    analyse()
