"""
generate_sentences_b.py
Phase 1 — Type B (MNIST Numbers)

Direction B: enumerate ALL unique (size, colour, number) combinations.
1 sentence per unique combination — no duplicates, no augmentation.

Sentence format: "{size} {colour} {number}"
Examples: "large blue 167", "small red 42", "large yellow 131337"

"""

import csv
import os
import random
import itertools

SEED = 42
random.seed(SEED)

COLOURS  = ['red', 'blue', 'green', 'yellow']
SIZES    = ['large', 'small']
TEMPLATE = '{size} {colour} {number}'

DIGIT_CONFIG = {
    1: list(range(1, 10)),                           #   9 numbers: 1–9
    2: list(range(10, 100)),                         #  90 numbers: 10–99
    3: random.sample(range(100,    1_000),    288),  # 288 sampled
    4: random.sample(range(1_000,  10_000),   288),  # 288 sampled
    5: random.sample(range(10_000, 100_000),  288),  # 288 sampled
    6: random.sample(range(100_000, 1_000_000), 288),# 288 sampled
}

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../../data/type-b/sentences_b.csv')


def generate(output_path=OUTPUT_PATH):
    all_numbers = [
        (num, n_digits)
        for n_digits, numbers in DIGIT_CONFIG.items()
        for num in numbers
    ]

    records = []
    idx = 0
    for (number, n_digits), (size, colour) in itertools.product(
            all_numbers, itertools.product(SIZES, COLOURS)):
        records.append({
            'sentence_id': f'b_{idx}',
            'sentence':    TEMPLATE.format(size=size, colour=colour, number=number),
            'n_digits':    n_digits,
        })
        idx += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sentence_id', 'sentence', 'n_digits'])
        writer.writeheader()
        writer.writerows(records)

    print(f"[type-b] {len(records)} unique sentences saved → {output_path}")
    print(f"\nbreakdown per digit length:")
    tot = 0 
    for n_digits, nums in DIGIT_CONFIG.items():
        count = len(nums) * len(SIZES) * len(COLOURS)
        print(f"\n{n_digits}-digit: {len(nums)} numbers × 8 = {count} sentences")
        tot += count
    print(f'\ntot sample sentences number: {tot}')
    return output_path


if __name__ == '__main__':
    generate()
