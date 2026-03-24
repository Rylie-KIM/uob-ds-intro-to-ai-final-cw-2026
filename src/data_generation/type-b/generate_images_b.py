"""
Type B (MNIST Numbers)

Reads sentences_b.csv and generates one image per sentence (Direction B).
MNIST is downloaded automatically via torchvision on first run.
"""

import csv
import os
import random
import numpy as np
from PIL import Image
from typing import TypeAlias
import torchvision.datasets as dsets

# ── data model ────────────────────────────────────────────────────────────────


# A colour-tinted digit patch after apply_colour() and resize
# Third axis = [R, G, B] channels, e.g. rgb[row, col] → [r, g, b]
RGBPatch: TypeAlias = np.ndarray     # shape (DIGIT_H, DIGIT_W, 3), dtype=uint8

# A horizontally stitched row of digit patches before canvas resize
# width = DIGIT_W × n_digits, e.g. 6-digit → shape (32, 192, 3)
Stitched: TypeAlias = np.ndarray     # shape (DIGIT_H, DIGIT_W * n_digits, 3), dtype=uint8

# RGB colour vector used for tinting, e.g. [220., 50., 50.] for red
ColourVec: TypeAlias = np.ndarray    # shape (3,), dtype=float32

# A single MNIST digit image: 2D grayscale array, pixel values 0 (black)–255 (white)
DigitImage: TypeAlias = np.ndarray   # shape (28, 28), dtype=uint8

# Maps each digit 0–9 to a list of its MNIST images
# e.g. digit_bank[3] → list of ~6,000 DigitImage arrays
DigitBank: TypeAlias = dict[int, list[DigitImage]]

MNIST_DIGIT_H = 32     
MNIST_DIGIT_W = 32  

# 4-digit raw width = 128px = canvas width (exact fit)
# 5,6-digit raw width > 128px → compressed to fit canvas
CANVAS_W = 128   
CANVAS_H = 128    
BG  = (245, 245, 245)
IMAGE_MODE = "RGB"

COLOUR_MAP = {
    'red':    np.array([220, 50,  50],  dtype=np.float32),
    'blue':   np.array([50,  100, 220], dtype=np.float32),
    'green':  np.array([50,  180, 50],  dtype=np.float32),
    'yellow': np.array([230, 200, 30],  dtype=np.float32),
}

SIZE_SCALE = {
    'large': 1.0,
    'small': 0.55,
}

INPUT_CSV  = os.path.join(os.path.dirname(__file__), '../../data/type-b/sentences_b.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../data/images/type-b')
OUTPUT_MAP = os.path.join(os.path.dirname(__file__), '../../data/type-b/image_map_b.csv')
MNIST_ROOT = os.path.join(os.path.dirname(__file__), '../../data/type-b/mnist_raw')

def load_mnist_by_digit(root: str) -> DigitBank:
    dataset    = dsets.MNIST(root=root, train=True, download=True)

    digit_bank = {} 
    for digit in range(10): 
        digit_bank[digit] = [] 

    for img, label in dataset:
        digit_bank[label].append(np.array(img))

    # sanity check 
    total = sum(len(v) for v in digit_bank.values())
    print(f"MNIST loaded: {total} images across 10 digits")

    return digit_bank


def apply_colour(gray_digit: DigitImage, colour_rgb: ColourVec, bg: tuple = BG) -> RGBPatch:
    # MNIST is black-background / white-stroke → stroke=1.0, background=0.0
    norm = gray_digit.astype(np.float32) / 255.0
    bg_f = np.array(bg, dtype=np.float32)
    rgb  = np.zeros((*gray_digit.shape, 3), dtype=np.uint8)
    for c in range(3):
        rgb[:, :, c] = (norm * colour_rgb[c] + (1.0 - norm) * bg_f[c]).astype(np.uint8)
    return rgb


def compose_image(number: int, colour: str, size_label: str, digit_bank: DigitBank, seed: int) -> Image.Image:
    rng        = random.Random(seed)
    digits     = [int(d) for d in str(number)]
    colour_rgb = COLOUR_MAP[colour]

    # build coloured digit patches
    patches = []
    for d in digits:
        raw    = rng.choice(digit_bank[d])                      
        tinted = apply_colour(raw, colour_rgb)    
        patch  = Image.fromarray(tinted).resize(
                     (MNIST_DIGIT_W, MNIST_DIGIT_H), Image.BILINEAR)   
        patches.append(np.array(patch))

    # stitch digits horizontally
    stitched  = np.concatenate(patches, axis=1)
    digit_img = Image.fromarray(stitched)

    # resize stitch to fit canvas, apply size scaling, centre
    canvas   = Image.new(IMAGE_MODE, (CANVAS_W, CANVAS_H), BG)
    scale    = SIZE_SCALE[size_label]
    target_w = int(CANVAS_W * scale)
    target_h = int(CANVAS_H * scale)

    # resize stitch to (target_w × target_h) — compresses wide stitches
    # only done on the digit image. The canvas size stays the same. 
    resized  = digit_img.resize((target_w, target_h), Image.BILINEAR)

    # paste centred on canvas
    offset_x = (CANVAS_W - target_w) // 2
    offset_y = (CANVAS_H - target_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas


# main function 
def generate(input_csv=INPUT_CSV, output_dir=OUTPUT_DIR, output_map=OUTPUT_MAP):
    os.makedirs(output_dir, exist_ok=True)
    digit_bank = load_mnist_by_digit(MNIST_ROOT)
    records    = []

    with open(input_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows):
        sid        = row['sentence_id']
        sentence   = row['sentence']
        tokens     = sentence.split() 
        size_label = tokens[0]
        colour     = tokens[1]
        number     = int(tokens[2])

        seed = int(sid.split('_')[1])        # deterministic per sentence
        img  = compose_image(number, colour, size_label, digit_bank, seed)

        fname = f'{sid}.png'
        img.save(os.path.join(output_dir, fname))
        records.append({'filename': fname, 'sentence_id': sid})

        if i % 1000 == 0:
            print(f"  {i}/{len(rows)} processed...")

    with open(output_map, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'sentence_id'])
        writer.writeheader()
        writer.writerows(records)

    print(f"[type-b] {len(records)} images saved → {output_dir}")
    print(f"  canvas: {CANVAS_W}×{CANVAS_H}px | 1 image per sentence (Direction B)")


if __name__ == '__main__':
    generate()