# type-b: Multi-Digit Numbers (MNIST)

Generates sentences and images for dataset type b.

Sentences describe a number with size and colour.
Example: `"large blue 167"`, `"small red 42"`

- `generate_sentences_b.py` — produces `sentences_b.csv` (covers 1–4 digit numbers, balanced sampling)
- `generate_images_b.py` — reads `sentences_b.csv`, composes MNIST digit images side-by-side with colour tinting, pads to fixed 128x64 canvas, outputs `image_map_b.csv`

Requires MNIST to be downloaded (handled automatically via torchvision on first run).
