# type-a: Shape Arrangements

Generates sentences and images for dataset type a.

Sentences describe two shapes with their size, colour, and spatial relation.
Example: `"a big blue circle is above a small red square"`

- `generate_sentences_a.py` — produces `sentences_a.csv`
- `generate_images_a.py` — reads `sentences_a.csv`, draws 128x128 PNG images using PIL, outputs `image_map_a.csv`
