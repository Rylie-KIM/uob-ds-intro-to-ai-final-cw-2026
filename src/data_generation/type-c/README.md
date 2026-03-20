# type-c: Tic-Tac-Toe Boards

Generates sentences and images for dataset type c.

Sentences describe a board state using position names.
Example: `"X in the center, O in the top-left"`

- `generate_sentences_c.py` — enumerates legal board states up to 4 moves, produces `sentences_c.csv`
- `generate_images_c.py` — reads `sentences_c.csv`, draws board grids with X and O using PIL, outputs `image_map_c.csv`
