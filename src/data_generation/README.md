# data_generation

Phase 1 — generates all training data (sentences + images) for the three dataset types.

Each dataset type has its own subfolder. Run sentence generators first, then image generators. After all images are ready, run `make_splits.py` to produce train/val/test splits before anyone starts Phase 2.

Subfolders:
- `type-a/` — shape arrangements (PIL)
- `type-b/` — multi-digit numbers (MNIST)
- `type-c/` — tic-tac-toe boards (PIL)
