"""
src/pipelines/data_loaders/type-b/type_b_loader.py
PyTorch Dataset and data-split utilities for the Type-B (coloured MNIST) dataset.

Data structure
--------------
  src/data/type-b/sentences_b.csv  : sentence_id, sentence, n_digits
  src/data/type-b/image_map_b.csv  : filename, sentence_id
  src/data/images/type-b/          : b_0.png … b_10007.png  (128x128 RGB)

Mapping chain (1-to-1, no duplicates):
  filename  ->  sentence_id  ->  sentence  ->  pre-computed embedding vector

Usage
-----
    # loaded via importlib (hyphen in path prevents direct import)

    train_set, val_set, test_set = make_splits(
        embedding_cache=Path('src/embeddings/.../sbert_embedding_result_typeb.pt'),
        device='cpu',
        seed=42,
    )

    # Each __getitem__ returns: (image_tensor, sentence_str, embedding_tensor)
    img, sentence, emb = train_set[0]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset
from torchvision import transforms

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import (
    TYPE_B_SENTENCES,
    TYPE_B_IMAGE_MAP,
    TYPE_B_IMAGES,
    TYPE_B_SPLITS,
)

# ── Default image transform ────────────────────────────────────────────────────
_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class TypeBDataset(Dataset):
    """
    PyTorch Dataset for Type-B (coloured MNIST numbers).

    Each item returns a triple:
        image_tensor : FloatTensor (3, 128, 128)  — normalised RGB image
        sentence     : str                         — ground-truth description
        embedding    : FloatTensor (dim,)          — pre-computed text embedding

    Parameters
    ----------
    records : list of (image_path, sentence, embedding_tensor)
    transform : torchvision transform applied to each PIL image
    """

    def __init__(
        self,
        records: list[tuple[Path, str, torch.Tensor]],
        transform=None,
    ) -> None:
        self.records   = records
        self.transform = transform or _DEFAULT_TRANSFORM

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, torch.Tensor]:
        img_path, sentence, emb = self.records[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, sentence, emb


# ══════════════════════════════════════════════════════════════════════════════
# Split helper
# ══════════════════════════════════════════════════════════════════════════════

def make_splits(
    embedding_cache: Path,
    device: str = 'cpu',
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    transform=None,
    save_split_csv: bool = True,
    load_split_csv: bool = True,
) -> tuple[Subset, Subset, Subset]:
    """
    Build stratified train / val / test splits for Type-B.

    Stratification key: n_digits (1–6) — ensures each digit-length group
    is proportionally represented in every split.

    Split indices are saved to / loaded from
        TYPE_B_SPLITS / type_b_splits_seed{seed}.csv
    so that all model × embedding experiments share the identical partition.

    Parameters
    ----------
    embedding_cache : path to a .pt file produced by generate_embeddings_type_b.py
                      Must contain keys 'sentences' (list[str]) and
                      'embeddings' (Tensor[N, dim]).
    device          : 'cpu' | 'cuda' | 'mps'  — target device for embedding tensors
    seed            : random seed (must be the same across all runs)
    train_ratio     : fraction of data used for training   (default 0.80)
    val_ratio       : fraction of data used for validation (default 0.10)
                      test_ratio = 1 - train_ratio - val_ratio
    transform       : optional torchvision transform; falls back to default (Resize+Normalize)
    save_split_csv  : persist split indices to CSV for reproducibility
    load_split_csv  : if CSV already exists, load indices from it (skip re-splitting)

    Returns
    -------
    (train_subset, val_subset, test_subset)  — all subsets of a single TypeBDataset
    """
    # ── Load embedding cache ───────────────────────────────────────────────────
    cache = torch.load(embedding_cache, map_location='cpu')
    emb_sentences: list[str]    = cache['sentences']
    emb_matrix:    torch.Tensor = cache['embeddings'].float()

    sentence_to_emb: dict[str, torch.Tensor] = {
        s: emb_matrix[i] for i, s in enumerate(emb_sentences)
    }

    # ── Load CSV data ──────────────────────────────────────────────────────────
    image_map  = pd.read_csv(TYPE_B_IMAGE_MAP)    # filename, sentence_id
    sentences  = pd.read_csv(TYPE_B_SENTENCES)    # sentence_id, sentence, n_digits
    df = image_map.merge(sentences, on='sentence_id')
    # df columns: filename, sentence_id, sentence, n_digits

    # Build record list: (image_path, sentence, embedding)
    records: list[tuple[Path, str, torch.Tensor]] = []
    for _, row in df.iterrows():
        img_path = TYPE_B_IMAGES / row['filename']
        sentence = row['sentence']
        emb      = sentence_to_emb.get(sentence)
        if emb is None:
            raise KeyError(
                f"Sentence not found in embedding cache: '{sentence}'\n"
                f"Re-run: python src/embeddings/computed-embeddings/type-b/generate_embeddings_type_b.py"
            )
        records.append((img_path, sentence, emb.to(device)))

    full_dataset = TypeBDataset(records, transform=transform)
    n = len(full_dataset)

    # ── Load or compute split indices ──────────────────────────────────────────
    split_csv = TYPE_B_SPLITS / f'type_b_splits_seed{seed}.csv'

    if load_split_csv and split_csv.exists():
        split_df   = pd.read_csv(split_csv)
        train_idx  = split_df[split_df['split'] == 'train']['idx'].tolist()
        val_idx    = split_df[split_df['split'] == 'val']['idx'].tolist()
        test_idx   = split_df[split_df['split'] == 'test']['idx'].tolist()
        print(f'[type_b_loader] Loaded split from {split_csv.name}  '
              f'train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')
    else:
        train_idx, val_idx, test_idx = _stratified_split(
            df=df, n=n, seed=seed,
            train_ratio=train_ratio, val_ratio=val_ratio,
        )
        if save_split_csv:
            _save_split_csv(train_idx, val_idx, test_idx, split_csv)
            print(f'[type_b_loader] Split saved to {split_csv}')
        print(f'[type_b_loader] New split  '
              f'train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}')

    return (
        Subset(full_dataset, train_idx),
        Subset(full_dataset, val_idx),
        Subset(full_dataset, test_idx),
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _stratified_split(
    df: pd.DataFrame,
    n: int,
    seed: int,
    train_ratio: float,
    val_ratio:   float,
) -> tuple[list[int], list[int], list[int]]:
    """
    Stratified split using n_digits as the stratification label.

    Strategy:
      1. First split: separate test set  (size = 1 - train_ratio - val_ratio)
      2. Second split: split remaining into train + val
    """
    all_idx       = list(range(n))
    n_digits_arr  = df['n_digits'].astype(str).values   # stratification labels

    test_ratio  = 1.0 - train_ratio - val_ratio
    val_of_rest = val_ratio / (train_ratio + val_ratio)  # val fraction of non-test

    # Step 1: separate test set
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    rest_idx, test_idx = next(sss1.split(all_idx, n_digits_arr))

    # Step 2: split rest into train + val
    rest_labels = n_digits_arr[rest_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_of_rest, random_state=seed)
    train_within_rest, val_within_rest = next(sss2.split(rest_idx, rest_labels))

    train_idx = rest_idx[train_within_rest].tolist()
    val_idx   = rest_idx[val_within_rest].tolist()
    test_idx  = test_idx.tolist()

    return train_idx, val_idx, test_idx


def _save_split_csv(
    train_idx: list[int],
    val_idx:   list[int],
    test_idx:  list[int],
    path:      Path,
) -> None:
    TYPE_B_SPLITS.mkdir(parents=True, exist_ok=True)
    rows = (
        [{'idx': i, 'split': 'train'} for i in train_idx]
        + [{'idx': i, 'split': 'val'} for i in val_idx]
        + [{'idx': i, 'split': 'test'} for i in test_idx]
    )
    pd.DataFrame(rows).to_csv(path, index=False)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Type-B data loader sanity check')
    parser.add_argument('--embedding', default='sbert',
                        help='Embedding name (default: sbert)')
    args = parser.parse_args()

    from src.config.paths import EMBED_RESULTS_B
    cache = EMBED_RESULTS_B / f'{args.embedding}_embedding_result_typeb.pt'

    if not cache.exists():
        print(f'[error] Embedding cache not found: {cache}')
        print('Run: python src/embeddings/computed-embeddings/type-b/generate_embeddings_type_b.py')
        raise SystemExit(1)

    train_set, val_set, test_set = make_splits(cache)

    print(f'\nDataset sizes:')
    print(f'  train : {len(train_set)}')
    print(f'  val   : {len(val_set)}')
    print(f'  test  : {len(test_set)}')
    print(f'  total : {len(train_set) + len(val_set) + len(test_set)}')

    img, sentence, emb = train_set[0]
    print(f'\nSample item:')
    print(f'  image shape : {img.shape}')
    print(f'  sentence    : "{sentence}"')
    print(f'  embedding   : shape={emb.shape}  dtype={emb.dtype}')
