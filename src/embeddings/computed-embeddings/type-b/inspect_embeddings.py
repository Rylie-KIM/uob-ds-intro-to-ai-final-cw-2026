"""
src/embeddings/computed-embeddings/type-b/inspect_embeddings.py

Quick debug tool to inspect pre-computed .pt embedding files.

Usage
-----
# Inspect a specific embedding
python src/embeddings/computed-embeddings/type-b/inspect_embeddings.py --embedding tfidf_lsa

# Inspect all available .pt files
python src/embeddings/computed-embeddings/type-b/inspect_embeddings.py --all

# Show more sample sentences
python src/embeddings/computed-embeddings/type-b/inspect_embeddings.py --embedding sbert --samples 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

_HERE    = Path(__file__).resolve().parent
_OUT_DIR = _HERE / 'results'


def inspect(pt_path: Path, n_samples: int = 5) -> None:
    print(f'\n{"="*60}')
    print(f'File   : {pt_path.name}')

    data       = torch.load(pt_path, map_location='cpu', weights_only=False)
    sentences  = data['sentences']
    embeddings = data['embeddings']
    method     = data.get('method', 'unknown')
    dataset    = data.get('dataset', 'unknown')

    emb = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

    print(f'Method : {method}   Dataset: {dataset}')
    print(f'Shape  : {emb.shape}   dtype: {emb.dtype}')
    print(f'N sentences: {len(sentences)}')

    # ── Sanity checks ─────────────────────────────────────────────────────────
    n_nan = np.isnan(emb).sum()
    n_inf = np.isinf(emb).sum()
    n_zero_rows = (emb == 0).all(axis=1).sum()
    print(f'\n[Sanity]')
    print(f'  NaN values    : {n_nan}  {"⚠ WARNING" if n_nan > 0 else "OK"}')
    print(f'  Inf values    : {n_inf}  {"⚠ WARNING" if n_inf > 0 else "OK"}')
    print(f'  All-zero rows : {n_zero_rows}  {"⚠ WARNING" if n_zero_rows > 0 else "OK"}')

    # ── Statistics ────────────────────────────────────────────────────────────
    print(f'\n[Statistics]')
    print(f'  mean  : {emb.mean():.4f}')
    print(f'  std   : {emb.std():.4f}')
    print(f'  min   : {emb.min():.4f}')
    print(f'  max   : {emb.max():.4f}')
    norms = np.linalg.norm(emb, axis=1)
    print(f'  L2 norm — mean: {norms.mean():.4f}  std: {norms.std():.4f}  min: {norms.min():.4f}  max: {norms.max():.4f}')

    # ── Duplicate check ───────────────────────────────────────────────────────
    n_unique_sentences  = len(set(sentences))
    n_unique_embeddings = len({tuple(row) for row in emb[:500]})  # sample 500 for speed
    print(f'\n[Uniqueness]')
    print(f'  unique sentences       : {n_unique_sentences} / {len(sentences)}')
    print(f'  unique embeddings (500 sample): {n_unique_embeddings} / min(500, {len(sentences)})')

    # ── Sample sentences + their embeddings ───────────────────────────────────
    print(f'\n[Sample sentences (first {n_samples})]')
    for i in range(min(n_samples, len(sentences))):
        vec_preview = ' '.join(f'{v:.3f}' for v in emb[i, :5])
        print(f'  [{i:4d}] "{sentences[i]}"')
        print(f'         vec[:5] = [{vec_preview} ...]')

    # ── Sentence-embedding alignment spot check ────────────────────────────────
    # Verify that two identical sentences map to the same embedding
    seen: dict[str, int] = {}
    mismatch = 0
    for idx, s in enumerate(sentences):
        if s in seen:
            prev = seen[s]
            if not np.allclose(emb[prev], emb[idx], atol=1e-5):
                mismatch += 1
        else:
            seen[s] = idx
    print(f'\n[Consistency] Same sentence → same embedding mismatches: {mismatch}'
          f'  {"⚠ WARNING" if mismatch > 0 else "OK"}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect Type-B embedding .pt files')
    parser.add_argument('--embedding', default=None,
                        help='Embedding method name (e.g. tfidf_lsa, sbert)')
    parser.add_argument('--all', action='store_true',
                        help='Inspect all .pt files in results/')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of sample sentences to show (default: 5)')
    args = parser.parse_args()

    if args.all:
        files = sorted(_OUT_DIR.glob('*_embedding_result_typeb.pt'))
        if not files:
            print(f'No .pt files found in {_OUT_DIR}')
            return
        for f in files:
            inspect(f, n_samples=args.samples)
    elif args.embedding:
        pt_path = _OUT_DIR / f'{args.embedding}_embedding_result_typeb.pt'
        if not pt_path.exists():
            print(f'File not found: {pt_path}')
            return
        inspect(pt_path, n_samples=args.samples)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
