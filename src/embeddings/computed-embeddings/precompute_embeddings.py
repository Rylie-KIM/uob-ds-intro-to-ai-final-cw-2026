"""
src/embeddings/precompute_embeddings.py
Phase 1: Pre-compute sentence embeddings and save to disk.

Run ONCE per (dataset, embedding) combination. Results are saved to
results/embeddings/type-{a,b,c}_{embedding}.pt and reused by all
subsequent training runs — no need to re-encode every time.

Usage
-----
# compute one combination
python src/embeddings/precompute_embeddings.py --dataset b --embedding sbert

# compute all 9 combinations (a/b/c × sbert/bert/tinybert)
python src/embeddings/precompute_embeddings.py

Saved file format
-----------------
torch.save({
    'sentences':   list[str],        # N sentences
    'embeddings':  Tensor(N, dim),   # float32, CPU
    'model_name':  str,
    'dim':         int,
    'dataset':     str,
}, path)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Config ─────────────────────────────────────────────────────────────────────

EMBEDDING_CONFIGS = {
    'sbert':    {'model_name': 'all-MiniLM-L6-v2',                       'dim': 384},
    'bert':     {'model_name': 'bert-base-uncased',                       'dim': 768},
    'tinybert': {'model_name': 'huawei-noah/TinyBERT_General_4L_312D',    'dim': 312},
}

# Sentence CSV paths per dataset type
SENTENCE_CSVS = {
    'a': _ROOT / 'src' / 'data' / 'type-a' / 'sentences_a.csv',
    'b': _ROOT / 'src' / 'data' / 'type-b' / 'sentences_b.csv',
    'c': _ROOT / 'src' / 'data' / 'type-c' / 'sentences_c.csv',
}

OUTPUT_DIR = _ROOT / 'results' / 'embeddings'


# ── Core function ──────────────────────────────────────────────────────────────

def precompute(dataset: str, embedding_name: str, device: str = 'cpu') -> None:
    """Encode all sentences for the given dataset and save to .pt file."""
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    csv_path = SENTENCE_CSVS[dataset]
    if not csv_path.exists():
        print(f"[precompute] SKIP — CSV not found: {csv_path}")
        return

    out_path = OUTPUT_DIR / f'type-{dataset}_{embedding_name}.pt'
    if out_path.exists():
        print(f"[precompute] SKIP — already exists: {out_path.name}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    sentences = df['sentence'].tolist()
    cfg = EMBEDDING_CONFIGS[embedding_name]

    print(f"[precompute] dataset={dataset}  embedding={embedding_name}  "
          f"model={cfg['model_name']}  n={len(sentences)}")

    if embedding_name in ('sbert',):
        # SentenceTransformer handles pooling internally
        model    = SentenceTransformer(cfg['model_name'], device=device)
        emb_tensor = model.encode(
            sentences, convert_to_tensor=True,
            show_progress_bar=True, device=device,
        ).cpu()                                          # (N, dim)

    else:
        # BERT / TinyBERT — use mean pooling over token embeddings
        tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], do_lower_case=True)
        bert      = AutoModel.from_pretrained(cfg['model_name']).to(device)
        bert.eval()

        embeddings = []
        batch_size = 64
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            enc   = tokenizer(batch, return_tensors='pt', padding=True,
                              truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = bert(**enc)
            # mean pool over token dimension, masked for padding
            mask   = enc['attention_mask'].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings.append(pooled.cpu())

        emb_tensor = torch.cat(embeddings, dim=0)       # (N, dim)

    torch.save({
        'sentences':  sentences,
        'embeddings': emb_tensor,
        'model_name': cfg['model_name'],
        'dim':        cfg['dim'],
        'dataset':    dataset,
    }, out_path)

    print(f"[precompute] saved → {out_path}  shape={tuple(emb_tensor.shape)}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   choices=['a', 'b', 'c'],           default=None,
                        help='Dataset type (default: all)')
    parser.add_argument('--embedding', choices=list(EMBEDDING_CONFIGS),   default=None,
                        help='Embedding model (default: all)')
    parser.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    datasets   = [args.dataset]   if args.dataset   else ['a', 'b', 'c']
    embeddings = [args.embedding] if args.embedding else list(EMBEDDING_CONFIGS)

    for d in datasets:
        for e in embeddings:
            precompute(d, e, device=args.device)

    print('\nDone. Files saved to:', OUTPUT_DIR)


if __name__ == '__main__':
    main()
