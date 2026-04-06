"""
src/embeddings/computed-embeddings/type-b/generate_embeddings_type_b.py

Pre-compute all 9 text embeddings for the Type-B dataset.
Results saved to: src/embeddings/computed-embeddings/type-b/results/

Usage
-----
python src/embeddings/computed-embeddings/type-b/generate_embeddings_type_b.py
python src/embeddings/computed-embeddings/type-b/generate_embeddings_type_b.py --embedding bert_mean

Imports from
------------
src/embeddings/pretrained/
    sbert_embeddings.py                       → SBERTEmbedder
    bert_mean_embeddings.py                   → BertMeanEmbedder
    bert_pooler_embeddings.py                 → BertPoolerEmbedder
    tinybert_mean_embeddings.py               → TinyBertMeanEmbedder
    tinybert_pooler_embeddings.py             → TinyBertPoolerEmbedder
    pretrained_word2vec_embeddings.py         → PretrainedWord2VecEmbedder
src/embeddings/non-pretrained/
    word2vec_skipgram_embeddings.py           → SkipGramEmbedder
    tfidf_embeddings.py                       → TFIDFEmbedder
src/embeddings/tfidf_embeddings.py            → TfidfEmbedder  (for TF-IDF × W2V weights)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent.parent.parent.parent
_HERE     = Path(__file__).resolve().parent
_DATA_DIR = _ROOT / 'src' / 'data' / 'type-b'
_OUT_DIR  = _HERE / 'results'

# Add repo root + src/ so 'embeddings.*' and 'src.embeddings.*' both resolve.
# non-pretrained uses a hyphen so also add it directly for bare-name imports.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'src' / 'embeddings' / 'non-pretrained'))


# ── Imports from src/embeddings/ ───────────────────────────────────────────────
from embeddings.pretrained.sbert_embeddings import SBERTEmbedder
from embeddings.pretrained.finetune_sbert import FinetunedSBERTEmbedder, finetune
from embeddings.pretrained.bert_mean_embeddings import BertMeanEmbedder
from embeddings.pretrained.bert_pooler_embeddings import BertPoolerEmbedder
from embeddings.pretrained.tinybert_mean_embeddings import TinyBertMeanEmbedder
from embeddings.pretrained.tinybert_pooler_embeddings import TinyBertPoolerEmbedder
from embeddings.pretrained.pretrained_word2vec_embeddings import PretrainedWord2VecEmbedder
from word2vec_skipgram_embeddings import SkipGramEmbedder                        # non-pretrained/
from tfidf_embeddings import TFIDFEmbedder                                       # non-pretrained/
from tfidf_lsa_embeddings import TFIDFLSAEmbedder                                # non-pretrained/
from tfidf_weighted_word2vec_embeddings import TFIDFWeightedWord2VecEmbedder     # non-pretrained/


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_sentences() -> list[str]:
    image_map    = pd.read_csv(_DATA_DIR / 'image_map_b.csv')
    sentences_df = pd.read_csv(_DATA_DIR / 'sentences_b.csv')
    df           = image_map.merge(sentences_df, on='sentence_id')
    sentences    = df['sentence'].drop_duplicates().tolist()
    print(f'[loaded] {len(sentences)} unique sentences from type-b')
    return sentences


def skip_if_exists(method_name: str) -> bool:
    out_path = _OUT_DIR / f'{method_name}_embedding_result_typeb.pt'
    if out_path.exists():
        print(f'[skip]   {out_path.name} already exists')
        return True
    return False


def _inspect(method_name: str, sentences: list[str], emb: torch.Tensor) -> None:
    """Print sanity checks immediately after an embedding is computed."""
    arr = emb.numpy() if isinstance(emb, torch.Tensor) else emb
    n_nan      = int(np.isnan(arr).sum())
    n_inf      = int(np.isinf(arr).sum())
    zero_rows  = int((arr == 0).all(axis=1).sum())
    norms      = np.linalg.norm(arr, axis=1)
    n_unique_s = len(set(sentences))

    # same sentence → same vector check (sample first 1000 for speed)
    seen: dict[str, int] = {}
    mismatch = 0
    for idx, s in enumerate(sentences[:1000]):
        if s in seen:
            if not np.allclose(arr[seen[s]], arr[idx], atol=1e-5):
                mismatch += 1
        else:
            seen[s] = idx

    warn = lambda v, label: f'{v}  ⚠ WARNING' if v > 0 else f'{v}  OK'
    print(f'  [inspect] NaN={warn(n_nan, "nan")}  |  Inf={warn(n_inf, "inf")}  |  zero-rows={warn(zero_rows, "zero")}  |  consistency-mismatches={warn(mismatch, "mismatch")}')
    print(f'  [inspect] unique_sentences={n_unique_s}/{len(sentences)}'
          f'  |  norm mean={norms.mean():.3f}  std={norms.std():.3f}'
          f'  |  val mean={arr.mean():.4f}  std={arr.std():.4f}')
    # 3 sample sentences
    for i in [0, len(sentences)//2, len(sentences)-1]:
        preview = ' '.join(f'{v:.3f}' for v in arr[i, :5])
        print(f'  [inspect] [{i:5d}] "{sentences[i]}"  →  [{preview} ...]')


def save_embedding(method_name: str, sentences: list[str], emb: torch.Tensor, elapsed: float = 0.0) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f'{method_name}_embedding_result_typeb.pt'
    torch.save({
        'sentences':  sentences,
        'embeddings': emb,
        'method':     method_name,
        'dataset':    'b',
    }, out_path)
    print(f'[saved]  {out_path.name}  shape={tuple(emb.shape)}  time={elapsed:.1f}s')
    _inspect(method_name, sentences, emb)


# ── Embedding methods ──────────────────────────────────────────────────────────

def _bert_loop(embedder, sentences: list[str], label: str) -> torch.Tensor:
    """Encode sentences one-by-one with progress reporting every 100 steps."""
    n    = len(sentences)
    vecs = []
    t0   = time.time()
    for i, s in enumerate(sentences):
        vecs.append(embedder.get_embedding(s).squeeze(0).cpu().detach())
        if (i + 1) % 100 == 0 or (i + 1) == n:
            pct     = (i + 1) / n * 100
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f'  [{label}] {i+1}/{n}  ({pct:.0f}%)  elapsed={elapsed:.0f}s  eta={eta:.0f}s', end='\r')
    print()
    return torch.stack(vecs)


def compute_sbert(sentences: list[str]) -> None:
    """SBERTEmbedder — all-MiniLM-L6-v2, 384-dim."""
    if skip_if_exists('sbert'):
        return
    t0 = time.time()
    embedder = SBERTEmbedder(model_name='all-MiniLM-L6-v2')
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('sbert', sentences, emb, elapsed=time.time() - t0)


def compute_sbert_finetuned(sentences: list[str]) -> None:
    """FinetunedSBERTEmbedder — all-MiniLM-L6-v2 fine-tuned on Type-B sentence pairs, 384-dim."""
    if skip_if_exists('sbert_finetuned'):
        return
    t0 = time.time()
    finetune(dataset='b')           # no-op if model already saved
    embedder = FinetunedSBERTEmbedder(dataset='b')
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('sbert_finetuned', sentences, emb, elapsed=time.time() - t0)


def compute_bert_mean(sentences: list[str]) -> None:
    """BertMeanEmbedder — bert-base-uncased, mean pooling, 768-dim."""
    if skip_if_exists('bert_mean'):
        return
    t0 = time.time()
    embedder = BertMeanEmbedder()
    emb = _bert_loop(embedder, sentences, 'bert_mean')
    save_embedding('bert_mean', sentences, emb, elapsed=time.time() - t0)


def compute_bert_pooler(sentences: list[str]) -> None:
    """BertPoolerEmbedder — bert-base-uncased, pooler output, 768-dim."""
    if skip_if_exists('bert_pooler'):
        return
    t0 = time.time()
    embedder = BertPoolerEmbedder()
    emb = _bert_loop(embedder, sentences, 'bert_pooler')
    save_embedding('bert_pooler', sentences, emb, elapsed=time.time() - t0)


def compute_tinybert_mean(sentences: list[str]) -> None:
    """TinyBertMeanEmbedder — TinyBERT_General_4L_312D, mean pooling, 312-dim."""
    if skip_if_exists('tinybert_mean'):
        return
    t0 = time.time()
    embedder = TinyBertMeanEmbedder()
    emb = _bert_loop(embedder, sentences, 'tinybert_mean')
    save_embedding('tinybert_mean', sentences, emb, elapsed=time.time() - t0)


def compute_tinybert_pooler(sentences: list[str]) -> None:
    """TinyBertPoolerEmbedder — TinyBERT_General_4L_312D, pooler output, 312-dim."""
    if skip_if_exists('tinybert_pooler'):
        return
    t0 = time.time()
    embedder = TinyBertPoolerEmbedder()
    emb = _bert_loop(embedder, sentences, 'tinybert_pooler')
    save_embedding('tinybert_pooler', sentences, emb, elapsed=time.time() - t0)


def compute_word2vec_skipgram(sentences: list[str]) -> None:
    """SkipGramEmbedder — trained from scratch on type-b corpus, 100-dim."""
    if skip_if_exists('word2vec_skipgram'):
        return
    t0 = time.time()
    embedder = SkipGramEmbedder(vector_size=100)
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('word2vec_skipgram', sentences, emb, elapsed=time.time() - t0)


def compute_word2vec_pretrained(sentences: list[str]) -> None:
    """PretrainedWord2VecEmbedder — Google News 300-dim."""
    if skip_if_exists('word2vec_pretrained'):
        return
    t0 = time.time()
    embedder = PretrainedWord2VecEmbedder(lowercase=True)
    embedder.oov_rate(sentences)
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('word2vec_pretrained', sentences, emb, elapsed=time.time() - t0)


def compute_tfidf(sentences: list[str]) -> None:
    """TFIDFEmbedder (non-pretrained) — TF-IDF sparse → 100-dim."""
    if skip_if_exists('tfidf'):
        return
    t0 = time.time()
    embedder = TFIDFEmbedder(vector_size=100)
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('tfidf', sentences, emb, elapsed=time.time() - t0)


def compute_tfidf_w2v(sentences: list[str]) -> None:
    """TFIDFWeightedWord2VecEmbedder — TF-IDF weighted Word2Vec (skip-gram), 100-dim."""
    if skip_if_exists('tfidf_w2v'):
        return
    t0 = time.time()
    embedder = TFIDFWeightedWord2VecEmbedder(vector_size=100)
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('tfidf_w2v', sentences, emb, elapsed=time.time() - t0)


def compute_tfidf_lsa(sentences: list[str]) -> None:
    """TFIDFLSAEmbedder — full vocab TF-IDF + TruncatedSVD (LSA), 100-dim.
    Handles digit-token corpora (e.g. '342 small red') without vocabulary truncation.
    """
    if skip_if_exists('tfidf_lsa'):
        return
    t0 = time.time()
    embedder = TFIDFLSAEmbedder(n_components=100)
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('tfidf_lsa', sentences, emb, elapsed=time.time() - t0)


# ── Entry point ────────────────────────────────────────────────────────────────

ALL_METHODS = [
    'sbert', 'sbert_finetuned',
    'bert_mean', 'bert_pooler',
    'tinybert_mean', 'tinybert_pooler',
    'word2vec_skipgram', 'word2vec_pretrained',
    'tfidf', 'tfidf_lsa', 'tfidf_w2v',
]

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Type-B embeddings')
    parser.add_argument('--embedding', choices=ALL_METHODS, default=None,
                        help='Embedding method to run (default: all)')
    args = parser.parse_args()

    sentences = load_sentences()
    methods   = [args.embedding] if args.embedding else ALL_METHODS

    dispatch = {
        'sbert':               lambda: compute_sbert(sentences),
        'sbert_finetuned':     lambda: compute_sbert_finetuned(sentences),
        'bert_mean':           lambda: compute_bert_mean(sentences),
        'bert_pooler':         lambda: compute_bert_pooler(sentences),
        'tinybert_mean':       lambda: compute_tinybert_mean(sentences),
        'tinybert_pooler':     lambda: compute_tinybert_pooler(sentences),
        'word2vec_skipgram':   lambda: compute_word2vec_skipgram(sentences),
        'word2vec_pretrained': lambda: compute_word2vec_pretrained(sentences),
        'tfidf':               lambda: compute_tfidf(sentences),
        'tfidf_lsa':           lambda: compute_tfidf_lsa(sentences),
        'tfidf_w2v':           lambda: compute_tfidf_w2v(sentences),
    }

    for method in methods:
        print(f'\n── {method} ──')
        dispatch[method]()

    print(f'\nDone. Results in: {_OUT_DIR}')


if __name__ == '__main__':
    main()
