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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent.parent.parent.parent
_HERE     = Path(__file__).resolve().parent
_DATA_DIR = _ROOT / 'src' / 'data' / 'type-b'
_OUT_DIR  = _HERE / 'results'

# Add repo root and non-pretrained dir (hyphenated name can't be imported normally)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src' / 'embeddings' / 'non-pretrained'))


# ── Imports from src/embeddings/ ───────────────────────────────────────────────
from embeddings.pretrained.sbert_embeddings import SBERTEmbedder
from embeddings.pretrained.bert_mean_embeddings import BertMeanEmbedder
from embeddings.pretrained.bert_pooler_embeddings import BertPoolerEmbedder
from embeddings.pretrained.tinybert_mean_embeddings import TinyBertMeanEmbedder
from embeddings.pretrained.tinybert_pooler_embeddings import TinyBertPoolerEmbedder
from src.embeddings.pretrained.pretrained_word2vec_embeddings import PretrainedWord2VecEmbedder
from src.embeddings.tfidf_embeddings import TfidfEmbedder       # top-level: sparse TF-IDF weights
from word2vec_skipgram_embeddings import SkipGramEmbedder        # non-pretrained/
from tfidf_embeddings import TFIDFEmbedder                       # non-pretrained/: SVD-reduced


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


def save_embedding(method_name: str, sentences: list[str], emb: torch.Tensor) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f'{method_name}_embedding_result_typeb.pt'
    torch.save({
        'sentences':  sentences,
        'embeddings': emb,
        'method':     method_name,
        # 'dataset':    'b',
    }, out_path)
    print(f'[saved]  {out_path.name}  shape={tuple(emb.shape)}')


# ── Embedding methods ──────────────────────────────────────────────────────────

def compute_sbert(sentences: list[str]) -> None:
    """SBERTEmbedder — all-MiniLM-L6-v2, 384-dim."""
    if skip_if_exists('sbert'):
        return
    embedder = SBERTEmbedder(model_name='all-MiniLM-L6-v2')
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('sbert', sentences, emb)


def compute_bert_mean(sentences: list[str]) -> None:
    """BertMeanEmbedder — bert-base-uncased, mean pooling, 768-dim."""
    if skip_if_exists('bert_mean'):
        return
    embedder = BertMeanEmbedder()
    vecs = []
    for i, s in enumerate(sentences):
        vecs.append(embedder.get_embedding(s).squeeze(0))
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(sentences)}', end='\r')
    print()
    save_embedding('bert_mean', sentences, torch.stack(vecs))


def compute_bert_pooler(sentences: list[str]) -> None:
    """BertPoolerEmbedder — bert-base-uncased, pooler output, 768-dim."""
    if skip_if_exists('bert_pooler'):
        return
    embedder = BertPoolerEmbedder()
    vecs = []
    for i, s in enumerate(sentences):
        vecs.append(embedder.get_embedding(s).squeeze(0))
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(sentences)}', end='\r')
    print()
    save_embedding('bert_pooler', sentences, torch.stack(vecs))


def compute_tinybert_mean(sentences: list[str]) -> None:
    """TinyBertMeanEmbedder — TinyBERT_General_4L_312D, mean pooling, 312-dim."""
    if skip_if_exists('tinybert_mean'):
        return
    embedder = TinyBertMeanEmbedder()
    vecs = []
    for i, s in enumerate(sentences):
        vecs.append(embedder.get_embedding(s).squeeze(0))
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(sentences)}', end='\r')
    print()
    save_embedding('tinybert_mean', sentences, torch.stack(vecs))


def compute_tinybert_pooler(sentences: list[str]) -> None:
    """TinyBertPoolerEmbedder — TinyBERT_General_4L_312D, pooler output, 312-dim."""
    if skip_if_exists('tinybert_pooler'):
        return
    embedder = TinyBertPoolerEmbedder()
    vecs = []
    for i, s in enumerate(sentences):
        vecs.append(embedder.get_embedding(s).squeeze(0))
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(sentences)}', end='\r')
    print()
    save_embedding('tinybert_pooler', sentences, torch.stack(vecs))


def compute_word2vec_skipgram(sentences: list[str]) -> None:
    """SkipGramEmbedder — trained from scratch on type-b corpus, 100-dim."""
    if skip_if_exists('word2vec_skipgram'):
        return
    embedder = SkipGramEmbedder(vector_size=100)
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('word2vec_skipgram', sentences, emb)


def compute_word2vec_pretrained(sentences: list[str]) -> None:
    """PretrainedWord2VecEmbedder — Google News 300-dim."""
    if skip_if_exists('word2vec_pretrained'):
        return
    embedder = PretrainedWord2VecEmbedder(lowercase=True)
    embedder.oov_rate(sentences)
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('word2vec_pretrained', sentences, emb)


def compute_tfidf(sentences: list[str]) -> None:
    """TFIDFEmbedder (non-pretrained) — TF-IDF sparse → SVD → 100-dim."""
    if skip_if_exists('tfidf'):
        return
    embedder = TFIDFEmbedder(vector_size=100)
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('tfidf', sentences, emb)


def compute_tfidf_w2v(sentences: list[str]) -> None:
    """TfidfEmbedder (weights) × SkipGramEmbedder (vectors) — weighted mean, 100-dim.

    TfidfEmbedder provides per-token TF-IDF scores.
    SkipGramEmbedder provides Word2Vec vectors.
    Sentence vector = weighted average of token vectors.
    """
    if skip_if_exists('tfidf_w2v'):
        return

    # TF-IDF weights (top-level TfidfEmbedder — uses sklearn TfidfVectorizer internally)
    tfidf_embedder = TfidfEmbedder()
    tfidf_embedder.fit(sentences)
    vocab_to_idx = {w: i for i, w in enumerate(tfidf_embedder.vectorizer.get_feature_names_out())}
    tfidf_matrix = tfidf_embedder.vectorizer.transform(sentences)   # sparse (N, vocab)

    # Word2Vec vectors (SkipGramEmbedder)
    w2v = SkipGramEmbedder(vector_size=100)
    w2v.fit(sentences)

    vecs = []
    for i, s in enumerate(sentences):
        tokens     = s.lower().split()
        token_vecs = []
        weights    = []
        for t in tokens:
            if t in w2v.model.wv and t in vocab_to_idx:
                token_vecs.append(w2v.model.wv[t])
                weights.append(tfidf_matrix[i, vocab_to_idx[t]])
        if token_vecs and sum(weights) > 0:
            vecs.append(np.average(token_vecs, axis=0, weights=weights))
        elif token_vecs:
            vecs.append(np.mean(token_vecs, axis=0))
        else:
            vecs.append(np.zeros(w2v.vector_size))

    emb = torch.tensor(np.array(vecs), dtype=torch.float32)
    save_embedding('tfidf_w2v', sentences, emb)


# ── Entry point ────────────────────────────────────────────────────────────────

ALL_METHODS = [
    'sbert',
    'bert_mean', 'bert_pooler',
    'tinybert_mean', 'tinybert_pooler',
    'word2vec_skipgram', 'word2vec_pretrained',
    'tfidf', 'tfidf_w2v',
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
        'bert_mean':           lambda: compute_bert_mean(sentences),
        'bert_pooler':         lambda: compute_bert_pooler(sentences),
        'tinybert_mean':       lambda: compute_tinybert_mean(sentences),
        'tinybert_pooler':     lambda: compute_tinybert_pooler(sentences),
        'word2vec_skipgram':   lambda: compute_word2vec_skipgram(sentences),
        'word2vec_pretrained': lambda: compute_word2vec_pretrained(sentences),
        'tfidf':               lambda: compute_tfidf(sentences),
        'tfidf_w2v':           lambda: compute_tfidf_w2v(sentences),
    }

    for method in methods:
        print(f'\n── {method} ──')
        dispatch[method]()

    print(f'\nDone. Results in: {_OUT_DIR}')


if __name__ == '__main__':
    main()
