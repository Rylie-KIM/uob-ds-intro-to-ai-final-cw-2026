from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT     = Path(__file__).resolve().parent.parent.parent.parent.parent
_HERE     = Path(__file__).resolve().parent
_DATA_DIR = _ROOT / 'src' / 'data' / 'type-b'
_OUT_DIR  = _HERE / 'results'

# Add repo root + src/ so 'embeddings.*' and 'src.embeddings.*' both resolve.
# non-pretrained and fine-tune use hyphens so also add them directly for bare-name imports.
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'src' / 'embeddings' / 'non-pretrained'))
sys.path.insert(0, str(_ROOT / 'src' / 'embeddings' / 'pretrained' / 'fine-tune'))



# fine-tune/
from only_typeb_finetune_sbert import FinetunedSBERTEmbedder, finetune  
# pretrained  
from embeddings.pretrained.sbert_embeddings import SBERTEmbedder
from embeddings.pretrained.glove_embedding import GloVeEmbedder
from embeddings.pretrained.only_type_b_bert_mean_embeddings import BertMeanEmbedder
from embeddings.pretrained.only_type_b_bert_pooler_embeddings import BertPoolerEmbedder
from embeddings.pretrained.only_type_b_tinybert_mean_embeddings import TinyBertMeanEmbedder
from embeddings.pretrained.only_type_b_tinybert_pooler_embeddings import TinyBertPoolerEmbedder
from embeddings.pretrained.pretrained_word2vec_embeddings import PretrainedWord2VecEmbedder
# non pretrained 
from word2vec_skipgram_embeddings import SkipGramEmbedder                      
from tfidf_lsa_embeddings import TFIDFLSAEmbedder                                
from tfidf_weighted_word2vec_embeddings import TFIDFWeightedWord2VecEmbedder     


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


def _inspect(label: str, sentences: list[str], arr: np.ndarray) -> None:
    """Print sanity checks for an embedding array."""
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

    warn = lambda v: f'{v}  ⚠ WARNING' if v > 0 else f'{v}  OK'
    print(f'  [inspect/{label}] NaN={warn(n_nan)}  |  Inf={warn(n_inf)}  |  zero-rows={warn(zero_rows)}  |  consistency-mismatches={warn(mismatch)}')
    print(f'  [inspect/{label}] unique_sentences={n_unique_s}/{len(sentences)}'
          f'  |  norm mean={norms.mean():.3f}  std={norms.std():.3f}'
          f'  |  val mean={arr.mean():.4f}  std={arr.std():.4f}')
    for i in [0, len(sentences)//2, len(sentences)-1]:
        preview = ' '.join(f'{v:.3f}' for v in arr[i, :5])
        print(f'  [inspect/{label}] [{i:5d}] "{sentences[i]}"  →  [{preview} ...]')


def save_embedding(method_name: str, sentences: list[str], emb: torch.Tensor, elapsed: float = 0.0) -> None:
    # Save both original and L2-normalised embeddings to a single .pt file
    # embeddings            : raw float32 tensor  (shape N×D)
    # embeddings_normalised : L2-normalised float32 tensor (shape N×D, all norms ≈ 1)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f'{method_name}_embedding_result_typeb.pt'

    emb_raw  = emb.float()
    emb_norm = torch.nn.functional.normalize(emb_raw, p=2, dim=1)

    torch.save({
        'sentences':             sentences,
        'embeddings':            emb_raw,
        'embeddings_normalised': emb_norm,
        'method':                method_name,
        'dataset':               'b',
    }, out_path)

    print(f'[saved]  {out_path.name}  shape={tuple(emb_raw.shape)}  time={elapsed:.1f}s')
    _inspect('raw',  sentences, emb_raw.numpy())
    _inspect('norm', sentences, emb_norm.numpy())

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
    emb = torch.tensor(embedder.transform(sentences))
    save_embedding('bert_mean', sentences, emb, elapsed=time.time() - t0)


def compute_bert_pooler(sentences: list[str]) -> None:
    """BertPoolerEmbedder — bert-base-uncased, pooler output, 768-dim."""
    if skip_if_exists('bert_pooler'):
        return
    t0 = time.time()
    embedder = BertPoolerEmbedder()
    emb = torch.tensor(embedder.transform(sentences))
    save_embedding('bert_pooler', sentences, emb, elapsed=time.time() - t0)


def compute_tinybert_mean(sentences: list[str]) -> None:
    """TinyBertMeanEmbedder — TinyBERT_General_4L_312D, mean pooling, 312-dim."""
    if skip_if_exists('tinybert_mean'):
        return
    t0 = time.time()
    embedder = TinyBertMeanEmbedder()
    emb = torch.tensor(embedder.transform(sentences))
    save_embedding('tinybert_mean', sentences, emb, elapsed=time.time() - t0)


def compute_tinybert_pooler(sentences: list[str]) -> None:
    """TinyBertPoolerEmbedder — TinyBERT_General_4L_312D, pooler output, 312-dim."""
    if skip_if_exists('tinybert_pooler'):
        return
    t0 = time.time()
    embedder = TinyBertPoolerEmbedder()
    emb = torch.tensor(embedder.transform(sentences))
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


def compute_glove(sentences: list[str]) -> None:
    """GloVeEmbedder — glove-wiki-gigaword-100, mean pooling over tokens, 100-dim."""
    if skip_if_exists('glove'):
        return
    t0 = time.time()
    embedder = GloVeEmbedder(model_name='glove-wiki-gigaword-300')
    embedder.oov_rate(sentences)
    emb = torch.tensor(embedder.fit_transform(sentences))
    save_embedding('glove', sentences, emb, elapsed=time.time() - t0)


# ── Entry point ────────────────────────────────────────────────────────────────

ALL_METHODS = [
    'sbert', 'sbert_finetuned',
    'bert_mean', 'bert_pooler',
    'tinybert_mean', 'tinybert_pooler',
    'word2vec_skipgram', 'word2vec_pretrained',
    'glove',
    'tfidf_lsa', 'tfidf_w2v',
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
        'glove':               lambda: compute_glove(sentences),
        'tfidf_lsa':           lambda: compute_tfidf_lsa(sentences),
        'tfidf_w2v':           lambda: compute_tfidf_w2v(sentences),
    }

    for method in methods:
        print(f'\n── {method} ──')
        dispatch[method]()

    print(f'\nDone. Results in: {_OUT_DIR}')


if __name__ == '__main__':
    main()
