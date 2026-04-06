"""
src/embeddings/pretrained/finetune_sbert.py

Fine-tune SBERT (all-MiniLM-L6-v2) on sentence pairs for a given dataset type.

Why fine-tune? (text-only)
--------------------------
Default SBERT was trained on general NLI/STS data. Dataset-specific sentences
(e.g. "large red 1", "a big blue hexagon is below a small red circle") are
short, out-of-distribution phrases where SBERT may not distinguish
attribute differences well.

Fine-tuning on word-overlap similarity pairs teaches SBERT to:
  - Give similar embeddings to sentences sharing the same words/attributes
  - Give different embeddings to sentences with little overlap

This keeps text representation as a clean, isolated experimental axis:
    frozen SBERT  vs  fine-tuned SBERT  (CNN stays the same)

Similarity scoring (CosineSimilarityLoss):
  Jaccard word-overlap — works for all three dataset types:

  Type-A: "a big blue circle above a red square"
          vs "a big blue circle below a red square"  →  high overlap → ~0.75

  Type-B: "large red 1" vs "large red 2"   →  2/3 ≈ 0.67
          "large red 1" vs "small blue 2"  →  0/3 = 0.0

  Type-C: "X in top left, O in center"
          vs "X in top left, O in bottom right"  →  partial overlap

Usage
-----
# Fine-tune for one dataset type:
python src/embeddings/pretrained/finetune_sbert.py --dataset b
python src/embeddings/pretrained/finetune_sbert.py --dataset a
python src/embeddings/pretrained/finetune_sbert.py --dataset c

# In generate_embeddings_type_{a,b,c}.py:
from embeddings.pretrained.finetune_sbert import FinetunedSBERTEmbedder
embedder = FinetunedSBERTEmbedder(dataset='b')
emb = torch.tensor(embedder.fit_transform(sentences))
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import TypeAlias
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

# Type Aliases
Sentences:       TypeAlias = list[str]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent.parent.parent
_DATA_DIR = _ROOT / 'src' / 'data'
_CKPT_DIR = _ROOT / 'results' / 'checkpoints'

# ── Dataset configs ────────────────────────────────────────────────────────────
# All three types share the same CSV structure: sentence_id, sentence, ...
DATASET_CONFIGS = {
    'a': _DATA_DIR / 'type-a' / 'sentences_a.csv',
    'b': _DATA_DIR / 'type-b' / 'sentences_b.csv',
    'c': _DATA_DIR / 'type-c' / 'sentences_c.csv',
}


# ── Sentence loading ───────────────────────────────────────────────────────────

def _load_sentences(dataset: str) -> Sentences:
    csv_path  = DATASET_CONFIGS[dataset]
    df        = pd.read_csv(csv_path)
    sentences = df['sentence'].drop_duplicates().tolist()
    print(f"[finetune_sbert] dataset={dataset}  loaded {len(sentences)} unique sentences")
    return sentences


# ── Similarity scoring ─────────────────────────────────────────────────────────

def _jaccard_similarity(s1: str, s2: str) -> float:
    """
    Word-overlap (Jaccard) similarity between two sentences.
    Works for all dataset types without domain-specific attribute parsing.

      |words(s1) ∩ words(s2)|
      ───────────────────────  ∈ [0, 1]
      |words(s1) ∪ words(s2)|

    Exact matches (1.0) are excluded from training pairs.
    """
    w1  = set(s1.lower().split())
    w2  = set(s2.lower().split())
    intersection = w1 & w2
    union        = w1 | w2
    if not union:
        return 0.0
    return round(len(intersection) / len(union), 4)


# ── Training pair generation ───────────────────────────────────────────────────

def _make_training_pairs(
    sentences: Sentences,
    n_pairs:   int = 50_000,
    seed:      int = 42,
) -> list[InputExample]:
    """
    Sample random sentence pairs and label them with Jaccard similarity.
    Exact matches (sim=1.0) are excluded — no training signal there.
    """
    rng    = random.Random(seed)
    unique = list(set(sentences))
    pairs  = []

    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        s1, s2 = rng.sample(unique, 2)
        sim    = _jaccard_similarity(s1, s2)
        if sim < 1.0:
            pairs.append(InputExample(texts=[s1, s2], label=float(sim)))
        attempts += 1

    print(f"[finetune_sbert] generated {len(pairs)} training pairs")
    return pairs


# ── Fine-tuning ────────────────────────────────────────────────────────────────

def finetune(
    dataset:    str,
    base_model: str   = 'all-MiniLM-L6-v2',
    epochs:     int   = 3,
    batch_size: int   = 64,
    n_pairs:    int   = 50_000,
    seed:       int   = 42,
) -> SentenceTransformer:
    """
    Fine-tune SBERT on sentence pairs for the given dataset type.

    Args:
        dataset:    'a', 'b', or 'c'
        base_model: HuggingFace model name to start from
        epochs:     number of fine-tuning epochs
        batch_size: training batch size
        n_pairs:    number of (sentence1, sentence2, similarity) training pairs
        seed:       random seed for reproducibility

    Returns:
        fine-tuned SentenceTransformer model
    """
    save_path = _CKPT_DIR / f'sbert_finetuned_type{dataset}'

    if save_path.exists():
        print(f"[finetune_sbert] already exists, skipping → {save_path}")
        return SentenceTransformer(str(save_path))

    print(f"[finetune_sbert] base model : {base_model}")
    print(f"[finetune_sbert] epochs={epochs}  batch_size={batch_size}  n_pairs={n_pairs}")

    sentences  = _load_sentences(dataset)
    model      = SentenceTransformer(base_model)
    train_data = _make_training_pairs(sentences, n_pairs=n_pairs, seed=seed)
    loader     = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    loss_fn    = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=epochs,
        warmup_steps=int(len(loader) * 0.1),
        show_progress_bar=True,
        output_path=str(save_path),
    )

    print(f"[finetune_sbert] model saved → {save_path}")
    return model


# ── Inference embedder ─────────────────────────────────────────────────────────

class FinetunedSBERTEmbedder:
    """
    Loads the fine-tuned SBERT model for a given dataset type and provides
    the standard fit / transform / fit_transform interface.

    Used in generate_embeddings_type_{a,b,c}.py as a drop-in replacement
    for SBERTEmbedder.
    """

    def __init__(self, dataset: str):
        model_path = _CKPT_DIR / f'sbert_finetuned_type{dataset}'
        if not model_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found: {model_path}\n"
                f"Run: python src/embeddings/pretrained/finetune_sbert.py --dataset {dataset}"
            )
        print(f"FinetunedSBERTEmbedder loading from {model_path}")
        self.model = SentenceTransformer(str(model_path))
        print(f"FinetunedSBERTEmbedder ready  dim: {self.model.get_sentence_embedding_dimension()}")

    def fit(self, _sentences: Sentences) -> 'FinetunedSBERTEmbedder':
        return self

    def transform(self, sentences: Sentences) -> EmbeddingMatrix:
        return self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix:
        return self.fit(sentences).transform(sentences)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune SBERT on dataset-specific sentence pairs')
    parser.add_argument('--dataset',    required=True, choices=['a', 'b', 'c'])
    parser.add_argument('--base_model', default='all-MiniLM-L6-v2')
    parser.add_argument('--epochs',     type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_pairs',    type=int, default=50_000)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    finetune(
        dataset    = args.dataset,
        base_model = args.base_model,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        n_pairs    = args.n_pairs,
        seed       = args.seed,
    )
