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

_ROOT     = Path(__file__).resolve().parent.parent.parent.parent.parent
_DATA_DIR = _ROOT / 'src' / 'data'
_CKPT_DIR = _ROOT / 'src' / 'pipelines' /'results' / 'checkpoints'

# All three types share the same CSV structure
DATASET_CONFIGS = {
    'b': _DATA_DIR / 'type-b' / 'sentences_b.csv',
}


def _load_sentences(dataset: str) -> Sentences:
    csv_path  = DATASET_CONFIGS[dataset]
    df        = pd.read_csv(csv_path)
    sentences = df['sentence'].drop_duplicates().tolist()
    print(f"[finetune_sbert] dataset={dataset}  loaded {len(sentences)} unique sentences")
    return sentences


# similarity scoring (use average percentage per token)
def _jaccard_similarity(s1: str, s2: str) -> float:
    w1  = set(s1.lower().split())
    w2  = set(s2.lower().split())
    intersection = w1 & w2
    union        = w1 | w2 # total number of tokens (no duplicated counts)
    if not union:
        return 0.0
    return round(len(intersection) / len(union), 4)

# random sentence pair generation and label with the Jaccard similarity 
def _make_training_pairs(
    sentences: Sentences,
    n_pairs:   int = 50_000,
    seed:      int = 42,
) -> list[InputExample]:
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

def finetune(
    dataset:    str,
    base_model: str   = 'all-MiniLM-L6-v2',
    epochs:     int   = 3,
    batch_size: int   = 64,
    n_pairs:    int   = 50_000,
    seed:       int   = 42,
) -> SentenceTransformer:
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

class FinetunedSBERTEmbedder:
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
