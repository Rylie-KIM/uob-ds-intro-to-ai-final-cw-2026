"""
src/embeddings/non-pretrained/tfidf_lsa_embeddings.py

TF-IDF + Truncated SVD (Latent Semantic Analysis) sentence embedder.

Motivation
----------
For corpora where tokens are digit strings (e.g. "342 small red"), the raw
TF-IDF vocabulary can exceed 1000 unique tokens.  Using max_features=100
would discard the vast majority of number tokens, making sentences with
different numbers indistinguishable.

This embedder uses the full vocabulary (no max_features cap), then compresses
the sparse TF-IDF matrix to a fixed-size dense vector via TruncatedSVD (LSA).

Pipeline
--------
1. TfidfVectorizer(max_features=None)  →  sparse (N, vocab_size)
2. TruncatedSVD(n_components=dim)      →  dense  (N, dim)   float32

Interface
---------
    embedder = TFIDFLSAEmbedder(n_components=100)
    embedder.fit(train_sentences)
    matrix = embedder.transform(sentences)   # np.ndarray (N, 100) float32

    # or in one call:
    matrix = embedder.fit_transform(sentences)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import TypeAlias

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

Sentences:       TypeAlias = list[str]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class TFIDFLSAEmbedder:
    """TF-IDF sparse matrix compressed to dense vectors via Truncated SVD (LSA).

    Parameters
    ----------
    n_components : int
        Output embedding dimension after SVD compression. Default 100.
    random_state : int
        Random seed for TruncatedSVD reproducibility.
    """

    def __init__(self, n_components: int = 100, random_state: int = 42) -> None:
        self.n_components  = n_components
        self.random_state  = random_state
        self._pipeline     = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                max_features=None,   # use full vocabulary — critical for digit tokens
            )),
            ('svd', TruncatedSVD(
                n_components=n_components,
                random_state=random_state,
            )),
        ])
        self._fitted = False

    def fit(self, sentences: Sentences) -> 'TFIDFLSAEmbedder':
        self._pipeline.fit(sentences)
        vocab_size = len(self._pipeline['tfidf'].vocabulary_)
        print(
            f'[TFIDFLSAEmbedder] vocab={vocab_size}  '
            f'svd_components={self.n_components}  '
            f'explained_var={self._pipeline["svd"].explained_variance_ratio_.sum():.3f}'
        )
        self._fitted = True
        return self

    def transform(self, sentences: Sentences) -> EmbeddingMatrix:
        if not self._fitted:
            raise RuntimeError('Call fit() before transform().')
        return self._pipeline.transform(sentences).astype(np.float32)

    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix:
        result = self._pipeline.fit_transform(sentences).astype(np.float32)
        vocab_size = len(self._pipeline['tfidf'].vocabulary_)
        print(
            f'[TFIDFLSAEmbedder] vocab={vocab_size}  '
            f'svd_components={self.n_components}  '
            f'explained_var={self._pipeline["svd"].explained_variance_ratio_.sum():.3f}'
        )
        self._fitted = True
        return result
