
# TF-IDF + Truncated SVD (Latent Semantic Analysis) sentence embedder.

# This embedder uses the full vocabulary (no max_features cap), then compresses the sparse TF-IDF matrix to a fixed-size dense vector via TruncatedSVD (LSA).

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
