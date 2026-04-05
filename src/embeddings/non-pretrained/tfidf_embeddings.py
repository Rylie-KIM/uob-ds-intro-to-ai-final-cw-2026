import numpy as np
import numpy.typing as npt
from typing import TypeAlias
from sklearn.feature_extraction.text import TfidfVectorizer

# Type Aliases
Sentences:       TypeAlias = list[str]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class TFIDFEmbedder:
    def __init__(
            self,
            vector_size: int = 100,  # max number of TF-IDF features (vocabulary size cap)
    ):
        self.vector_size = vector_size
        self.vectorizer  = TfidfVectorizer(
            max_features=vector_size,
            lowercase=True,
        )

    def fit(self, sentences: Sentences) -> 'TFIDFEmbedder':
        self.vectorizer.fit(sentences)
        print(f"TFIDFEmbedder fitted. vocab size: {len(self.vectorizer.vocabulary_)}  dim: {self.vector_size}")
        return self

    def transform(self, sentences: Sentences) -> EmbeddingMatrix:
        return self.vectorizer.transform(sentences).toarray().astype(np.float32)

    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix:
        result = self.vectorizer.fit_transform(sentences).toarray().astype(np.float32)
        print(f"TFIDFEmbedder fitted. vocab size: {len(self.vectorizer.vocabulary_)}  dim: {self.vector_size}")
        return result
