import numpy as np
import numpy.typing as npt
from typing import TypeAlias
import gensim.downloader

Sentences:       TypeAlias = list[str]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]

# Available gensim GloVe models:
#   'glove-wiki-gigaword-50'   →  50-dim
#   'glove-wiki-gigaword-100'  → 100-dim  (default)
#   'glove-wiki-gigaword-200'  → 200-dim
#   'glove-wiki-gigaword-300'  → 300-dim
#   'glove-twitter-25'         →  25-dim  (Twitter corpus)
#   'glove-twitter-100'        → 100-dim  (Twitter corpus)

class GloVeEmbedder:
    def __init__(
        self,
        model_name: str = 'glove-wiki-gigaword-100',
    ):
        self.model_name  = model_name
        print(f"GloVeEmbedder loading '{model_name}'")
        self.model       = gensim.downloader.load(model_name)
        self.vector_size = self.model.vector_size
        print(f"GloVeEmbedder ready  vocab size: {len(self.model)}  dim: {self.vector_size}")

    # Vectors are pretrained — no training needed
    def fit(self, _sentences: Sentences) -> 'GloVeEmbedder':
        return self

    def transform(self, sentences: Sentences) -> EmbeddingMatrix:
        sentence_embeddings = []
        for s in sentences:
            tokens = s.lower().split()
            found  = [self.model[t] for t in tokens if t in self.model]
            if not found:
                sentence_embeddings.append(np.zeros(self.vector_size, dtype=np.float32))
            else:
                sentence_embeddings.append(np.mean(found, axis=0).astype(np.float32))
        return np.array(sentence_embeddings, dtype=np.float32)

    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix:
        return self.fit(sentences).transform(sentences)

    def oov_rate(self, sentences: Sentences) -> float:
        """Returns fraction of tokens not found in the GloVe vocabulary."""
        total = oov = 0
        for s in sentences:
            tokens  = s.lower().split()
            total  += len(tokens)
            oov    += sum(1 for t in tokens if t not in self.model)
        rate = oov / total if total > 0 else 0.0
        print(f"OOV: {oov} / {total} tokens  ({rate * 100:.1f} %)")
        return rate
