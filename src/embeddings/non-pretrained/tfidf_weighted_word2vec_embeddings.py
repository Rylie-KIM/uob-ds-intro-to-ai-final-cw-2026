import numpy as np
import numpy.typing as npt
from typing import TypeAlias

from tfidf_embeddings import TFIDFEmbedder
from word2vec_skipgram_embeddings import SkipGramEmbedder

# Type Aliases
Sentences:       TypeAlias = list[str]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class TFIDFWeightedWord2VecEmbedder:
    """
    Sentence embedding = TF-IDF weighted mean of Word2Vec (skip-gram) word vectors.

    For each sentence:
      - token weight  : TF-IDF score from the corpus vocabulary
      - token vector  : Word2Vec vector trained on the same corpus
      - sentence vec  : weighted average of token vectors

    Fallback behaviour:
      - token in W2V but not in TF-IDF vocab (or weight=0) → excluded
      - all tokens excluded but some W2V vectors found → unweighted mean
      - no W2V vectors found at all → zero vector
    """

    def __init__(self, vector_size: int = 100):
        self.vector_size = vector_size
        self._tfidf = TFIDFEmbedder(vector_size=vector_size)
        self._w2v   = SkipGramEmbedder(vector_size=vector_size)

    def fit(self, sentences: Sentences) -> 'TFIDFWeightedWord2VecEmbedder':
        self._tfidf.fit(sentences)
        self._w2v.fit(sentences)
        return self

    def transform(self, sentences: Sentences) -> EmbeddingMatrix:
        vocab_to_idx = {w: i for i, w in enumerate(self._tfidf.vectorizer.get_feature_names_out())}
        tfidf_matrix = self._tfidf.vectorizer.transform(sentences)   # sparse (N, vocab)

        vecs = []
        for i, s in enumerate(sentences):
            tokens     = s.lower().split()
            token_vecs = []
            weights    = []
            for t in tokens:
                if t in self._w2v.model.wv and t in vocab_to_idx:
                    token_vecs.append(self._w2v.model.wv[t])
                    weights.append(tfidf_matrix[i, vocab_to_idx[t]])
            if token_vecs and sum(weights) > 0:
                vecs.append(np.average(token_vecs, axis=0, weights=weights))
            elif token_vecs:
                vecs.append(np.mean(token_vecs, axis=0))
            else:
                vecs.append(np.zeros(self.vector_size))

        return np.array(vecs, dtype=np.float32)

    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix:
        return self.fit(sentences).transform(sentences)
