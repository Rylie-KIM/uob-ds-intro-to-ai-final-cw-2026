import numpy as np
import numpy.typing as npt
from typing import TypeAlias
import gensim.downloader

# Type Alias
Sentences:          TypeAlias = list[str]           # original string sentences
TokenisedSentences: TypeAlias = list[list[str]]     # sentence into words
EmbeddingMatrix:    TypeAlias = npt.NDArray[np.float32]  # sentence embedding matrix

class PretrainedWord2VecEmbedder:
    def __init__(
            self,
            lowercase: bool = False,  # Google News is case-sensitive (Apple != apple)
    ):
        self.vector_size = 300        # fixed. Google News vectors are always 300-d
        self.lowercase   = lowercase

        print(f"PretrainedWord2VecEmbedder loading 'word2vec-google-news-300'")
        self.model = gensim.downloader.load('word2vec-google-news-300')
        print(f"PretrainedWord2VecEmbedder ready  vocab size: {len(self.model)}  dim: {self.vector_size}")

    # vectors are already pretrained, no training needed 
    def fit(self, _sentences: Sentences) -> 'PretrainedWord2VecEmbedder':
        return self

    def transform(self, sentences: Sentences) -> EmbeddingMatrix:  # (N, 300) matrix, N: number of sentences
        # mean-pool (word vectors >> single sentence vector)
        sentence_embeddings = []
        for s in sentences:
            tokens = s.lower().split() if self.lowercase else s.split()
            word_embeddings_found = []
            for t in tokens:
                if t not in self.model:
                    continue
                word_embeddings_found.append(self.model[t])
            if not word_embeddings_found:
                sentence_embeddings.append(np.zeros(self.vector_size, dtype=np.float32))
            else:
                sentence_embeddings.append(np.mean(word_embeddings_found, axis=0).astype(np.float32))
        return np.array(sentence_embeddings, dtype=np.float32)

    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix:
        return self.fit(sentences).transform(sentences)

    def oov_rate(self, sentences: Sentences) -> float:
        # For data type b - returns fraction of tokens not found in the Google News vocabulary 
        total = 0
        oov   = 0
        for s in sentences:
            tokens = s.lower().split() if self.lowercase else s.split()
            total += len(tokens)
            oov   += sum(1 for t in tokens if t not in self.model)
        rate = oov / total if total > 0 else 0.0
        print(f"OOV: {oov} / {total} tokens  ({rate * 100:.1f} %)")
        return rate
