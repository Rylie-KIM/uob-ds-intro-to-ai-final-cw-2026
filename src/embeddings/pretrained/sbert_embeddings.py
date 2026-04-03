import numpy as np
import numpy.typing as npt
from typing import TypeAlias
from sentence_transformers import SentenceTransformer

# Type Aliases
Sentences: TypeAlias = list[str]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class SBERTEmbedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self.model: SentenceTransformer | None = None

    def fit(self, sentences: Sentences) -> "SBERTEmbedder":
        # SBERT is pretrained → just load model
        self.model = SentenceTransformer(self.model_name, device=self.device)

        print(
            f"SBERT loaded: {self.model_name} | dim: {self.model.get_sentence_embedding_dimension()}"
        )
        return self

    def transform(self, sentences: Sentences) -> EmbeddingMatrix:
        if self.model is None:
            raise ValueError("Call fit() before transform()")

        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=False
        )

        return embeddings.astype(np.float32)

    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix:
        return self.fit(sentences).transform(sentences)