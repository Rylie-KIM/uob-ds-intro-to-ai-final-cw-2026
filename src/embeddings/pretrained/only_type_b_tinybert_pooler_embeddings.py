from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import TypeAlias

import torch
from transformers import AutoModel, AutoTokenizer

# TinyBERT (huawei-noah/TinyBERT_General_4L_312D) — pooler output
# ([CLS] token after linear + tanh projection).
# Output dimension: 312

# fit() is a no-op (model is pretrained); included so this class shares the same
# fit / transform / fit_transform interface as TFIDFLSAEmbedder.

# NOTE: do_lower_case=True is required — TinyBERT tokenizer does not recognise
#       capitalised words.

Sentences:       TypeAlias = list[str]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class TinyBertPoolerEmbedder:
    OUTPUT_DIM: int = 312

    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            'huawei-noah/TinyBERT_General_4L_312D',
            do_lower_case=True,   # required: tokenizer does not handle capitalised words
        )
        self._model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
        self._model.eval()

    def fit(self, sentences: Sentences) -> 'TinyBertPoolerEmbedder':
        """No-op — TinyBERT is pretrained. Returns self for chaining."""
        return self

    def transform(self, sentences: Sentences) -> EmbeddingMatrix:
        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for sentence in sentences:
                processed = self._tokenizer(sentence, return_tensors='pt')
                output    = self._model(
                    input_ids      = processed['input_ids'],
                    attention_mask = processed['attention_mask'],
                )
                pooler = output.pooler_output.squeeze(0)   # (312,)
                embeddings.append(pooler.cpu().numpy())

        return np.array(embeddings, dtype=np.float32)

    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix:
        return self.fit(sentences).transform(sentences)
