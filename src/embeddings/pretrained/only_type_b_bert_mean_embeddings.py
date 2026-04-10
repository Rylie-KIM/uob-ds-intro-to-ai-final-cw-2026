from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import TypeAlias

import torch
from transformers import BertModel, BertTokenizer

# BERT (bert-base-uncased) — mean pooling over content tokens (CLS and SEP excluded).
# Output dimension: 768

# fit() is a no-op (model is pretrained); included so this class shares the same
# fit / transform / fit_transform interface as TFIDFLSAEmbedder.

Sentences:       TypeAlias = list[str]
EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]


class BertMeanEmbedder:
    OUTPUT_DIM: int = 768

    def __init__(self) -> None:
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model     = BertModel.from_pretrained('bert-base-uncased')
        self._model.eval()
    def fit(self, sentences: Sentences) -> 'BertMeanEmbedder':
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
                last_hidden = output.last_hidden_state[0]   # (seq_len, 768)
                last_hidden = last_hidden[1:-1]              # strip CLS + SEP
                mean_emb    = last_hidden.mean(dim=0)        # (768,)
                embeddings.append(mean_emb.cpu().numpy())

        return np.array(embeddings, dtype=np.float32)

    def fit_transform(self, sentences: Sentences) -> EmbeddingMatrix:
        return self.fit(sentences).transform(sentences)
