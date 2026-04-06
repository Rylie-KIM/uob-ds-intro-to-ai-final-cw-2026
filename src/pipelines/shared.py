"""
src/pipelines/shared.py
Shared training and validation loop utilities used by all dataset-type pipelines.

Functions
---------
train_one_epoch  : run one full pass over the training DataLoader
run_validation   : evaluate model on a DataLoader without gradient updates
set_seed         : fix all random seeds for reproducibility
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds to ensure reproducible training across runs.

    Sets:
      - Python random module
      - NumPy random
      - PyTorch CPU and CUDA seeds
      - cuDNN deterministic mode (disables non-deterministic algorithms)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    str,
) -> float:
    """
    Run one complete pass over the training DataLoader.

    Each batch is a triple (images, sentences, embeddings) as returned by
    TypeBDataset.__getitem__. Only images and embeddings are used here;
    sentences are passed through to keep the DataLoader interface consistent.

    Parameters
    ----------
    model     : CNN model in training mode after this call
    loader    : DataLoader yielding (image_tensor, sentence_str, embedding_tensor)
    optimizer : optimiser (e.g. Adam)
    criterion : loss function (e.g. MSELoss or CombinedLoss)
    device    : 'cpu' | 'cuda' | 'mps'

    Returns
    -------
    mean training loss over all batches in this epoch
    """
    model.train()
    total_loss = 0.0

    for imgs, _sentences, embs in loader:
        imgs = imgs.to(device)
        embs = embs.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, embs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════════════════
# Validation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_validation(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> float:
    """
    Evaluate model on a DataLoader without gradient computation.

    Parameters
    ----------
    model     : CNN model (set to eval mode internally)
    loader    : DataLoader yielding (image_tensor, sentence_str, embedding_tensor)
    criterion : loss function used during training (same as train_one_epoch)
    device    : 'cpu' | 'cuda' | 'mps'

    Returns
    -------
    mean validation loss over all batches
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, _sentences, embs in loader:
            imgs = imgs.to(device)
            embs = embs.to(device)
            preds = model(imgs)
            total_loss += criterion(preds, embs).item()

    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════════════════════
# Combined loss (MSE + Cosine)
# ══════════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    Weighted combination of MSELoss and Cosine Loss.

    L = alpha * MSE(y_hat, y) + (1 - alpha) * (1 - cosine_similarity(y_hat, y)).mean()

    Rationale
    ---------
    MSELoss penalises both direction and magnitude errors in embedding space.
    CosineLoss (1 - cosine_sim) penalises only directional error, aligning the
    training objective with the cosine-based retrieval metric used at evaluation.
    CombinedLoss optimises both simultaneously.

    Parameters
    ----------
    alpha : weight for MSE term; (1-alpha) for cosine term. Default 0.5.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha   = alpha
        self._mse    = nn.MSELoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse_loss    = self._mse(y_hat, y)
        cosine_sim  = torch.nn.functional.cosine_similarity(y_hat, y, dim=1)
        cosine_loss = (1.0 - cosine_sim).mean()
        return self.alpha * mse_loss + (1.0 - self.alpha) * cosine_loss
