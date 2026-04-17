from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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



def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    str,
) -> float:
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

def run_validation(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, _sentences, embs in loader:
            imgs = imgs.to(device)
            embs = embs.to(device)
            preds = model(imgs)
            total_loss += criterion(preds, embs).item()

    return total_loss / len(loader)

def run_val_retrieval(
    model:          nn.Module,
    val_loader:     DataLoader,
    all_embeddings: torch.Tensor,   # (N_corpus, dim) — full corpus on CPU
    all_sentences:  list[str],
    device:         str,
    top_k:          tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, float]:
    """Compute retrieval metrics on the validation set against the full corpus.

    For each val image the model predicts an embedding, then ranks all corpus
    sentences by cosine similarity.  Returns val_top1–5, val_mrr,
    val_mean_cosine, val_mean_rank, val_median_rank.
    """
    model.eval()
    corpus = all_embeddings.to(device)              # (N, dim)
    sent_to_idx: dict[str, int] = {s: i for i, s in enumerate(all_sentences)}

    ranks:       list[int]   = []
    cosine_sims: list[float] = []
    correct_at_k             = {k: 0 for k in top_k}
    total                    = 0

    with torch.no_grad():
        for imgs, true_sentences, _true_embs in val_loader:
            imgs      = imgs.to(device)
            pred_embs = model(imgs)                 # (B, dim)

            # (B, N) cosine similarity matrix
            sims = F.cosine_similarity(
                pred_embs.unsqueeze(1),             # (B, 1, dim)
                corpus.unsqueeze(0),                # (1, N, dim)
                dim=2,
            )
            sorted_idx = sims.argsort(dim=1, descending=True)  # (B, N)

            for b_i, true_sent in enumerate(true_sentences):
                true_idx = sent_to_idx.get(true_sent)
                if true_idx is None:
                    continue
                rank = int(
                    (sorted_idx[b_i] == true_idx).nonzero(as_tuple=True)[0][0].item()
                ) + 1   # 1-indexed
                cosine_sims.append(float(sims[b_i, true_idx].item()))
                ranks.append(rank)
                for k in top_k:
                    if rank <= k:
                        correct_at_k[k] += 1
                total += 1

    if total == 0:
        return {}

    return {
        **{f'val_top{k}': round(correct_at_k[k] / total, 6) for k in top_k},
        'val_mrr':         round(float(sum(1.0 / r for r in ranks) / total), 6),
        'val_mean_cosine': round(float(sum(cosine_sims) / total), 6),
        'val_mean_rank':   round(float(sum(ranks) / total), 2),
        'val_median_rank': round(float(np.median(ranks)), 2),
    }


def train_one_epoch_normalised(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    str,
) -> float:
    """Same as train_one_epoch but L2-normalises target embeddings before loss.

    Normalising the target to unit norm makes MSELoss mathematically equivalent
    to (1 - cosine_similarity), aligning the training objective with the cosine-
    based retrieval metric used at evaluation time.
    """
    model.train()
    total_loss = 0.0

    for imgs, _sentences, embs in loader:
        imgs = imgs.to(device)
        embs = F.normalize(embs.float(), dim=1).to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, embs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def run_validation_normalised(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str,
) -> float:
    """Same as run_validation but L2-normalises target embeddings before loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, _sentences, embs in loader:
            imgs = imgs.to(device)
            embs = F.normalize(embs.float(), dim=1).to(device)
            preds = model(imgs)
            total_loss += criterion(preds, embs).item()

    return total_loss / len(loader)


class CosineLoss(nn.Module):
    """L = (1 - cosine_similarity(y_hat, y)).mean()

    Directly optimises the same directional objective used at retrieval time.
    Scale-invariant: only the direction of the predicted embedding matters.
    """

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cosine_sim = F.cosine_similarity(y_hat, y, dim=1)
        return (1.0 - cosine_sim).mean()


class CombinedLoss(nn.Module):

    # L = alpha * MSE(y_hat, y) + (1 - alpha) * (1 - cosine_similarity(y_hat, y)).mean()

    # Rationale: MSELoss penalises both direction and magnitude errors in embedding space.
    # CosineLoss (1 - cosine_sim) penalises only directional error, aligning the
    # training objective with the cosine-based retrieval metric used at evaluation.
    # CombinedLoss optimises both simultaneously.

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha   = alpha
        self._mse    = nn.MSELoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse_loss    = self._mse(y_hat, y)
        cosine_sim  = torch.nn.functional.cosine_similarity(y_hat, y, dim=1)
        cosine_loss = (1.0 - cosine_sim).mean()
        return self.alpha * mse_loss + (1.0 - self.alpha) * cosine_loss
