"""
src/pipelines/evaluation/evaluate.py
Retrieval-based evaluation utilities for image-to-sentence experiments.

Evaluation protocol
-------------------
1. For each test image, the trained CNN predicts an embedding vector.
2. The predicted embedding is compared (cosine similarity) against ALL
   corpus embeddings (full dataset, not just test set).
3. Sentences are ranked by cosine similarity; top-k accuracy is computed.

This mirrors the real use case: given a new image, retrieve the most
likely description from the known sentence corpus.

Functions
---------
evaluate     : compute retrieval metrics for a trained model on the test set
save_results : persist metrics to results_summary.csv and per-run predictions CSV
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model:          nn.Module,
    test_loader:    DataLoader,
    all_embeddings: torch.Tensor,
    all_sentences:  list[str],
    device:         str,
    top_k:          tuple[int, ...] = (1, 5),
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Evaluate a trained CNN on the test set using cosine-similarity retrieval.

    For each test image:
      - Predict embedding with CNN
      - Rank all corpus sentences by cosine similarity to the prediction
      - Record rank of the true sentence and whether it appears in top-k

    Parameters
    ----------
    model          : trained CNN (will be set to eval mode)
    test_loader    : DataLoader for test set; yields (img, sentence, embedding)
    all_embeddings : Tensor (N_corpus, dim) — pre-computed embeddings of all sentences
    all_sentences  : list of N_corpus sentence strings (same order as all_embeddings)
    device         : 'cpu' | 'cuda' | 'mps'
    top_k          : tuple of k values for Recall@k computation

    Returns
    -------
    metrics    : dict with keys top_{k}_acc, mean_cosine_sim, mean_rank
    results_df : per-sample DataFrame for error analysis
    """
    model.eval()
    all_embeddings = all_embeddings.to(device)          # (N_corpus, dim)

    rows: list[dict] = []
    correct_at_k = {k: 0 for k in top_k}
    total = 0
    cosine_sims: list[float] = []
    ranks: list[int] = []

    with torch.no_grad():
        for imgs, true_sentences, _true_embs in test_loader:
            imgs = imgs.to(device)                       # (B, 3, 128, 128)
            pred_embs = model(imgs)                      # (B, dim)

            # Cosine similarity: pred_embs (B, dim) vs all_embeddings (N, dim)
            # Result: (B, N)
            sims = F.cosine_similarity(
                pred_embs.unsqueeze(1),                  # (B, 1, dim)
                all_embeddings.unsqueeze(0),             # (1, N, dim)
                dim=2,
            )                                            # (B, N)

            # Rank sentences by similarity (descending)
            sorted_idx = sims.argsort(dim=1, descending=True)  # (B, N)

            for b_i, true_sent in enumerate(true_sentences):
                # Find position of the true sentence in the ranking
                try:
                    true_corpus_idx = all_sentences.index(true_sent)
                except ValueError:
                    # True sentence not in corpus — skip (should not happen)
                    continue

                rank_tensor   = (sorted_idx[b_i] == true_corpus_idx).nonzero(as_tuple=True)[0]
                rank          = int(rank_tensor[0].item()) + 1  # 1-indexed
                best_sim      = float(sims[b_i, true_corpus_idx].item())
                pred_sent_idx = int(sorted_idx[b_i, 0].item())
                pred_sent     = all_sentences[pred_sent_idx]

                for k in top_k:
                    if rank <= k:
                        correct_at_k[k] += 1

                cosine_sims.append(best_sim)
                ranks.append(rank)
                total += 1

                rows.append({
                    'true_sentence': true_sent,
                    'pred_sentence': pred_sent,
                    'true_rank':     rank,
                    'cosine_sim':    best_sim,
                    **{f'top_{k}_correct': int(rank <= k) for k in top_k},
                })

    if total == 0:
        raise RuntimeError('No samples were evaluated. Check test_loader and all_sentences.')

    metrics = {
        **{f'top_{k}_acc': correct_at_k[k] / total for k in top_k},
        'mean_cosine_sim': float(sum(cosine_sims) / total),
        'mean_rank':       float(sum(ranks) / total),
        'n_test':          total,
    }

    results_df = pd.DataFrame(rows)
    return metrics, results_df


# ══════════════════════════════════════════════════════════════════════════════
# Result persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    metrics:     dict[str, float],
    results_df:  pd.DataFrame,
    metrics_dir: Path,
    run_name:    str,
) -> None:
    """
    Persist evaluation results to disk.

    1. Appends a summary row to  metrics_dir/results_summary.csv
       (creates the file if it does not exist).
    2. Saves per-sample predictions to  metrics_dir/{run_name}_predictions.csv

    Parameters
    ----------
    metrics     : dict returned by evaluate()
    results_df  : per-sample DataFrame returned by evaluate()
    metrics_dir : directory where CSVs are written
    run_name    : identifier string, e.g. 'b_cnn_sbert'
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-run predictions ────────────────────────────────────────────────────
    pred_path = metrics_dir / f'{run_name}_predictions.csv'
    results_df.to_csv(pred_path, index=False)
    print(f'  [saved] predictions → {pred_path.name}')

    # ── Per-run summary ────────────────────────────────────────────────────────
    row_df           = pd.DataFrame([{'run_name': run_name, **metrics}])
    run_summary_path = metrics_dir / f'{run_name}_summary.csv'
    row_df.to_csv(run_summary_path, index=False)
    print(f'  [saved] run summary → {run_summary_path.name}')

    # ── Aggregate summary (all runs combined) ──────────────────────────────────
    agg_path = metrics_dir / 'results_summary.csv'
    if agg_path.exists():
        existing = pd.read_csv(agg_path)
        existing = existing[existing['run_name'] != run_name]
        pd.concat([existing, row_df], ignore_index=True).to_csv(agg_path, index=False)
    else:
        row_df.to_csv(agg_path, index=False)
    print(f'  [saved] agg summary → {agg_path.name}')
