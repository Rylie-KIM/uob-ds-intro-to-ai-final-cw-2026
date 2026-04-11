from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def evaluate(
    model:          nn.Module,
    test_loader:    DataLoader,
    all_embeddings: torch.Tensor,
    all_sentences:  list[str],
    device:         str,
    top_k:          tuple[int, ...] = (1, 5),
) -> tuple[dict[str, float], pd.DataFrame]:
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


def save_results(
    metrics:     dict[str, float],
    results_df:  pd.DataFrame,
    metrics_dir: Path,
    run_name:    str,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_path = metrics_dir / f'{run_name}_predictions.csv'
    results_df.to_csv(pred_path, index=False)
    print(f'  [saved] predictions → {pred_path.name}')

    row_df           = pd.DataFrame([{'run_name': run_name, **metrics}])
    run_summary_path = metrics_dir / f'{run_name}_summary.csv'
    row_df.to_csv(run_summary_path, index=False)
    print(f'  [saved] run summary → {run_summary_path.name}')

    agg_path = metrics_dir / 'results_summary.csv'
    if agg_path.exists():
        existing = pd.read_csv(agg_path)
        existing = existing[existing['run_name'] != run_name]
        pd.concat([existing, row_df], ignore_index=True).to_csv(agg_path, index=False)
    else:
        row_df.to_csv(agg_path, index=False)
    print(f'  [saved] agg summary → {agg_path.name}')
