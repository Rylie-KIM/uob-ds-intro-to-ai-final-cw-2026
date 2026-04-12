"""
src/pipelines/evaluation/type-b/final_analysis_normed.py

Final embedding comparison analysis for Type-B normalised runs.

Mirrors final_analysis.py but reads from:
  metrics/type-b/prediction-normalised/test_results_normed.csv

and saves all outputs to:
  metrics/type-b/prediction-normalised/
    final_ranking_normed.csv     — ranked summary with composite score + collapse flag
    final_per_digit_normed.csv   — top-1 and mean_rank broken down by n_digits per run

Console output:
  Ranked table with composite score breakdown
  Collapse diagnosis per run
  Top-3 summary

Ranking methodology (identical to final_analysis.py)
-----------------------------------------------------
  composite_score = 0.4 * mrr_norm
                  + 0.35 * median_rank_norm
                  + 0.25 * colour_size_correct

  mrr_norm         = test_mrr / max(test_mrr)
  median_rank_norm = 1 - (test_median_rank / CORPUS_SIZE)

  collapsed = colour_size_correct < 0.95

Usage
-----
  python src/pipelines/evaluation/type-b/final_analysis_normed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import PREDICTIONS_B_NORMED  # noqa: E402

CORPUS_SIZE = 1001

W_MRR         = 0.40
W_MEDIAN_RANK = 0.35
W_COLLAPSE    = 0.25

COLLAPSE_THRESHOLD = 0.95

_DESCRIPTIONS: dict[str, str] = {
    'B0n':  'TF-IDF + LSA (100-dim), L2-normed targets — normalised baseline',
    'E2an': 'SBERT all-MiniLM-L6-v2 (384-dim), L2-normed targets',
    'E2bn': 'SBERT fine-tuned on Type-B corpus (384-dim), L2-normed targets',
    'E2en': 'TinyBERT mean-pool (312-dim), L2-normed targets',
    'E2fn': 'TinyBERT [CLS] pooler (312-dim), L2-normed targets',
    'E2gn': 'GloVe word-avg (300-dim), L2-normed targets',
    'E2hn': 'Word2Vec Google News pretrained (300-dim), L2-normed targets',
    'E2in': 'Word2Vec skip-gram in-domain (100-dim), L2-normed targets',
    'E2kn': 'TF-IDF weighted Word2Vec (100-dim), L2-normed targets',
    'E2ln': 'BERT base mean-pool (768-dim), L2-normed targets',
    'E2mn': 'BERT base [CLS] pooler (768-dim), L2-normed targets',
}


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

def compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalised score components and composite_score, then rank rows.

    mrr_norm         = test_mrr / max(test_mrr)
    median_rank_norm = 1 - (test_median_rank / CORPUS_SIZE)
    composite_score  = W_MRR * mrr_norm
                     + W_MEDIAN_RANK * median_rank_norm
                     + W_COLLAPSE    * colour_size_correct
    """
    out = df.copy()

    mrr_max = out['test_mrr'].max()
    out['mrr_norm']         = out['test_mrr'] / mrr_max if mrr_max > 0 else 0.0
    out['median_rank_norm'] = 1.0 - (out['test_median_rank'] / CORPUS_SIZE)
    out['collapsed']        = out['colour_size_correct'] < COLLAPSE_THRESHOLD

    out['composite_score'] = (
        W_MRR         * out['mrr_norm']
        + W_MEDIAN_RANK * out['median_rank_norm']
        + W_COLLAPSE    * out['colour_size_correct']
    )

    out = out.sort_values('composite_score', ascending=False).reset_index(drop=True)
    out.insert(0, 'rank', range(1, len(out) + 1))
    out['description'] = out['run_id'].map(_DESCRIPTIONS).fillna('')

    return out


def build_per_digit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape top1_{n}d and mean_rank_{n}d columns into long-form:
    run_id | embedding | n_digits | top1 | mean_rank
    """
    records = []
    for _, row in df.iterrows():
        for nd in range(1, 7):
            records.append({
                'run_id':    row['run_id'],
                'embedding': row['embedding'],
                'n_digits':  nd,
                'top1':      row.get(f'top1_{nd}d', float('nan')),
                'mean_rank': row.get(f'mean_rank_{nd}d', float('nan')),
            })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Console output
# ══════════════════════════════════════════════════════════════════════════════

def _print_ranking(ranked: pd.DataFrame) -> None:
    sep = '=' * 90
    print(f'\n{sep}')
    print('  Type-B Normalised — Final Embedding Ranking')
    print(f'  Composite score = {W_MRR}×MRR_norm + {W_MEDIAN_RANK}×MedianRank_norm'
          f' + {W_COLLAPSE}×ColourSize')
    print(sep)
    print(f"  {'Rank':<5} {'Run':<7} {'Embedding':<26} {'top-1':>6} {'MRR':>7} "
          f"{'median_rank':>12} {'mean_rank':>10} {'CS_correct':>11} "
          f"{'collapsed':>9} {'score':>7}")
    print(f"  {'-'*88}")

    for _, row in ranked.iterrows():
        flag = '  !!!' if row['collapsed'] else ''
        print(
            f"  {int(row['rank']):<5} {row['run_id']:<7} {row['embedding']:<26} "
            f"{row['test_top1']:>6.4f} {row['test_mrr']:>7.4f} "
            f"{int(row['test_median_rank']):>12} {row['test_mean_rank']:>10.1f} "
            f"{row['colour_size_correct']:>11.3f} "
            f"{'YES' if row['collapsed'] else 'no':>9} "
            f"{row['composite_score']:>7.4f}{flag}"
        )

    print(f"  {sep}")
    print('  !! = embedding collapse detected (colour_size_correct < 0.95)')
    print(f'{sep}\n')


def _print_collapse_diagnosis(ranked: pd.DataFrame) -> None:
    collapsed = ranked[ranked['collapsed']]
    healthy   = ranked[~ranked['collapsed']]

    print('── Collapse Diagnosis ──────────────────────────────────────────────')
    if collapsed.empty:
        print('  All runs: no collapse detected.')
    else:
        print(f'  Collapsed runs ({len(collapsed)}):')
        for _, row in collapsed.iterrows():
            print(f'    {row["run_id"]} ({row["embedding"]}): '
                  f'colour_size_correct={row["colour_size_correct"]:.3f}, '
                  f'mean_rank={row["test_mean_rank"]:.0f}')
        print()
        print(f'  Healthy runs ({len(healthy)}):')
        for _, row in healthy.iterrows():
            print(f'    {row["run_id"]} ({row["embedding"]}): '
                  f'colour_size_correct={row["colour_size_correct"]:.3f}')
    print()


def _print_top3(ranked: pd.DataFrame) -> None:
    print('── Top-3 Summary ───────────────────────────────────────────────────')
    for _, row in ranked[ranked['rank'] <= 3].iterrows():
        print(f'\n  #{int(row["rank"])}  {row["run_id"]} — {row["embedding"]}')
        print(f'      {row["description"]}')
        print(f'      top-1={row["test_top1"]:.4f}  MRR={row["test_mrr"]:.4f}  '
              f'median_rank={int(row["test_median_rank"])}  '
              f'mean_rank={row["test_mean_rank"]:.1f}')
        print(f'      composite_score={row["composite_score"]:.4f}  '
              f'collapsed={"YES" if row["collapsed"] else "no"}')
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    results_csv = PREDICTIONS_B_NORMED / 'test_results_normed.csv'
    if not results_csv.exists():
        print(f'ERROR: test_results_normed.csv not found at {results_csv}')
        print('Run normalised evaluations first:')
        print('  python src/pipelines/evaluation/type-b/run_evals_stage1_normed_b.py')
        return

    df = pd.read_csv(results_csv)
    if df.empty:
        print('ERROR: test_results_normed.csv is empty.')
        return

    print(f'\nLoaded {len(df)} normed runs from {results_csv.name}')

    # ── Compute ranking ────────────────────────────────────────────────────────
    ranked    = compute_ranking(df)
    per_digit = build_per_digit(ranked)

    # ── Console output ─────────────────────────────────────────────────────────
    _print_ranking(ranked)
    _print_collapse_diagnosis(ranked)
    _print_top3(ranked)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    out_cols = [
        'rank', 'run_id', 'embedding', 'dim', 'loss_fn',
        'best_epoch', 'total_epochs',
        'test_top1', 'test_top5', 'test_mrr',
        'test_mean_rank', 'test_median_rank', 'test_mean_cosine',
        'colour_size_correct',
        'mrr_norm', 'median_rank_norm', 'composite_score',
        'collapsed', 'description',
    ]
    out_cols = [c for c in out_cols if c in ranked.columns]

    PREDICTIONS_B_NORMED.mkdir(parents=True, exist_ok=True)

    final_csv = PREDICTIONS_B_NORMED / 'final_ranking_normed.csv'
    ranked[out_cols].to_csv(final_csv, index=False)
    print(f'  [saved] {final_csv}')

    digit_csv = PREDICTIONS_B_NORMED / 'final_per_digit_normed.csv'
    per_digit.to_csv(digit_csv, index=False)
    print(f'  [saved] {digit_csv}')

    print(f'\nDone.\n')


if __name__ == '__main__':
    main()
