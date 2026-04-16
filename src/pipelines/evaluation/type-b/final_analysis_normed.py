from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import (  # noqa: E402
    PREDICTIONS_B_NORMED,
    PREDICTIONS_B_S2_NORMED,
)

# Step 1 — Collapse hard filter
# Step 2 — Normalise each metric to [0, 1] across all runs
# Step 3 — Weighted composite (non-collapsed runs only)

CORPUS_SIZE = 10008  # Type-B full corpus sentence count

W_MRR         = 0.35
W_MEDIAN_RANK = 0.25
W_TOP1        = 0.20
W_TOP5        = 0.15
W_COSINE      = 0.05

COLLAPSE_THRESHOLD = 0.95

_DESCRIPTIONS: dict[str, str] = {
    # Stage 1 normed — embedding axis (cnn_1layer fixed, L2-normed targets)
    'B0n':  'TF-IDF + LSA (100-dim), MSE — normalised baseline',
    'E2an': 'SBERT all-MiniLM-L6-v2 (384-dim), frozen, CosineLoss — normed',
    'E2bn': 'SBERT fine-tuned on Type-B corpus (384-dim), CosineLoss — normed',
    'E2en': 'TinyBERT mean-pool (312-dim), MSE — normed',
    'E2fn': 'TinyBERT [CLS] pooler (312-dim), MSE — normed',
    'E2gn': 'GloVe word-avg (300-dim), MSE — normed',
    'E2hn': 'Word2Vec Google News pretrained (300-dim), MSE — normed',
    'E2in': 'Word2Vec skip-gram in-domain (100-dim), MSE — normed',
    'E2kn': 'TF-IDF weighted Word2Vec (100-dim), MSE — normed',
    'E2ln': 'BERT base mean-pool (768-dim), MSE — normed',
    'E2mn': 'BERT base [CLS] pooler (768-dim), MSE — normed',
    # Stage 2 normed — architecture axis (tinybert_mean fixed, L2-normed targets)
    'S2an': 'Stage-2 cnn_1layer + TinyBERT-mean (312-dim), MSE — normed',
    'S2bn': 'Stage-2 cnn_3layer + TinyBERT-mean (312-dim), MSE — normed',
    'S2cn': 'Stage-2 ResNet18-pretrained + TinyBERT-mean (312-dim), MSE — normed',
    'S2ad': 'Stage-2 cnn_1layer + TinyBERT-mean (312-dim), Combined — normed',
    'S2bd': 'Stage-2 cnn_3layer + TinyBERT-mean (312-dim), Combined — normed',
    'S2cd': 'Stage-2 ResNet18-pretrained + TinyBERT-mean (312-dim), Combined — normed',
}


# scoring logic
def compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalised score components and composite_score to the results dataframe.

    Normalisation:
      mrr_norm         = test_mrr        / max(test_mrr)        ∈ [0, 1]
      median_rank_norm = 1 - (test_median_rank / CORPUS_SIZE)   ∈ [0, 1]
      top1_norm        = test_top1       / max(test_top1)        ∈ [0, 1]
      top5_norm        = test_top5       / max(test_top5)        ∈ [0, 1]
      cosine_norm      = test_mean_cosine                        ∈ [0, 1]

    Composite (non-collapsed runs only):
      composite_score = W_MRR * mrr_norm + W_MEDIAN_RANK * median_rank_norm
                      + W_TOP1 * top1_norm + W_TOP5 * top5_norm
                      + W_COSINE * cosine_norm

    Collapsed runs (colour_size_correct < COLLAPSE_THRESHOLD) receive
    composite_score = 0.0 unconditionally.
    """
    out = df.copy()
    out['collapsed'] = out['colour_size_correct'] < COLLAPSE_THRESHOLD

    def _safe_norm(series: pd.Series) -> pd.Series:
        mx = series.max()
        return series / mx if mx > 0 else series * 0.0

    out['mrr_norm']         = _safe_norm(out['test_mrr'])
    out['median_rank_norm'] = 1.0 - (out['test_median_rank'] / CORPUS_SIZE)
    out['top1_norm']        = _safe_norm(out['test_top1'])
    out['top5_norm']        = _safe_norm(out['test_top5'])
    out['cosine_norm']      = out['test_mean_cosine']

    raw_score = (
        W_MRR         * out['mrr_norm']
        + W_MEDIAN_RANK * out['median_rank_norm']
        + W_TOP1        * out['top1_norm']
        + W_TOP5        * out['top5_norm']
        + W_COSINE      * out['cosine_norm']
    )

    out['composite_score'] = raw_score.where(~out['collapsed'], other=0.0)
    out = out.sort_values('composite_score', ascending=False).reset_index(drop=True)
    out.insert(0, 'rank', range(1, len(out) + 1))
    out['description'] = out['run_id'].map(_DESCRIPTIONS).fillna('')
    return out


def build_per_digit(df: pd.DataFrame) -> pd.DataFrame:
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


# console output
def _print_ranking(ranked: pd.DataFrame) -> None:
    sep = '=' * 90
    print(f'\n{sep}')
    print('  Type-B Normalised — Stage-1 + Stage-2 Final Ranking')
    print(f'  Composite score = {W_MRR}×MRR_norm + {W_MEDIAN_RANK}×MedianRank_norm'
          f' + {W_TOP1}×Top1_norm + {W_TOP5}×Top5_norm + {W_COSINE}×Cosine'
          f'  [collapsed → 0.0]')
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

    print(f'\nLoaded {len(df)} Stage-1 normed runs from {results_csv.name}')

    # Append Stage-2 normalised results (if available)
    s2_normed_csv = PREDICTIONS_B_S2_NORMED / 'test_results_s2_normed.csv'
    if s2_normed_csv.exists():
        s2_df = pd.read_csv(s2_normed_csv)
        if not s2_df.empty:
            df = pd.concat([df, s2_df], ignore_index=True)
            print(f'  → {len(s2_df)} Stage-2 normed run(s) appended from {s2_normed_csv.name}')
    else:
        print(f'  [info] Stage-2 normed results not found at {s2_normed_csv}')
        print('         Run: python src/pipelines/evaluation/type-b/run_evals_stage2_b.py '
              '--variant normalised')

    # Compute ranking
    ranked    = compute_ranking(df)
    per_digit = build_per_digit(ranked)

    # Console output
    _print_ranking(ranked)
    _print_collapse_diagnosis(ranked)
    _print_top3(ranked)

    # Save CSVs
    out_cols = [
        'rank', 'run_id', 'embedding', 'dim', 'loss_fn',
        'best_epoch', 'total_epochs',
        'test_top1', 'test_top5', 'test_mrr',
        'test_mean_rank', 'test_median_rank', 'test_mean_cosine',
        'colour_size_correct',
        'mrr_norm', 'median_rank_norm', 'top1_norm', 'top5_norm', 'cosine_norm',
        'composite_score',
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

    # Leaderboard CSV
    leaderboard_cols = [
        'rank', 'run_id', 'cnn', 'embedding', 'dim', 'loss_fn',
        'test_top1', 'test_top5', 'test_mrr',
        'test_median_rank', 'collapsed', 'composite_score',
    ]
    leaderboard_cols = [c for c in leaderboard_cols if c in ranked.columns]
    lb = ranked[leaderboard_cols].copy()
    for col in ('test_top1', 'test_top5', 'test_mrr', 'composite_score'):
        if col in lb.columns:
            lb[col] = lb[col].round(4)

    lb_csv = PREDICTIONS_B_NORMED / 'leaderboard_normed.csv'
    lb.to_csv(lb_csv, index=False)
    print(f'  [saved] {lb_csv}')

    print(f'\nDone.\n')


if __name__ == '__main__':
    main()
