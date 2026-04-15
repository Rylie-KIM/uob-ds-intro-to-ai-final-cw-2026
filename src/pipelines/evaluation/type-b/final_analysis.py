"""
src/pipelines/evaluation/type-b/final_analysis.py

Final embedding comparison analysis for Type-B (Stage 1).

Reads test_results.csv and per-sample prediction CSVs, then produces:

  CSV outputs (METRICS_B/):
    final_ranking.csv         — ranked summary with composite score + collapse flag
    final_per_digit.csv       — top-1 and mean_rank broken down by n_digits per run

  Console output:
    Ranked table with composite score breakdown
    Collapse diagnosis per run

Ranking methodology
-------------------
  Step 1 — Collapse hard filter
     collapsed = colour_size_correct < COLLAPSE_THRESHOLD (0.95)
     Any collapsed run receives composite_score = 0.0 and is ranked last.
     Rationale: a model that cannot distinguish sentences at all has no retrieval
     value; a soft penalty would let it compete with genuinely discriminative models.

  Step 2 — Normalise each metric to [0, 1] across all runs
     mrr_norm          = test_mrr        / max(test_mrr)          (relative)
     median_rank_norm  = 1 - (test_median_rank / CORPUS_SIZE)     (absolute)
     top1_norm         = test_top1       / max(test_top1)         (relative)
     top5_norm         = test_top5       / max(test_top5)         (relative)
     cosine_norm       = test_mean_cosine                         (already [0,1])

  Step 3 — Weighted composite (non-collapsed runs only)
     composite_score = 0.35 * mrr_norm
                     + 0.25 * median_rank_norm
                     + 0.20 * top1_norm
                     + 0.15 * top5_norm
                     + 0.05 * cosine_norm

  Weight rationale:
     MRR (0.35)         — captures rank quality across all test samples; most
                          comprehensive single retrieval metric
     Median rank (0.25) — robust to outliers; reflects typical retrieval experience
     Top-1 (0.20)       — strictest correctness criterion; most interpretable in report
     Top-5 (0.15)       — practical retrieval success (answer in first 5 results)
     Cosine (0.05)      — tiebreaker only; excluded from collapse penalty because
                          near-constant outputs produce artificially high cosine

Usage
-----
  python src/pipelines/evaluation/type-b/final_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap ────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import PREDICTIONS_B, PREDICTIONS_B_COMMERCIAL_AI  # noqa: E402

CORPUS_SIZE = 10008  # Type-B full corpus sentence count (all sentences, not just test split)

# Weighting scheme for composite score (collapsed runs always receive 0.0)
W_MRR         = 0.35
W_MEDIAN_RANK = 0.25
W_TOP1        = 0.20
W_TOP5        = 0.15
W_COSINE      = 0.05

# Collapse threshold: colour_size_correct below this = collapsed
COLLAPSE_THRESHOLD = 0.95

# Experiment descriptions (for the report)
_DESCRIPTIONS: dict[str, str] = {
    'B0':              'TF-IDF + LSA (100-dim), MSE — statistical baseline',
    'E2a':             'SBERT all-MiniLM-L6-v2 (384-dim), frozen, CosineLoss',
    'E2b':             'SBERT fine-tuned on Type-B corpus (384-dim), CosineLoss',
    'E2e':             'TinyBERT mean-pool (312-dim), MSE',
    'E2f':             'TinyBERT [CLS] pooler (312-dim), MSE',
    'E2g':             'GloVe word-avg (300-dim), MSE',
    'E2h':             'Word2Vec Google News pretrained (300-dim), MSE',
    'E2i':             'Word2Vec skip-gram in-domain (100-dim), MSE',
    'E2k':             'TF-IDF weighted Word2Vec (100-dim), MSE',
    'LLM-gemini-lite': 'Gemini 2.0 Flash Lite → TinyBERT-mean retrieval (detailed prompt)',
    'LLM-gemini-lite-p2':   'Gemini 2.0 Flash Lite → TinyBERT-mean retrieval (minimal prompt)',
}

# ── LLM prediction files (model tag → run_id) ─────────────────────────────────
_LLM_PRED_FILES: dict[str, str] = {
    'openrouter_google-gemini-2.0-flash-lite-001_predictions.csv': 'LLM-gemini-lite',
    'openrouter_prompt2_google-gemini-2.0-flash-lite-001_predictions.csv': 'LLM-gemini-lite-p2',
}

_COLOURS = {'red', 'blue', 'green', 'yellow'}
_SIZES   = {'large', 'small'}


# ══════════════════════════════════════════════════════════════════════════════
# LLM result loader
# ══════════════════════════════════════════════════════════════════════════════

def _extract_colour_size(sentence: str) -> tuple[str | None, str | None]:
    tokens = str(sentence).lower().split()
    size   = next((t for t in tokens if t in _SIZES),   None)
    colour = next((t for t in tokens if t in _COLOURS), None)
    return size, colour

# llm result loader 
def _load_llm_rows() -> pd.DataFrame:
    """
    Read each LLM predictions CSV from PREDICTIONS_B_COMMERCIAL_AI, compute all
    metrics that match the test_results.csv schema, and return a DataFrame that
    can be concatenated directly with the CNN/embedding runs.

    Metrics computed per run:
      test_top1..5, test_mrr, test_mean_cosine, test_mean_rank, test_median_rank
      colour_size_correct  — fraction where pred_sentence matches true on colour+size
      top1_{n}d / mean_rank_{n}d / mean_cosine_{n}d  — breakdown by digit count (1-6)
    """
    rows: list[dict] = []

    for filename, run_id in _LLM_PRED_FILES.items():
        pred_path = PREDICTIONS_B_COMMERCIAL_AI / filename
        if not pred_path.exists():
            print(f'  [LLM] skipping {filename} — file not found')
            continue

        df = pd.read_csv(pred_path)

        
        ranks   = df['true_rank'].values
        n       = len(df)
        row: dict = {
            'run_id':            run_id,
            'cnn':               'none',
            'embedding':         'tinybert_mean_312d',
            'dim':               312,
            'loss_fn':           'N/A',
            'best_epoch':        float('nan'),
            'total_epochs':      float('nan'),
            'test_top1':         float(df['top_1_correct'].mean()),
            'test_top2':         float(df['top_2_correct'].mean()),
            'test_top3':         float(df['top_3_correct'].mean()),
            'test_top4':         float(df['top_4_correct'].mean()),
            'test_top5':         float(df['top_5_correct'].mean()),
            'test_mrr':          float(np.mean(1.0 / ranks)),
            'test_mean_cosine':  float(df['cosine_sim'].mean()),
            'test_mean_rank':    float(np.mean(ranks)),
            'test_median_rank':  float(np.median(ranks)),
        }

        # ── colour+size correctness ───────────────────────────────────────────
        cs_hits = sum(
            _extract_colour_size(p) == _extract_colour_size(t)
            for p, t in zip(df['pred_sentence'], df['true_sentence'])
        )
        row['colour_size_correct'] = cs_hits / n

        # ── per-digit breakdown ───────────────────────────────────────────────
        for nd in range(1, 7):
            sub = df[df['n_digits'] == nd]
            if len(sub) > 0:
                row[f'top1_{nd}d']       = float(sub['top_1_correct'].mean())
                row[f'mean_rank_{nd}d']  = float(sub['true_rank'].mean())
                row[f'mean_cosine_{nd}d'] = float(sub['cosine_sim'].mean())
            else:
                row[f'top1_{nd}d']       = float('nan')
                row[f'mean_rank_{nd}d']  = float('nan')
                row[f'mean_cosine_{nd}d'] = float('nan')

        rows.append(row)
        print(f'  [LLM] loaded {run_id}: top-1={row["test_top1"]:.4f}, '
              f'MRR={row["test_mrr"]:.4f}, colour_size={row["colour_size_correct"]:.4f}')

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

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

    # LLM runs are exempt from collapse detection (different retrieval paradigm)
    is_llm = out['run_id'].str.startswith('LLM-')
    out['collapsed'] = (~is_llm) & (out['colour_size_correct'] < COLLAPSE_THRESHOLD)

    # Normalise all metrics (use max across ALL runs for a fair relative scale)
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

    # Hard penalty: collapsed → 0.0
    out['composite_score'] = raw_score.where(~out['collapsed'], other=0.0)

    out = out.sort_values('composite_score', ascending=False).reset_index(drop=True)
    out.insert(0, 'rank', range(1, len(out) + 1))
    out['description'] = out['run_id'].map(_DESCRIPTIONS).fillna('')

    return out


def build_per_digit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape top1_{n}d and mean_rank_{n}d columns into a long-form table:
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
    print('  Type-B Stage-1 Final Embedding Ranking')
    print(f'  Composite score = {W_MRR}×MRR_norm + {W_MEDIAN_RANK}×MedianRank_norm'
          f' + {W_TOP1}×Top1_norm + {W_TOP5}×Top5_norm + {W_COSINE}×Cosine'
          f'  [collapsed → 0.0]')
    print(sep)
    print(f"  {'Rank':<5} {'Run':<6} {'Embedding':<22} {'top-1':>6} {'MRR':>7} "
          f"{'median_rank':>12} {'mean_rank':>10} {'CS_correct':>11} "
          f"{'collapsed':>9} {'score':>7}")
    print(f"  {'-'*86}")

    for _, row in ranked.iterrows():
        flag = '  !!!' if row['collapsed'] else ''
        print(
            f"  {int(row['rank']):<5} {row['run_id']:<6} {row['embedding']:<22} "
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


# main 
def main() -> None:
    test_results_csv = PREDICTIONS_B / 'test_results.csv'
    if not test_results_csv.exists():
        print(f'ERROR: test_results.csv not found at {test_results_csv}')
        print('Run evaluations first:')
        print('  python src/pipelines/evaluation/type-b/run_evals_stage1_b.py')
        return

    df = pd.read_csv(test_results_csv)
    if df.empty:
        print('ERROR: test_results.csv is empty.')
        return

    print(f'\nLoaded {len(df)} runs from {test_results_csv.name}')

    # ── Append LLM comparison rows ────────────────────────────────────────────
    llm_df = _load_llm_rows()
    if not llm_df.empty:
        df = pd.concat([df, llm_df], ignore_index=True)
        print(f'  → {len(llm_df)} LLM run(s) appended; total {len(df)} rows')

    # ── Compute ranking
    ranked     = compute_ranking(df)
    per_digit  = build_per_digit(ranked)

    # ── Console output 
    _print_ranking(ranked)
    _print_collapse_diagnosis(ranked)
    _print_top3(ranked)

    # ── Save CSVs 
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

    PREDICTIONS_B.mkdir(parents=True, exist_ok=True)

    final_csv = PREDICTIONS_B / 'final_ranking.csv'
    ranked[out_cols].to_csv(final_csv, index=False)
    print(f'  [saved] {final_csv}')

    digit_csv = PREDICTIONS_B / 'final_per_digit.csv'
    per_digit.to_csv(digit_csv, index=False)
    print(f'  [saved] {digit_csv}')

    # ── Leaderboard CSV (report-ready, reduced columns) ────────────────────
    leaderboard_cols = [
        'rank', 'run_id', 'cnn', 'embedding', 'dim', 'loss_fn',
        'test_top1', 'test_top5', 'test_mrr',
        'test_median_rank', 'collapsed', 'composite_score',
    ]
    leaderboard_cols = [c for c in leaderboard_cols if c in ranked.columns]
    lb = ranked[leaderboard_cols].copy()

    # Round floats for readability
    for col in ('test_top1', 'test_top5', 'test_mrr', 'composite_score'):
        if col in lb.columns:
            lb[col] = lb[col].round(4)

    lb_csv = PREDICTIONS_B / 'leaderboard.csv'
    lb.to_csv(lb_csv, index=False)
    print(f'  [saved] {lb_csv}')

    print(f'\nDone.\n')


if __name__ == '__main__':
    main()
