from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import (  # noqa: E402
    PREDICTIONS_B,
    PREDICTIONS_B_S2,
    PREDICTIONS_B_COMMERCIAL_AI,
    FIGURES_EVAL_B,
)

# Step 1 — Collapse hard filter
# Step 2 — Normalise each metric to [0, 1] across all runs
# Step 3 — Weighted composite (non-collapsed runs only)

FIGURES_LLM = FIGURES_EVAL_B / 'llm'

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
    # Stage 1 — embedding axis (cnn_1layer fixed)
    'B0':              'TF-IDF + LSA (100-dim), MSE — statistical baseline',
    'E2a':             'SBERT all-MiniLM-L6-v2 (384-dim), frozen, CosineLoss',
    'E2b':             'SBERT fine-tuned on Type-B corpus (384-dim), CosineLoss',
    'E2e':             'TinyBERT mean-pool (312-dim), MSE',
    'E2f':             'TinyBERT [CLS] pooler (312-dim), MSE',
    'E2g':             'GloVe word-avg (300-dim), MSE',
    'E2h':             'Word2Vec Google News pretrained (300-dim), MSE',
    'E2i':             'Word2Vec skip-gram in-domain (100-dim), MSE',
    'E2k':             'TF-IDF weighted Word2Vec (100-dim), MSE',
    # Stage 2 — architecture axis (tinybert_mean fixed)
    'S2a':             'Stage-2 cnn_1layer + TinyBERT-mean (312-dim), MSE',
    'S2b':             'Stage-2 cnn_3layer + TinyBERT-mean (312-dim), MSE',
    'S2c':             'Stage-2 ResNet18-pretrained + TinyBERT-mean (312-dim), MSE',
    # LLM baseline
    'LLM-gemini-lite': 'Gemini 2.0 Flash Lite → TinyBERT-mean retrieval (detailed prompt)',
    'LLM-gemini-lite-p2':   'Gemini 2.0 Flash Lite → TinyBERT-mean retrieval (minimal prompt)',
}

# LLM prediction files (model tag → run_id) 
_LLM_PRED_FILES: dict[str, str] = {
    'openrouter_google-gemini-2.0-flash-lite-001_predictions.csv': 'LLM-gemini-lite',
    'openrouter_prompt2_google-gemini-2.0-flash-lite-001_predictions.csv': 'LLM-gemini-lite-p2',
}

_COLOURS = {'red', 'blue', 'green', 'yellow'}
_SIZES   = {'large', 'small'}


# llm result loader 
def _extract_colour_size(sentence: str) -> tuple[str | None, str | None]:
    tokens = str(sentence).lower().split()
    size   = next((t for t in tokens if t in _SIZES),   None)
    colour = next((t for t in tokens if t in _COLOURS), None)
    return size, colour

def _load_llm_rows() -> pd.DataFrame:
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

        # color size correctness 
        cs_hits = sum(
            _extract_colour_size(p) == _extract_colour_size(t)
            for p, t in zip(df['pred_sentence'], df['true_sentence'])
        )
        row['colour_size_correct'] = cs_hits / n

        #  per-digit breakdown 
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

# scoring logic 
def compute_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalised score components and composite_score to the results dataframe.

    Normalisation:
      mrr_norm         = test_mrr        / max_cnn(test_mrr)        ∈ [0, 1] for CNN; may exceed 1 for LLM
      median_rank_norm = 1 - (test_median_rank / CORPUS_SIZE)       ∈ [0, 1]
      top1_norm        = test_top1       / max_cnn(test_top1)       ∈ [0, 1] for CNN; may exceed 1 for LLM
      top5_norm        = test_top5       / max_cnn(test_top5)       ∈ [0, 1] for CNN; may exceed 1 for LLM
      cosine_norm      = test_mean_cosine                           ∈ [0, 1]

    max_cnn = max over CNN-only runs; LLM runs are placed on the same scale
    but their norms can exceed 1.0 (showing how much better than the best CNN).

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

    # Normalise using CNN-only max so LLM does not inflate the denominator.
    # LLM scores are placed on the same CNN-anchored scale (may exceed 1.0).
    def _safe_norm(series: pd.Series, mask: pd.Series) -> pd.Series:
        mx = series[mask].max()
        return series / mx if mx > 0 else series * 0.0

    cnn_mask = ~is_llm
    out['mrr_norm']         = _safe_norm(out['test_mrr'],          cnn_mask)
    out['median_rank_norm'] = 1.0 - (out['test_median_rank'] / CORPUS_SIZE)
    out['top1_norm']        = _safe_norm(out['test_top1'],         cnn_mask)
    out['top5_norm']        = _safe_norm(out['test_top5'],         cnn_mask)
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


# console output 
def _print_ranking(ranked: pd.DataFrame) -> None:
    sep = '=' * 90
    print(f'\n{sep}')
    print('  Type-B Stage-1 + Stage-2 Final Ranking')
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



# llm confusion matrix 
def _plot_llm_confusion(figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _cmap = LinearSegmentedColormap.from_list('wb', ['#ffffff', '#1f77b4'])

    for filename, run_id in _LLM_PRED_FILES.items():
        pred_path = PREDICTIONS_B_COMMERCIAL_AI / filename
        if not pred_path.exists():
            continue

        df = pd.read_csv(pred_path)

        def extract(s: str) -> tuple[str, str]:
            tokens = str(s).lower().split()
            size   = next((t for t in tokens if t in _SIZES),   'unknown')
            colour = next((t for t in tokens if t in _COLOURS), 'unknown')
            return size, colour

        df['true_size'],  df['true_colour']  = zip(*df['true_sentence'].map(extract))
        df['llm_size'],   df['llm_colour']   = zip(*df['llm_raw'].map(extract))

        n = len(df)
        size_acc   = (df['llm_size']   == df['true_size']).mean()
        colour_acc = (df['llm_colour'] == df['true_colour']).mean()
        both_acc   = ((df['llm_size'] == df['true_size']) &
                      (df['llm_colour'] == df['true_colour'])).mean()

        # 1. Size confusion matrix 
        size_labels = sorted(_SIZES)
        size_cm = pd.crosstab(
            df['true_size'], df['llm_size'],
            rownames=['True'], colnames=['Predicted'],
        ).reindex(index=size_labels, columns=size_labels, fill_value=0)

        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(size_cm.values, cmap=_cmap, aspect='auto')
        ax.set_xticks(range(len(size_labels))); ax.set_xticklabels(size_labels, fontsize=11)
        ax.set_yticks(range(len(size_labels))); ax.set_yticklabels(size_labels, fontsize=11)
        ax.set_xlabel('Predicted size', fontsize=12)
        ax.set_ylabel('True size',      fontsize=12)
        ax.set_title(f'Size Confusion Matrix\n{run_id}  (acc={size_acc:.2%})', fontsize=12)
        for i in range(len(size_labels)):
            for j in range(len(size_labels)):
                v = int(size_cm.values[i, j])
                pct = v / size_cm.values[i].sum() if size_cm.values[i].sum() > 0 else 0
                ax.text(j, i, f'{v}\n({pct:.0%})',
                        ha='center', va='center', fontsize=10,
                        color='white' if size_cm.values[i, j] > size_cm.values.max() * 0.5 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = figures_dir / f'size_confusion_{run_id}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [saved] {out}')

        #  2. Colour confusion matrix 
        colour_labels = sorted(_COLOURS)
        colour_cm = pd.crosstab(
            df['true_colour'], df['llm_colour'],
            rownames=['True'], colnames=['Predicted'],
        ).reindex(index=colour_labels, columns=colour_labels, fill_value=0)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(colour_cm.values, cmap=_cmap, aspect='auto')
        ax.set_xticks(range(len(colour_labels))); ax.set_xticklabels(colour_labels, fontsize=11, rotation=20)
        ax.set_yticks(range(len(colour_labels))); ax.set_yticklabels(colour_labels, fontsize=11)
        ax.set_xlabel('Predicted colour', fontsize=12)
        ax.set_ylabel('True colour',      fontsize=12)
        ax.set_title(f'Colour Confusion Matrix\n{run_id}  (acc={colour_acc:.2%})', fontsize=12)
        for i in range(len(colour_labels)):
            for j in range(len(colour_labels)):
                v = int(colour_cm.values[i, j])
                pct = v / colour_cm.values[i].sum() if colour_cm.values[i].sum() > 0 else 0
                ax.text(j, i, f'{v}\n({pct:.0%})',
                        ha='center', va='center', fontsize=9,
                        color='white' if colour_cm.values[i, j] > colour_cm.values.max() * 0.5 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = figures_dir / f'colour_confusion_{run_id}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [saved] {out}')

        # 3. Attribute accuracy bar chart 
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            ['Size', 'Colour', 'Both'],
            [size_acc, colour_acc, both_acc],
            color=['#4C72B0', '#DD8452', '#55A868'],
            edgecolor='black', linewidth=0.7, width=0.5,
        )
        for bar, val in zip(bars, [size_acc, colour_acc, both_acc]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'LLM Attribute Accuracy\n{run_id}  (n={n})', fontsize=12)
        ax.axhline(1.0, color='grey', linewidth=0.8, linestyle='--')
        fig.tight_layout()
        out = figures_dir / f'attribute_accuracy_{run_id}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  [saved] {out}')


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

    print(f'\nLoaded {len(df)} Stage-1 runs from {test_results_csv.name}')

    # Append Stage-2 non-normalised results (if available)
    s2_csv = PREDICTIONS_B_S2 / 'test_results_s2.csv'
    if s2_csv.exists():
        s2_df = pd.read_csv(s2_csv)
        if not s2_df.empty:
            df = pd.concat([df, s2_df], ignore_index=True)
            print(f'  → {len(s2_df)} Stage-2 run(s) appended from {s2_csv.name}')
    else:
        print(f'  [info] Stage-2 results not found at {s2_csv}')
        print('         Run: python src/pipelines/evaluation/type-b/run_evals_stage2_b.py '
              '--variant non-normalised')

    # Append LLM comparison rows
    llm_df = _load_llm_rows()
    if not llm_df.empty:
        df = pd.concat([df, llm_df], ignore_index=True)
        print(f'  → {len(llm_df)} LLM run(s) appended; total {len(df)} rows')

    #  Compute ranking
    ranked     = compute_ranking(df)
    per_digit  = build_per_digit(ranked)

    #  Console output 
    _print_ranking(ranked)
    _print_collapse_diagnosis(ranked)
    _print_top3(ranked)

    #  Save CSVs 
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

    #  Leaderboard CSV (report-ready, reduced columns) 
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

    #  LLM confusion matrix plots 
    _plot_llm_confusion(FIGURES_LLM)

    print(f'\nDone.\n')


if __name__ == '__main__':
    main()
