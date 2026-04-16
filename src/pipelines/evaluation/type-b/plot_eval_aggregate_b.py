"""
src/pipelines/evaluation/type-b/plot_eval_aggregate_b.py

Unified aggregate visualiser for ALL Type-B experiments (replaces
plot_eval_aggregate_normed_b.py, which is no longer needed).

Reads per-sample prediction CSVs and produces cross-run comparison figures.
The correct predictions directory is resolved automatically from the run ID.

--stage choices and their prediction sources
--------------------------------------------
  s1          → prediction/                     figures/evaluation/
  s1-normed   → prediction-normalised/          figures/evaluation/normalised/
  s2-non-normed → prediction-s2/               figures/evaluation/s2/non-normalised/
  s2-normed   → prediction-s2-normalised/      figures/evaluation/s2/normalised/
  all         → all of the above               figures/evaluation/

Usage
-----
  python src/pipelines/evaluation/type-b/plot_eval_aggregate_b.py
  python src/pipelines/evaluation/type-b/plot_eval_aggregate_b.py --stage s1-normed
  python src/pipelines/evaluation/type-b/plot_eval_aggregate_b.py --runs B0 E2an S2a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import (  # noqa: E402
    PREDICTIONS_B,
    PREDICTIONS_B_NORMED,
    PREDICTIONS_B_S2,
    PREDICTIONS_B_S2_NORMED,
    FIGURES_DIR,
    FIGURES_EVAL_NORM_B,
    FIGURES_EVAL_B_S2_NON_NORMED,
    FIGURES_EVAL_B_S2_NORMED,
)

FIGURES_EVAL_B = FIGURES_DIR / 'evaluation'

# ── Experiment registry ────────────────────────────────────────────────────────
# Stage 1 — embedding axis, non-normalised (cnn_1layer fixed)
_EXPERIMENTS_S1: dict[str, str] = {
    'B0':  'tfidf_lsa',
    'E2a': 'sbert',
    'E2b': 'sbert_finetuned',
    'E2e': 'tinybert_mean',
    'E2f': 'tinybert_pooler',
    'E2g': 'glove',
    'E2h': 'word2vec_pretrained',
    'E2i': 'word2vec_skipgram',
    'E2k': 'tfidf_w2v',
}

# Stage 1 — embedding axis, normalised targets (cnn_1layer fixed)
_EXPERIMENTS_S1_NORMED: dict[str, str] = {
    'B0n':  'tfidf_lsa_normed',
    'E2an': 'sbert_normed',
    'E2bn': 'sbert_finetuned_normed',
    'E2en': 'tinybert_mean_normed',
    'E2fn': 'tinybert_pooler_normed',
    'E2gn': 'glove_normed',
    'E2hn': 'word2vec_pretrained_normed',
    'E2in': 'word2vec_skipgram_normed',
    'E2kn': 'tfidf_w2v_normed',
    'E2ln': 'bert_mean_normed',
    'E2mn': 'bert_pooler_normed',
}

# Stage 2 — architecture axis (tinybert_mean fixed), non-normalised
_EXPERIMENTS_S2_NON_NORMED: dict[str, str] = {
    'S2a': 'cnn_1layer / tinybert_mean / MSE',
    'S2b': 'cnn_3layer / tinybert_mean / MSE',
    'S2c': 'resnet18_pt / tinybert_mean / MSE',
}

# Stage 2 — architecture axis, normalised targets
_EXPERIMENTS_S2_NORMED: dict[str, str] = {
    'S2an': 'cnn_1layer / tinybert_mean_normed / MSE',
    'S2bn': 'cnn_3layer / tinybert_mean_normed / MSE',
    'S2cn': 'resnet18_pt / tinybert_mean_normed / MSE',
    'S2ad': 'cnn_1layer / tinybert_mean_normed / Combined',
    'S2bd': 'cnn_3layer / tinybert_mean_normed / Combined',
    'S2cd': 'resnet18_pt / tinybert_mean_normed / Combined',
}

# Combined registry (all stages / variants)
_EXPERIMENTS: dict[str, str] = {
    **_EXPERIMENTS_S1,
    **_EXPERIMENTS_S1_NORMED,
    **_EXPERIMENTS_S2_NON_NORMED,
    **_EXPERIMENTS_S2_NORMED,
}

_PALETTE: dict[str, str] = {
    # Stage 1 non-normed
    'B0':  '#607D8B',
    'E2a': '#2196F3',
    'E2b': '#00897B',
    'E2e': '#4CAF50',
    'E2f': '#F48FB1',
    'E2g': '#FF9800',
    'E2h': '#FF5722',
    'E2i': '#9C27B0',
    'E2k': '#E91E63',
    # Stage 1 normed (lighter shades of S1)
    'B0n':  '#90A4AE',
    'E2an': '#64B5F6',
    'E2bn': '#4DB6AC',
    'E2en': '#81C784',
    'E2fn': '#F8BBD9',
    'E2gn': '#FFB74D',
    'E2hn': '#FF8A65',
    'E2in': '#CE93D8',
    'E2kn': '#F48FB1',
    'E2ln': '#80CBC4',
    'E2mn': '#A5D6A7',
    # Stage 2 non-normed
    'S2a': '#1565C0',
    'S2b': '#2E7D32',
    'S2c': '#B71C1C',
    # Stage 2 normed
    'S2an': '#42A5F5',
    'S2bn': '#66BB6A',
    'S2cn': '#EF5350',
    'S2ad': '#7E57C2',
    'S2bd': '#26C6DA',
    'S2cd': '#FFA726',
}

# Corpus size for Type-B (used as rank axis upper bound)
_CORPUS_SIZE = 1001

# Mapping: stage name → (experiment dict, figures output dir)
_STAGE_CONFIG: dict[str, tuple[dict, Path]] = {
    's1':           (_EXPERIMENTS_S1,           FIGURES_EVAL_B),
    's1-normed':    (_EXPERIMENTS_S1_NORMED,    FIGURES_EVAL_NORM_B),
    's2-non-normed': (_EXPERIMENTS_S2_NON_NORMED, FIGURES_EVAL_B_S2_NON_NORMED),
    's2-normed':    (_EXPERIMENTS_S2_NORMED,    FIGURES_EVAL_B_S2_NORMED),
}


def _pred_dir_for(run_id: str) -> Path:
    if run_id in _EXPERIMENTS_S1_NORMED:
        return PREDICTIONS_B_NORMED
    if run_id in _EXPERIMENTS_S2_NON_NORMED:
        return PREDICTIONS_B_S2
    if run_id in _EXPERIMENTS_S2_NORMED:
        return PREDICTIONS_B_S2_NORMED
    return PREDICTIONS_B


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_predictions(run_ids: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load per-sample prediction CSVs for each run.
    Automatically resolves the correct predictions directory based on the run ID.
    Expected columns: true_sentence, pred_sentence, n_digits,
                      true_rank, cosine_sim, top_1_correct, …
    """
    loaded: dict[str, pd.DataFrame] = {}
    for run_id in run_ids:
        pred_dir = _pred_dir_for(run_id)
        path     = pred_dir / f'{run_id.lower()}_test_predictions.csv'
        if not path.exists():
            print(f'  [skip] {run_id} — prediction CSV not found: {path}')
            if run_id.startswith('S2'):
                runner = 'run_evals_stage2_b.py'
            elif run_id.endswith('n') and run_id not in _EXPERIMENTS_S1:
                runner = 'run_evals_stage1_normed_b.py'
            else:
                runner = 'run_evals_stage1_b.py'
            print(f'         Run evaluation first: '
                  f'python src/pipelines/evaluation/type-b/{runner} --runs {run_id}')
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f'  [skip] {run_id} — prediction CSV is empty')
            continue
        loaded[run_id] = df
        print(f'  [load] {run_id} ← {path.relative_to(path.parents[3])}  ({len(df)} samples)')
    return loaded


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_rank_cdf(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Rank CDF curves — for each run, plot P(rank ≤ k) as a function of k.

    This is the most informative aggregate view:
    - y-axis: fraction of test images retrieved at rank ≤ k
    - x-axis: rank k  (1 … corpus_size)
    - All runs overlaid, with B0 dashed as baseline reference

    Interpretation: a curve that rises steeply near k=1 is best.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    k_vals = np.arange(1, _CORPUS_SIZE + 1)

    for run_id, df in preds.items():
        ranks  = df['true_rank'].values
        cdf    = np.array([(ranks <= k).mean() for k in k_vals])
        colour = _PALETTE.get(run_id, '#333333')
        emb    = _EXPERIMENTS.get(run_id, run_id)
        ls     = '--' if run_id == 'B0' else '-'
        lw     = 1.4  if run_id == 'B0' else 1.8
        ax.plot(k_vals, cdf, ls, color=colour, linewidth=lw,
                label=f'{run_id} ({emb})')

    # Mark Top-1 / Top-5 with vertical reference lines
    for k, style in [(1, ':'), (5, '--')]:
        ax.axvline(k, color='black', linewidth=0.8, linestyle=style, alpha=0.5)
        ax.text(k + 5, 0.02, f'k={k}', fontsize=8, color='black', alpha=0.6)

    ax.set_xlabel('Rank k')
    ax.set_ylabel('Fraction of test images with rank ≤ k')
    ax.set_title('Retrieval Rank CDF — All Embeddings (cnn_1layer, test set)')
    ax.set_xlim(0, min(200, _CORPUS_SIZE))   # zoom in: first 200 ranks most informative
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    out = out_dir / 'rank_cdf.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_rank_boxplot(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Box plot of rank distributions — one box per run, sorted by median rank.

    Shows: median, IQR, whiskers (1.5×IQR), and outliers.
    Lower rank = better → boxes near the bottom are best.
    """
    # Sort runs by median rank ascending
    order = sorted(preds, key=lambda r: preds[r]['true_rank'].median())

    data    = [preds[r]['true_rank'].values for r in order]
    labels  = [f"{r}\n({_EXPERIMENTS.get(r, r)})" for r in order]
    colours = [_PALETTE.get(r, '#333333') for r in order]

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(order)), 6))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='.', markersize=2, alpha=0.3),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        notch=False,
    )

    for patch, colour in zip(bp['boxes'], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Rank (lower = better)')
    ax.set_title('Rank Distribution by Embedding — Test Set\n'
                 '(sorted by median rank ↑ = better)')
    ax.invert_yaxis()   # lower rank at top
    ax.grid(True, axis='y', alpha=0.3)

    out = out_dir / 'rank_boxplot.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_cosine_sim_kde(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Subplot grid of cosine similarity KDE — one panel per run, independent y-axis.

    Overlaying all runs on one axis collapses the view because spike embeddings
    (cosine ≈ 1.0) push the y-scale to ~400, making all other distributions
    invisible. A per-run subplot with its own y-axis reveals each distribution.

    Each panel also shows: mean (solid line), median (dashed line).
    """
    import math

    n     = len(preds)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        'Cosine Similarity Distribution — Each Embedding (test set)\n'
        'Note: spike near 1.0 = embedding collapse (high cosine, bad rank)',
        fontsize=12,
    )

    x_grid = np.linspace(-0.2, 1.05, 600)

    for ax_idx, (run_id, df) in enumerate(preds.items()):
        row, col = divmod(ax_idx, ncols)
        ax       = axes[row][col]
        colour   = _PALETTE.get(run_id, '#333333')
        emb      = _EXPERIMENTS.get(run_id, run_id)
        sims     = df['cosine_sim'].dropna().values

        if len(sims) < 2:
            ax.set_visible(False)
            continue

        try:
            kde  = gaussian_kde(sims, bw_method='scott')
            dens = kde(x_grid)
        except Exception:
            ax.set_visible(False)
            continue

        ax.plot(x_grid, dens, '-', color=colour, linewidth=1.8)
        ax.fill_between(x_grid, dens, alpha=0.18, color=colour)

        # Mean and median reference lines
        mean_val   = float(np.mean(sims))
        median_val = float(np.median(sims))
        ax.axvline(mean_val,   color='black',  linewidth=1.0,
                   linestyle='-',  label=f'mean={mean_val:.3f}')
        ax.axvline(median_val, color='black',  linewidth=1.0,
                   linestyle='--', label=f'med={median_val:.3f}')

        ax.set_title(f'{run_id}  ({emb})', fontsize=9, color=colour)
        ax.set_xlabel('Cosine similarity', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 1.05)

    # Hide unused subplots
    for ax_idx in range(n, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out = out_dir / 'cosine_sim_kde.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_top1_bar(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Horizontal bar chart of Top-1 accuracy, sorted descending."""
    order = sorted(
        preds,
        key=lambda r: (preds[r]['true_rank'] == 1).mean(),
        reverse=True,
    )
    vals    = [(preds[r]['true_rank'] == 1).mean() for r in order]
    labels  = [f"{r}  ({_EXPERIMENTS.get(r, r)})" for r in order]
    colours = [_PALETTE.get(r, '#333333') for r in order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(order))))
    bars = ax.barh(labels, vals, color=colours, alpha=0.85)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    ax.set_xlabel('Top-1 Accuracy')
    ax.set_title('Top-1 Retrieval Accuracy — All Embeddings (test set)')
    ax.set_xlim(0, max(vals) * 1.3 + 0.001)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    out = out_dir / 'top1_bar.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_mrr_bar(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Horizontal bar chart of MRR, sorted descending."""
    def _mrr(run_id: str) -> float:
        return float((1.0 / preds[run_id]['true_rank']).mean())

    order   = sorted(preds, key=_mrr, reverse=True)
    vals    = [_mrr(r) for r in order]
    labels  = [f"{r}  ({_EXPERIMENTS.get(r, r)})" for r in order]
    colours = [_PALETTE.get(r, '#333333') for r in order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(order))))
    bars = ax.barh(labels, vals, color=colours, alpha=0.85)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.0001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    ax.set_xlabel('MRR (Mean Reciprocal Rank)')
    ax.set_title('MRR — All Embeddings (test set)')
    ax.set_xlim(0, max(vals) * 1.3 + 0.001)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    out = out_dir / 'mrr_bar.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_rank_cdf_by_ndigits(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Rank CDF curves faceted by number of digits (1–6).
    6 subplots in a 2×3 grid — one per digit count.
    Helps identify whether embeddings struggle with longer numbers.
    """
    digit_counts = list(range(1, 7))
    ncols = 3
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), sharey=True)
    fig.suptitle('Rank CDF by Number of Digits — All Embeddings (test set)',
                 fontsize=12)

    k_vals = np.arange(1, _CORPUS_SIZE + 1)

    for ax_idx, nd in enumerate(digit_counts):
        row, col = divmod(ax_idx, ncols)
        ax       = axes[row][col]

        for run_id, df in preds.items():
            sub    = df[df['n_digits'] == nd]
            if sub.empty:
                continue
            ranks  = sub['true_rank'].values
            cdf    = np.array([(ranks <= k).mean() for k in k_vals])
            colour = _PALETTE.get(run_id, '#333333')
            ls     = '--' if run_id == 'B0' else '-'
            ax.plot(k_vals, cdf, ls, color=colour, linewidth=1.4,
                    label=run_id if ax_idx == 0 else '_')

        first_df   = next(iter(preds.values()))
        n_samples  = int((first_df['n_digits'] == nd).sum())
        ax.set_title(f'{nd}-digit numbers  (n={n_samples})', fontsize=9)
        ax.set_xlabel('Rank k', fontsize=8)
        ax.set_ylabel('P(rank ≤ k)', fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Shared legend on the first subplot
    handles = [
        plt.Line2D([0], [0], color=_PALETTE.get(r, '#333333'),
                   linewidth=1.6, linestyle='--' if r == 'B0' else '-',
                   label=f'{r} ({_EXPERIMENTS.get(r, r)})')
        for r in preds
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    out = out_dir / 'rank_cdf_by_ndigits.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate evaluation plots for Type-B experiments'
    )
    parser.add_argument(
        '--stage',
        choices=['s1', 's1-normed', 's2-non-normed', 's2-normed', 'all'],
        default='s1',
        help=(
            's1           = stage-1 non-normalised (default) | '
            's1-normed    = stage-1 normalised | '
            's2-non-normed = stage-2 non-normalised | '
            's2-normed    = stage-2 normalised | '
            'all          = every available run'
        ),
    )
    parser.add_argument(
        '--runs', nargs='+', default=None,
        metavar='RUN_ID',
        help='Explicit run IDs to include (overrides --stage). E.g. --runs B0 E2a S2a',
    )
    parser.add_argument(
        '--tsne', action='store_true',
        help='Also generate t-SNE corpus visualisation (~5 min for 10,008 points)',
    )
    parser.add_argument(
        '--tsne-embeddings', nargs='+',
        default=['sbert', 'sbert_finetuned', 'tinybert_mean', 'tfidf_lsa'],
        metavar='EMBEDDING',
        help='Embedding names to include in t-SNE plot',
    )
    parser.add_argument(
        '--tsne-colour-by', default='all', choices=['n_digits', 'colour', 'size', 'all'],
        help='Colour t-SNE points by n_digits / colour / size / all (default: all)',
    )
    args = parser.parse_args()

    # Resolve which runs to plot
    if args.runs:
        run_ids = args.runs
        unknown = [r for r in run_ids if r not in _EXPERIMENTS]
        if unknown:
            parser.error(f'Unknown run IDs: {unknown}. Available: {sorted(_EXPERIMENTS)}')
        # Pick output dir from the first recognised stage bucket
        if any(r in _EXPERIMENTS_S2_NORMED for r in run_ids):
            out_dir = FIGURES_EVAL_B_S2_NORMED
        elif any(r in _EXPERIMENTS_S2_NON_NORMED for r in run_ids):
            out_dir = FIGURES_EVAL_B_S2_NON_NORMED
        elif any(r in _EXPERIMENTS_S1_NORMED for r in run_ids):
            out_dir = FIGURES_EVAL_NORM_B
        else:
            out_dir = FIGURES_EVAL_B
    elif args.stage == 'all':
        run_ids = list(_EXPERIMENTS)
        out_dir = FIGURES_EVAL_B
    else:
        run_ids = list(_STAGE_CONFIG[args.stage][0])
        out_dir  = _STAGE_CONFIG[args.stage][1]

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nType-B Aggregate Evaluation Plots')
    print(f'Stage  : {args.stage}')
    print(f'Runs   : {run_ids}')
    print(f'Output : {out_dir}\n')

    preds = _load_predictions(run_ids)
    if not preds:
        print('No prediction CSVs found. Run evaluations first:')
        hints = {
            's1':            'run_evals_stage1_b.py',
            's1-normed':     'run_evals_stage1_normed_b.py',
            's2-non-normed': 'run_evals_stage2_b.py --variant non-normalised',
            's2-normed':     'run_evals_stage2_b.py --variant normalised',
            'all':           'run_evals_stage1_b.py  /  run_evals_stage1_normed_b.py  /  run_evals_stage2_b.py',
        }
        print(f'  python src/pipelines/evaluation/type-b/{hints.get(args.stage, "")}')
        return

    print('\n── Aggregate plots ──')
    plot_rank_cdf(preds, out_dir)
    plot_rank_boxplot(preds, out_dir)
    plot_cosine_sim_kde(preds, out_dir)
    plot_top1_bar(preds, out_dir)
    plot_mrr_bar(preds, out_dir)
    plot_rank_cdf_by_ndigits(preds, out_dir)

    if args.tsne:
        import sys as _sys
        _eval_dir = Path(__file__).resolve().parent
        if str(_eval_dir) not in _sys.path:
            _sys.path.insert(0, str(_eval_dir))
        from eval_metrics_b import plot_tsne_corpus
        colour_by_list = (
            ['n_digits', 'colour', 'size']
            if args.tsne_colour_by == 'all'
            else [args.tsne_colour_by]
        )
        print(f'\n── t-SNE corpus visualisation (~5 min per colour axis) ──')
        print(f'   Embeddings : {args.tsne_embeddings}')
        print(f'   Colour by  : {colour_by_list}')
        for colour_by in colour_by_list:
            plot_tsne_corpus(
                embedding_names=args.tsne_embeddings,
                colour_by=colour_by,
                figures_dir=out_dir,
            )

    print(f'\nDone → {out_dir}')


if __name__ == '__main__':
    main()
