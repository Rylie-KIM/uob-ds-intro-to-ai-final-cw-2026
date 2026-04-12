"""
src/pipelines/evaluation/type-b/plot_eval_aggregate_normed_b.py

Aggregate evaluation visualiser for Type-B normalised-embedding experiments.

Mirrors plot_eval_aggregate_b.py but reads from:
  metrics/type-b/prediction-normalised/

and writes figures to:
  figures/type-b/evaluation/normalised/

Plots produced:
  rank_cdf.png              — Rank CDF curves for all normed runs (B0n dashed as baseline)
  rank_boxplot.png          — Box plot of rank distributions per run
  cosine_sim_kde.png        — Per-run cosine similarity KDE (subplot grid)
  top1_bar.png              — Top-1 accuracy bar chart (sorted)
  mrr_bar.png               — MRR bar chart (sorted)
  rank_cdf_by_ndigits.png   — Rank CDF faceted by number of digits (1–6)
  tsne_corpus_{axis}.png    — t-SNE of corpus embeddings (optional, --tsne flag)

Usage
-----
  # All normalised runs
  python src/pipelines/evaluation/type-b/plot_eval_aggregate_normed_b.py
  
  # Also generate t-SNE corpus visualisation
  python src/pipelines/evaluation/type-b/plot_eval_aggregate_normed_b.py --tsne

  # t-SNE with specific embeddings and colour axis
  python src/pipelines/evaluation/type-b/plot_eval_aggregate_normed_b.py --tsne --tsne-embeddings sbert tfidf_lsa --tsne-colour-by colour
"""

from __future__ import annotations

import argparse
import math
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

from src.config.paths import PREDICTIONS_B_NORMED, FIGURES_EVAL_NORM_B  # noqa: E402

# ── Experiment registry ────────────────────────────────────────────────────────
_EXPERIMENTS: dict[str, str] = {
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

_PALETTE: dict[str, str] = {
    'B0n':  '#90A4AE',   # lighter blue-grey
    'E2an': '#64B5F6',   # lighter blue
    'E2bn': '#4DB6AC',   # lighter teal
    'E2en': '#81C784',   # lighter green
    'E2fn': '#F8BBD9',   # lighter pink
    'E2gn': '#FFB74D',   # lighter orange
    'E2hn': '#FF8A65',   # lighter deep-orange
    'E2in': '#CE93D8',   # lighter purple
    'E2kn': '#F48FB1',   # lighter pink
    'E2ln': '#80CBC4',   # teal
    'E2mn': '#A5D6A7',   # green
}

# Corpus size for Type-B
_CORPUS_SIZE = 1001


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_predictions(run_ids: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load per-sample prediction CSVs from PREDICTIONS_B_NORMED.
    Expected columns: true_sentence, pred_sentence, n_digits,
                      true_rank, cosine_sim, top_1_correct, …
    """
    loaded: dict[str, pd.DataFrame] = {}
    for run_id in run_ids:
        path = PREDICTIONS_B_NORMED / f'{run_id.lower()}_test_predictions.csv'
        if not path.exists():
            print(f'  [skip] {run_id} — prediction CSV not found: {path.name}')
            print(f'         Run evaluation first: '
                  f'python src/pipelines/evaluation/type-b/run_evals_stage1_normed_b.py '
                  f'--runs {run_id}')
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f'  [skip] {run_id} — prediction CSV is empty')
            continue
        loaded[run_id] = df
        print(f'  [load] {run_id} ← {path.name}  ({len(df)} samples)')
    return loaded


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_rank_cdf(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Rank CDF curves — P(rank ≤ k) as a function of k for each normed run.
    B0n is shown dashed as the normalised baseline reference.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    k_vals = np.arange(1, _CORPUS_SIZE + 1)

    for run_id, df in preds.items():
        ranks  = df['true_rank'].values
        cdf    = np.array([(ranks <= k).mean() for k in k_vals])
        colour = _PALETTE.get(run_id, '#333333')
        emb    = _EXPERIMENTS.get(run_id, run_id)
        ls     = '--' if run_id == 'B0n' else '-'
        lw     = 1.4  if run_id == 'B0n' else 1.8
        ax.plot(k_vals, cdf, ls, color=colour, linewidth=lw,
                label=f'{run_id} ({emb})')

    for k, style in [(1, ':'), (5, '--')]:
        ax.axvline(k, color='black', linewidth=0.8, linestyle=style, alpha=0.5)
        ax.text(k + 5, 0.02, f'k={k}', fontsize=8, color='black', alpha=0.6)

    ax.set_xlabel('Rank k')
    ax.set_ylabel('Fraction of test images with rank ≤ k')
    ax.set_title('Retrieval Rank CDF — Normalised Embeddings (cnn_1layer, L2-normed, test set)')
    ax.set_xlim(0, min(200, _CORPUS_SIZE))
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    out = out_dir / 'rank_cdf.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_rank_boxplot(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Box plot of rank distributions — one box per normed run, sorted by median rank.
    Lower rank = better → boxes near the bottom are best.
    """
    order   = sorted(preds, key=lambda r: preds[r]['true_rank'].median())
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
    ax.set_title('Rank Distribution — Normalised Embeddings (test set)\n'
                 '(sorted by median rank ↑ = better)')
    ax.invert_yaxis()
    ax.grid(True, axis='y', alpha=0.3)

    out = out_dir / 'rank_boxplot.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_cosine_sim_kde(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Subplot grid of cosine similarity KDE — one panel per normed run.
    Each panel shows the distribution of cosine(pred, true) with mean/median lines.
    """
    n     = len(preds)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        'Cosine Similarity Distribution — Normalised Embeddings (test set)\n'
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

        mean_val   = float(np.mean(sims))
        median_val = float(np.median(sims))
        ax.axvline(mean_val,   color='black', linewidth=1.0,
                   linestyle='-',  label=f'mean={mean_val:.3f}')
        ax.axvline(median_val, color='black', linewidth=1.0,
                   linestyle='--', label=f'med={median_val:.3f}')

        ax.set_title(f'{run_id}  ({emb})', fontsize=9, color=colour)
        ax.set_xlabel('Cosine similarity', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 1.05)

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
    order   = sorted(preds, key=lambda r: (preds[r]['true_rank'] == 1).mean(), reverse=True)
    vals    = [(preds[r]['true_rank'] == 1).mean() for r in order]
    labels  = [f"{r}  ({_EXPERIMENTS.get(r, r)})" for r in order]
    colours = [_PALETTE.get(r, '#333333') for r in order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(order))))
    bars = ax.barh(labels, vals, color=colours, alpha=0.85)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    ax.set_xlabel('Top-1 Accuracy')
    ax.set_title('Top-1 Retrieval Accuracy — Normalised Embeddings (test set)')
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
    ax.set_title('MRR — Normalised Embeddings (test set)')
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
    """
    digit_counts = list(range(1, 7))
    ncols = 3
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), sharey=True)
    fig.suptitle('Rank CDF by Number of Digits — Normalised Embeddings (test set)',
                 fontsize=12)

    k_vals = np.arange(1, _CORPUS_SIZE + 1)

    for ax_idx, nd in enumerate(digit_counts):
        row, col = divmod(ax_idx, ncols)
        ax       = axes[row][col]

        for run_id, df in preds.items():
            sub = df[df['n_digits'] == nd]
            if sub.empty:
                continue
            ranks  = sub['true_rank'].values
            cdf    = np.array([(ranks <= k).mean() for k in k_vals])
            colour = _PALETTE.get(run_id, '#333333')
            ls     = '--' if run_id == 'B0n' else '-'
            ax.plot(k_vals, cdf, ls, color=colour, linewidth=1.4,
                    label=run_id if ax_idx == 0 else '_')

        first_df  = next(iter(preds.values()))
        n_samples = int((first_df['n_digits'] == nd).sum())
        ax.set_title(f'{nd}-digit numbers  (n={n_samples})', fontsize=9)
        ax.set_xlabel('Rank k', fontsize=8)
        ax.set_ylabel('P(rank ≤ k)', fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    handles = [
        plt.Line2D([0], [0], color=_PALETTE.get(r, '#333333'),
                   linewidth=1.6, linestyle='--' if r == 'B0n' else '-',
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
        description='Aggregate evaluation plots for Type-B normalised-embedding experiments'
    )
    parser.add_argument(
        '--runs', nargs='+', default=list(_EXPERIMENTS),
        metavar='RUN_ID',
        help='Run IDs to include (default: all). E.g. --runs B0n E2an E2bn',
    )
    parser.add_argument(
        '--tsne', action='store_true',
        help='Also generate t-SNE corpus visualisation (~5 min for 10,008 points)',
    )
    parser.add_argument(
        '--tsne-embeddings', nargs='+',
        default=['sbert', 'sbert_finetuned', 'tinybert_mean', 'tfidf_lsa'],
        metavar='EMBEDDING',
        help='Base embedding names for t-SNE (default: sbert sbert_finetuned tinybert_mean tfidf_lsa)',
    )
    parser.add_argument(
        '--tsne-colour-by', default='all', choices=['n_digits', 'colour', 'size', 'all'],
        help='Colour t-SNE points by n_digits / colour / size / all (default: all)',
    )
    args = parser.parse_args()

    unknown = [r for r in args.runs if r not in _EXPERIMENTS]
    if unknown:
        parser.error(f'Unknown run IDs: {unknown}. Available: {list(_EXPERIMENTS)}')

    FIGURES_EVAL_NORM_B.mkdir(parents=True, exist_ok=True)

    print(f'\nType-B Normalised Aggregate Evaluation Plots')
    print(f'Runs   : {args.runs}')
    print(f'Input  : {PREDICTIONS_B_NORMED}')
    print(f'Output : {FIGURES_EVAL_NORM_B}\n')

    preds = _load_predictions(args.runs)
    if not preds:
        print('No prediction CSVs found. Run evaluations first:')
        print('  python src/pipelines/evaluation/type-b/run_evals_stage1_normed_b.py')
        return

    print('\n── Aggregate plots ──')
    plot_rank_cdf(preds, FIGURES_EVAL_NORM_B)
    plot_rank_boxplot(preds, FIGURES_EVAL_NORM_B)
    plot_cosine_sim_kde(preds, FIGURES_EVAL_NORM_B)
    plot_top1_bar(preds, FIGURES_EVAL_NORM_B)
    plot_mrr_bar(preds, FIGURES_EVAL_NORM_B)
    plot_rank_cdf_by_ndigits(preds, FIGURES_EVAL_NORM_B)

    if args.tsne:
        _eval_dir = Path(__file__).resolve().parent
        if str(_eval_dir) not in sys.path:
            sys.path.insert(0, str(_eval_dir))
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
                figures_dir=FIGURES_EVAL_NORM_B,
            )

    print(f'\nDone → {FIGURES_EVAL_NORM_B}')


if __name__ == '__main__':
    main()
