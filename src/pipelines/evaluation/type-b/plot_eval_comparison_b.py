"""
Base vs Normalised embedding comparison plots for Type-B.

Reads:
  metrics/type-b/prediction/test_results.csv           (base runs)
  metrics/type-b/prediction-normalised/test_results_normed.csv  (normed runs)

Saves to:
  figures/type-b/evaluation/comparison/
    cmp_top1.png          — paired bar: top-1 accuracy per embedding (base vs normed)
    cmp_mrr.png           — paired bar: MRR per embedding
    cmp_mean_rank.png     — paired bar: mean rank per embedding (lower = better)
    cmp_top1_by_ndigits.png — faceted line: top-1 by n_digits, base (solid) vs normed (dashed)

Pairs (base → normed):
  B0 → B0n | E2a → E2an | E2b → E2bn | E2e → E2en | E2f → E2fn
  E2g → E2gn | E2h → E2hn | E2i → E2in | E2k → E2kn

Note: E2ln (bert_mean_normed) and E2mn (bert_pooler_normed) have no base counterpart
and are shown normed-only in the comparison plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import (  # noqa: E402
    PREDICTIONS_B,
    PREDICTIONS_B_NORMED,
    FIGURES_EVAL_CMP_B,
)

# ── Pairing: base run_id → normed run_id ──────────────────────────────────────
_PAIRS: list[tuple[str, str]] = [
    ('B0',  'B0n'),
    ('E2a', 'E2an'),
    ('E2b', 'E2bn'),
    ('E2e', 'E2en'),
    ('E2f', 'E2fn'),
    ('E2g', 'E2gn'),
    ('E2h', 'E2hn'),
    ('E2i', 'E2in'),
    ('E2k', 'E2kn'),
]
# Normed-only (no base counterpart)
_NORMED_ONLY: list[str] = ['E2ln', 'E2mn']

_COLOUR_BASE   = '#4472C4'   # solid blue — base
_COLOUR_NORMED = '#ED7D31'   # orange — normed


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_csv   = PREDICTIONS_B       / 'test_results.csv'
    normed_csv = PREDICTIONS_B_NORMED / 'test_results_normed.csv'

    missing = []
    if not base_csv.exists():
        missing.append(str(base_csv))
    if not normed_csv.exists():
        missing.append(str(normed_csv))
    if missing:
        raise FileNotFoundError(
            'Missing results CSV(s):\n' + '\n'.join(f'  {p}' for p in missing) +
            '\nRun evaluations first:\n'
            '  python src/pipelines/evaluation/type-b/run_evals_stage1_b.py\n'
            '  python src/pipelines/evaluation/type-b/run_evals_stage1_normed_b.py'
        )

    df_base   = pd.read_csv(base_csv).set_index('run_id')
    df_normed = pd.read_csv(normed_csv).set_index('run_id')
    return df_base, df_normed


def _build_paired(
    df_base:   pd.DataFrame,
    df_normed: pd.DataFrame,
    metric:    str,
) -> tuple[list[str], list[float], list[float]]:
    """
    Returns (labels, base_vals, normed_vals) for paired embeddings.
    Missing values are NaN.
    """
    labels, base_vals, normed_vals = [], [], []
    for base_id, normed_id in _PAIRS:
        labels.append(df_base.loc[base_id, 'embedding']
                      if base_id in df_base.index else base_id)
        base_vals.append(
            float(df_base.loc[base_id, metric])
            if base_id in df_base.index else float('nan')
        )
        normed_vals.append(
            float(df_normed.loc[normed_id, metric])
            if normed_id in df_normed.index else float('nan')
        )
    return labels, base_vals, normed_vals


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def _paired_bar(
    labels:      list[str],
    base_vals:   list[float],
    normed_vals: list[float],
    ylabel:      str,
    title:       str,
    out_path:    Path,
    lower_better: bool = False,
) -> None:
    """Grouped bar chart: base (blue) vs normed (orange) per embedding."""
    n = len(labels)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(9, n * 1.1), 5))
    bars_b = ax.bar(x - w / 2, base_vals,   w, label='base',   color=_COLOUR_BASE,   alpha=0.85)
    bars_n = ax.bar(x + w / 2, normed_vals, w, label='normed', color=_COLOUR_NORMED, alpha=0.85)

    # Value labels
    for bar in bars_b:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars_n:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Embedding')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    if lower_better:
        ax.annotate('lower = better', xy=(0.01, 0.97), xycoords='axes fraction',
                    fontsize=8, color='gray', va='top')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out_path.name}')


def plot_cmp_top1(df_base: pd.DataFrame, df_normed: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_base, df_normed, 'test_top1')
    _paired_bar(labels, bv, nv,
                ylabel='Top-1 Accuracy',
                title='Top-1 Accuracy — Base vs Normalised',
                out_path=FIGURES_EVAL_CMP_B / 'cmp_top1.png')


def plot_cmp_mrr(df_base: pd.DataFrame, df_normed: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_base, df_normed, 'test_mrr')
    _paired_bar(labels, bv, nv,
                ylabel='MRR',
                title='MRR — Base vs Normalised',
                out_path=FIGURES_EVAL_CMP_B / 'cmp_mrr.png')


def plot_cmp_mean_rank(df_base: pd.DataFrame, df_normed: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_base, df_normed, 'test_mean_rank')
    _paired_bar(labels, bv, nv,
                ylabel='Mean Rank (lower = better)',
                title='Mean Rank — Base vs Normalised',
                out_path=FIGURES_EVAL_CMP_B / 'cmp_mean_rank.png',
                lower_better=True)


def plot_cmp_top1_by_ndigits(df_base: pd.DataFrame, df_normed: pd.DataFrame) -> None:
    """
    Line plot: top-1 by n_digits per embedding pair.
    Base = solid line, Normed = dashed line.
    One subplot per paired embedding.
    """
    digit_groups = list(range(1, 7))
    cols         = [f'top1_{nd}d' for nd in digit_groups]

    n_pairs = len(_PAIRS)
    ncols   = 3
    nrows   = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             sharey=True)
    axes_flat = axes.flatten()

    for idx, (base_id, normed_id) in enumerate(_PAIRS):
        ax = axes_flat[idx]
        emb_label = (df_base.loc[base_id, 'embedding']
                     if base_id in df_base.index else base_id)

        if base_id in df_base.index:
            bv = [float(df_base.loc[base_id, c]) if c in df_base.columns
                  else float('nan') for c in cols]
            ax.plot(digit_groups, bv, '-o', color=_COLOUR_BASE,
                    linewidth=1.8, markersize=5, label='base')

        if normed_id in df_normed.index:
            nv = [float(df_normed.loc[normed_id, c]) if c in df_normed.columns
                  else float('nan') for c in cols]
            ax.plot(digit_groups, nv, '--s', color=_COLOUR_NORMED,
                    linewidth=1.8, markersize=5, label='normed')

        ax.set_title(emb_label, fontsize=9)
        ax.set_xlabel('n_digits', fontsize=8)
        ax.set_ylabel('Top-1', fontsize=8)
        ax.set_xticks(digit_groups)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Top-1 Accuracy by n_digits — Base vs Normalised', fontsize=12)
    fig.tight_layout()
    out = FIGURES_EVAL_CMP_B / 'cmp_top1_by_ndigits.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print('\nType-B — Base vs Normalised Comparison Plots')
    print(f'Base    : {PREDICTIONS_B / "test_results.csv"}')
    print(f'Normed  : {PREDICTIONS_B_NORMED / "test_results_normed.csv"}')
    print(f'Output  : {FIGURES_EVAL_CMP_B}\n')

    try:
        df_base, df_normed = _load()
    except FileNotFoundError as exc:
        print(f'ERROR: {exc}')
        return

    print(f'  Loaded {len(df_base)} base runs, {len(df_normed)} normed runs')

    FIGURES_EVAL_CMP_B.mkdir(parents=True, exist_ok=True)

    plot_cmp_top1(df_base, df_normed)
    plot_cmp_mrr(df_base, df_normed)
    plot_cmp_mean_rank(df_base, df_normed)
    plot_cmp_top1_by_ndigits(df_base, df_normed)

    print(f'\nDone → {FIGURES_EVAL_CMP_B}')


if __name__ == '__main__':
    main()
