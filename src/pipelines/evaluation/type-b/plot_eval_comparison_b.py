"""
Comparison plots for Type-B evaluation stages.

Comparison modes (--comparison):
  s1-base-vs-normed  (default)
    Reads: prediction/test_results.csv  vs  prediction-normalised/test_results_normed.csv
    Saves: figures/type-b/evaluation/comparison/
      cmp_top1.png, cmp_mrr.png, cmp_mean_rank.png, cmp_top1_by_ndigits.png

  s2-non-vs-normed
    Reads: prediction-s2/test_results_s2.csv  vs  prediction-s2-normalised/test_results_s2_normed.csv
    Saves: figures/type-b/evaluation/comparison/
      s2_cmp_top1.png, s2_cmp_mrr.png, s2_cmp_mean_rank.png, s2_cmp_top1_by_ndigits.png

  s1-vs-s2
    Reads: prediction/test_results.csv  vs  prediction-s2/test_results_s2.csv
    Saves: figures/type-b/evaluation/comparison/
      s1s2_cmp_top1.png, s1s2_cmp_mrr.png, s1s2_cmp_mean_rank.png

Stage-1 pairs (base → normed):
  B0 → B0n | E2a → E2an | E2b → E2bn | E2e → E2en | E2f → E2fn
  E2g → E2gn | E2h → E2hn | E2i → E2in | E2k → E2kn

Stage-2 pairs (non-normed → normed):
  S2a → S2an | S2b → S2bn | S2c → S2cn
  (additionally normed-only: S2ad, S2bd, S2cd for Combined loss)

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
    PREDICTIONS_B_S2,
    PREDICTIONS_B_S2_NORMED,
    FIGURES_EVAL_CMP_B,
)

# ── Stage-1 pairs: base run_id → normed run_id ────────────────────────────────
_PAIRS_S1: list[tuple[str, str]] = [
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

# ── Stage-2 pairs: non-normed run_id → normed MSE run_id ─────────────────────
_PAIRS_S2: list[tuple[str, str]] = [
    ('S2a', 'S2an'),
    ('S2b', 'S2bn'),
    ('S2c', 'S2cn'),
]

# ── Stage-1 vs Stage-2 pairs (same architecture, diff stage): ─────────────────
# E2e = S1 cnn_1layer/tinybert_mean; S2a = S2 cnn_1layer/tinybert_mean
_PAIRS_S1_VS_S2: list[tuple[str, str]] = [
    ('E2e', 'S2a'),
]

_COLOUR_BASE   = '#4472C4'   # solid blue — base / stage-1
_COLOUR_NORMED = '#ED7D31'   # orange — normed / stage-2


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_s1() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_csv   = PREDICTIONS_B        / 'test_results.csv'
    normed_csv = PREDICTIONS_B_NORMED / 'test_results_normed.csv'
    missing = [str(p) for p in (base_csv, normed_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing results CSV(s):\n' + '\n'.join(f'  {p}' for p in missing) +
            '\nRun evaluations first:\n'
            '  python src/pipelines/evaluation/type-b/run_evals_stage1_b.py\n'
            '  python src/pipelines/evaluation/type-b/run_evals_stage1_normed_b.py'
        )
    return (
        pd.read_csv(base_csv).set_index('run_id'),
        pd.read_csv(normed_csv).set_index('run_id'),
    )


def _load_s2() -> tuple[pd.DataFrame, pd.DataFrame]:
    non_csv = PREDICTIONS_B_S2        / 'test_results_s2.csv'
    nor_csv = PREDICTIONS_B_S2_NORMED / 'test_results_s2_normed.csv'
    missing = [str(p) for p in (non_csv, nor_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing Stage-2 results CSV(s):\n' + '\n'.join(f'  {p}' for p in missing) +
            '\nRun evaluations first:\n'
            '  python src/pipelines/evaluation/type-b/run_evals_stage2_b.py'
        )
    return (
        pd.read_csv(non_csv).set_index('run_id'),
        pd.read_csv(nor_csv).set_index('run_id'),
    )


def _load_s1_vs_s2() -> tuple[pd.DataFrame, pd.DataFrame]:
    s1_csv = PREDICTIONS_B    / 'test_results.csv'
    s2_csv = PREDICTIONS_B_S2 / 'test_results_s2.csv'
    missing = [str(p) for p in (s1_csv, s2_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            'Missing results CSV(s):\n' + '\n'.join(f'  {p}' for p in missing)
        )
    return (
        pd.read_csv(s1_csv).set_index('run_id'),
        pd.read_csv(s2_csv).set_index('run_id'),
    )


# Legacy alias
def _build_paired(
    df_left:  pd.DataFrame,
    df_right: pd.DataFrame,
    metric:   str,
    pairs:    list[tuple[str, str]] | None = None,
) -> tuple[list[str], list[float], list[float]]:
    """
    Returns (labels, left_vals, right_vals) for given pairs.
    Label is taken from df_left['embedding'] (or the run_id as fallback).
    Missing values are NaN.
    """
    if pairs is None:
        pairs = _PAIRS_S1
    labels, left_vals, right_vals = [], [], []
    for left_id, right_id in pairs:
        labels.append(
            str(df_left.loc[left_id, 'embedding'])
            if left_id in df_left.index and 'embedding' in df_left.columns
            else left_id
        )
        left_vals.append(
            float(df_left.loc[left_id, metric])
            if left_id in df_left.index else float('nan')
        )
        right_vals.append(
            float(df_right.loc[right_id, metric])
            if right_id in df_right.index else float('nan')
        )
    return labels, left_vals, right_vals


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
    labels, bv, nv = _build_paired(df_base, df_normed, 'test_top1', _PAIRS_S1)
    _paired_bar(labels, bv, nv,
                ylabel='Top-1 Accuracy',
                title='Top-1 Accuracy — Base vs Normalised (Stage 1)',
                out_path=FIGURES_EVAL_CMP_B / 'cmp_top1.png')


def plot_cmp_mrr(df_base: pd.DataFrame, df_normed: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_base, df_normed, 'test_mrr', _PAIRS_S1)
    _paired_bar(labels, bv, nv,
                ylabel='MRR',
                title='MRR — Base vs Normalised (Stage 1)',
                out_path=FIGURES_EVAL_CMP_B / 'cmp_mrr.png')


def plot_cmp_mean_rank(df_base: pd.DataFrame, df_normed: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_base, df_normed, 'test_mean_rank', _PAIRS_S1)
    _paired_bar(labels, bv, nv,
                ylabel='Mean Rank (lower = better)',
                title='Mean Rank — Base vs Normalised (Stage 1)',
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

    n_pairs = len(_PAIRS_S1)
    ncols   = 3
    nrows   = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             sharey=True)
    axes_flat = axes.flatten()

    for idx, (base_id, normed_id) in enumerate(_PAIRS_S1):
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

    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Top-1 Accuracy by n_digits — Base vs Normalised (Stage 1)', fontsize=12)
    fig.tight_layout()
    out = FIGURES_EVAL_CMP_B / 'cmp_top1_by_ndigits.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


# ── Stage-2: non-normalised vs normalised ─────────────────────────────────────

def plot_s2_cmp_top1(df_non: pd.DataFrame, df_nor: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_non, df_nor, 'test_top1', _PAIRS_S2)
    _paired_bar(labels, bv, nv,
                ylabel='Top-1 Accuracy',
                title='Top-1 Accuracy — Non-Normed vs Normed (Stage 2, MSE)',
                out_path=FIGURES_EVAL_CMP_B / 's2_cmp_top1.png')


def plot_s2_cmp_mrr(df_non: pd.DataFrame, df_nor: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_non, df_nor, 'test_mrr', _PAIRS_S2)
    _paired_bar(labels, bv, nv,
                ylabel='MRR',
                title='MRR — Non-Normed vs Normed (Stage 2, MSE)',
                out_path=FIGURES_EVAL_CMP_B / 's2_cmp_mrr.png')


def plot_s2_cmp_mean_rank(df_non: pd.DataFrame, df_nor: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_non, df_nor, 'test_mean_rank', _PAIRS_S2)
    _paired_bar(labels, bv, nv,
                ylabel='Mean Rank (lower = better)',
                title='Mean Rank — Non-Normed vs Normed (Stage 2, MSE)',
                out_path=FIGURES_EVAL_CMP_B / 's2_cmp_mean_rank.png',
                lower_better=True)


def plot_s2_cmp_top1_by_ndigits(df_non: pd.DataFrame, df_nor: pd.DataFrame) -> None:
    digit_groups = list(range(1, 7))
    cols         = [f'top1_{nd}d' for nd in digit_groups]
    _ARCH_LABELS = {'S2a': 'cnn_1layer', 'S2b': 'cnn_3layer', 'S2c': 'resnet18_pt'}

    n_pairs = len(_PAIRS_S2)
    ncols   = 3
    nrows   = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             sharey=True)
    axes_flat = axes.flatten()

    for idx, (non_id, nor_id) in enumerate(_PAIRS_S2):
        ax        = axes_flat[idx]
        arch_label = _ARCH_LABELS.get(non_id, non_id)

        if non_id in df_non.index:
            bv = [float(df_non.loc[non_id, c]) if c in df_non.columns
                  else float('nan') for c in cols]
            ax.plot(digit_groups, bv, '-o', color=_COLOUR_BASE,
                    linewidth=1.8, markersize=5, label='non-normed')

        if nor_id in df_nor.index:
            nv = [float(df_nor.loc[nor_id, c]) if c in df_nor.columns
                  else float('nan') for c in cols]
            ax.plot(digit_groups, nv, '--s', color=_COLOUR_NORMED,
                    linewidth=1.8, markersize=5, label='normed (MSE)')

        ax.set_title(arch_label, fontsize=9)
        ax.set_xlabel('n_digits', fontsize=8)
        ax.set_ylabel('Top-1', fontsize=8)
        ax.set_xticks(digit_groups)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Top-1 by n_digits — Stage-2 Non-Normed vs Normed', fontsize=12)
    fig.tight_layout()
    out = FIGURES_EVAL_CMP_B / 's2_cmp_top1_by_ndigits.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


# ── Stage-1 vs Stage-2 (architecture effect of cnn_1layer) ───────────────────

def plot_s1_vs_s2_cmp_top1(df_s1: pd.DataFrame, df_s2: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_s1, df_s2, 'test_top1', _PAIRS_S1_VS_S2)
    _paired_bar(labels, bv, nv,
                ylabel='Top-1 Accuracy',
                title='Top-1 Accuracy — Stage 1 vs Stage 2 (cnn_1layer / tinybert_mean)',
                out_path=FIGURES_EVAL_CMP_B / 's1s2_cmp_top1.png')


def plot_s1_vs_s2_cmp_mrr(df_s1: pd.DataFrame, df_s2: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_s1, df_s2, 'test_mrr', _PAIRS_S1_VS_S2)
    _paired_bar(labels, bv, nv,
                ylabel='MRR',
                title='MRR — Stage 1 vs Stage 2 (cnn_1layer / tinybert_mean)',
                out_path=FIGURES_EVAL_CMP_B / 's1s2_cmp_mrr.png')


def plot_s1_vs_s2_cmp_mean_rank(df_s1: pd.DataFrame, df_s2: pd.DataFrame) -> None:
    labels, bv, nv = _build_paired(df_s1, df_s2, 'test_mean_rank', _PAIRS_S1_VS_S2)
    _paired_bar(labels, bv, nv,
                ylabel='Mean Rank (lower = better)',
                title='Mean Rank — Stage 1 vs Stage 2 (cnn_1layer / tinybert_mean)',
                out_path=FIGURES_EVAL_CMP_B / 's1s2_cmp_mean_rank.png',
                lower_better=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse as _ap
    parser = _ap.ArgumentParser(
        description='Type-B comparison plots across stages and normalisation variants'
    )
    parser.add_argument(
        '--comparison',
        choices=['s1-base-vs-normed', 's2-non-vs-normed', 's1-vs-s2', 'all'],
        default='s1-base-vs-normed',
        help=(
            's1-base-vs-normed : Stage-1 base vs normalised (default) | '
            's2-non-vs-normed  : Stage-2 non-normalised vs normalised | '
            's1-vs-s2          : Stage-1 vs Stage-2 (cnn_1layer shared pair) | '
            'all               : run all comparisons'
        ),
    )
    args = parser.parse_args()

    FIGURES_EVAL_CMP_B.mkdir(parents=True, exist_ok=True)
    run_s1   = args.comparison in ('s1-base-vs-normed', 'all')
    run_s2   = args.comparison in ('s2-non-vs-normed',  'all')
    run_1v2  = args.comparison in ('s1-vs-s2',          'all')

    print(f'\nType-B Comparison Plots  (--comparison {args.comparison})')
    print(f'Output : {FIGURES_EVAL_CMP_B}\n')

    # ── Stage-1: base vs normed ────────────────────────────────────────────────
    if run_s1:
        print('── Stage-1: Base vs Normalised ──')
        try:
            df_base, df_normed = _load_s1()
            print(f'  Loaded {len(df_base)} base, {len(df_normed)} normed runs')
            plot_cmp_top1(df_base, df_normed)
            plot_cmp_mrr(df_base, df_normed)
            plot_cmp_mean_rank(df_base, df_normed)
            plot_cmp_top1_by_ndigits(df_base, df_normed)
        except FileNotFoundError as exc:
            print(f'  [skip] {exc}')

    # ── Stage-2: non-normalised vs normalised ──────────────────────────────────
    if run_s2:
        print('── Stage-2: Non-Normalised vs Normalised ──')
        try:
            df_non, df_nor = _load_s2()
            print(f'  Loaded {len(df_non)} non-normed, {len(df_nor)} normed S2 runs')
            plot_s2_cmp_top1(df_non, df_nor)
            plot_s2_cmp_mrr(df_non, df_nor)
            plot_s2_cmp_mean_rank(df_non, df_nor)
            plot_s2_cmp_top1_by_ndigits(df_non, df_nor)
        except FileNotFoundError as exc:
            print(f'  [skip] {exc}')

    # ── Stage-1 vs Stage-2 ────────────────────────────────────────────────────
    if run_1v2:
        print('── Stage-1 vs Stage-2 ──')
        try:
            df_s1, df_s2 = _load_s1_vs_s2()
            print(f'  Loaded {len(df_s1)} S1, {len(df_s2)} S2 non-normed runs')
            plot_s1_vs_s2_cmp_top1(df_s1, df_s2)
            plot_s1_vs_s2_cmp_mrr(df_s1, df_s2)
            plot_s1_vs_s2_cmp_mean_rank(df_s1, df_s2)
        except FileNotFoundError as exc:
            print(f'  [skip] {exc}')

    print(f'\nDone → {FIGURES_EVAL_CMP_B}')


if __name__ == '__main__':
    main()
