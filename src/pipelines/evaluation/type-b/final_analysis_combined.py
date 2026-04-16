"""
src/pipelines/evaluation/type-b/final_analysis_combined.py

Unified cross-variant analysis: combines ALL Type-B results
(Stage-1 non-normed, Stage-1 normed, Stage-2 non-normed, Stage-2 normed)
into a single comparison table and set of figures.

Unlike final_analysis.py / final_analysis_normed.py, this file does NOT
compute a composite ranking across variants — normalised and non-normalised
embeddings live in different metric spaces and cannot be fairly scored
against each other.  Instead it shows raw metric comparisons so the reader
can judge the effect of normalisation and architecture choice side-by-side.

Sources (all optional — missing CSVs are silently skipped)
----------------------------------------------------------
  prediction/test_results.csv                  → S1 non-normed
  prediction-normalised/test_results_normed.csv → S1 normed
  prediction-s2/test_results_s2.csv            → S2 non-normed
  prediction-s2-normalised/test_results_s2_normed.csv → S2 normed

Outputs
-------
  prediction-combined/combined_results.csv     — unified table (all variants)
  figures/type-b/evaluation/combined/
    combined_top1_bar.png        — top-1 accuracy, all runs, grouped by variant
    combined_mrr_bar.png         — MRR, same grouping
    combined_mean_rank_bar.png   — mean rank (lower = better)
    combined_cosine_bar.png      — mean cosine similarity
    variant_box_top1.png         — box chart: metric distribution per variant
    normed_vs_base_scatter.png   — scatter: non-normed top-1 vs normed top-1
                                   (paired by embedding, S1 only)

Usage
-----
  python src/pipelines/evaluation/type-b/final_analysis_combined.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import (  # noqa: E402
    PREDICTIONS_B,
    PREDICTIONS_B_NORMED,
    PREDICTIONS_B_S2,
    PREDICTIONS_B_S2_NORMED,
    PREDICTIONS_B_COMBINED,
    FIGURES_EVAL_B_COMBINED,
)

# ── Variant metadata ───────────────────────────────────────────────────────────
_SOURCES: list[dict] = [
    {
        'csv':     PREDICTIONS_B        / 'test_results.csv',
        'variant': 'S1-non-normed',
        'stage':   'S1',
        'normed':  False,
        'colour':  '#2196F3',
    },
    {
        'csv':     PREDICTIONS_B_NORMED / 'test_results_normed.csv',
        'variant': 'S1-normed',
        'stage':   'S1',
        'normed':  True,
        'colour':  '#64B5F6',
    },
    {
        'csv':     PREDICTIONS_B_S2     / 'test_results_s2.csv',
        'variant': 'S2-non-normed',
        'stage':   'S2',
        'normed':  False,
        'colour':  '#FF5722',
    },
    {
        'csv':     PREDICTIONS_B_S2_NORMED / 'test_results_s2_normed.csv',
        'variant': 'S2-normed',
        'stage':   'S2',
        'normed':  True,
        'colour':  '#FF8A65',
    },
]

# S1 run_id pairs for normed-vs-base scatter (base_id → normed_id)
_S1_PAIRS = [
    ('B0', 'B0n'), ('E2e', 'E2en'), ('E2a', 'E2an'), ('E2b', 'E2bn'),
    ('E2f', 'E2fn'), ('E2g', 'E2gn'), ('E2h', 'E2hn'),
    ('E2i', 'E2in'), ('E2k', 'E2kn'),
]

_METRICS = ['test_top1', 'test_mrr', 'test_mean_rank', 'test_mean_cosine']
_METRIC_LABELS = {
    'test_top1':        'Top-1 Accuracy',
    'test_mrr':         'MRR',
    'test_mean_rank':   'Mean Rank (lower = better)',
    'test_mean_cosine': 'Mean Cosine Similarity',
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_all() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for src in _SOURCES:
        if not src['csv'].exists():
            print(f'  [skip] {src["variant"]} — {src["csv"].name} not found')
            continue
        df = pd.read_csv(src['csv'])
        if df.empty:
            print(f'  [skip] {src["variant"]} — CSV is empty')
            continue
        df['variant'] = src['variant']
        df['stage']   = src['stage']
        df['normed']  = src['normed']
        frames.append(df)
        print(f'  [load] {src["variant"]:20s}  {len(df):3d} runs  ← {src["csv"].name}')

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Console output ─────────────────────────────────────────────────────────────

def _print_combined_table(df: pd.DataFrame) -> None:
    sep = '=' * 92
    print(f'\n{sep}')
    print('  Type-B Combined Results — All Variants')
    print(sep)
    print(f"  {'Variant':<16} {'Run':<7} {'Embedding':<26} "
          f"{'top-1':>6} {'MRR':>7} {'mean_rank':>10} {'cosine':>8} {'CS':>7}")
    print(f"  {'-'*88}")

    for variant, grp in df.groupby('variant', sort=False):
        grp_sorted = grp.sort_values('test_top1', ascending=False)
        for _, row in grp_sorted.iterrows():
            print(
                f"  {str(row['variant']):<16} {str(row['run_id']):<7} "
                f"{str(row['embedding']):<26} "
                f"{row['test_top1']:>6.4f} {row['test_mrr']:>7.4f} "
                f"{row['test_mean_rank']:>10.1f} {row['test_mean_cosine']:>8.4f} "
                f"{row['colour_size_correct']:>7.3f}"
            )
        # Variant summary line
        print(
            f"  {'':16} {'— avg':7} {'':26} "
            f"{grp['test_top1'].mean():>6.4f} {grp['test_mrr'].mean():>7.4f} "
            f"{grp['test_mean_rank'].mean():>10.1f} {grp['test_mean_cosine'].mean():>8.4f} "
            f"{grp['colour_size_correct'].mean():>7.3f}"
        )
        print(f"  {'-'*88}")

    print(f'{sep}\n')


# ── Plots ──────────────────────────────────────────────────────────────────────

def _variant_colour(variant: str) -> str:
    return next((s['colour'] for s in _SOURCES if s['variant'] == variant), '#888888')


def _plot_metric_bar(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """Horizontal grouped bar: all runs, colour-coded by variant."""
    lower_better = metric == 'test_mean_rank'
    label        = _METRIC_LABELS[metric]

    fig, axes = plt.subplots(
        1, len(_SOURCES),
        figsize=(5 * len(_SOURCES), max(5, 0.35 * df['run_id'].nunique())),
        sharey=False,
    )
    if len(_SOURCES) == 1:
        axes = [axes]

    for ax, src in zip(axes, _SOURCES):
        grp = df[df['variant'] == src['variant']].copy()
        if grp.empty:
            ax.set_visible(False)
            continue

        grp = grp.sort_values(metric, ascending=lower_better)
        vals   = grp[metric].tolist()
        labels = grp['run_id'].tolist()
        colour = src['colour']

        bars = ax.barh(labels, vals, color=colour, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(val + (abs(val) * 0.01 + 1e-4) * (-1 if lower_better else 1),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=7,
                    ha='right' if lower_better else 'left')

        ax.set_title(src['variant'], fontsize=10, color=colour)
        ax.set_xlabel(label, fontsize=9)
        if lower_better:
            ax.annotate('lower = better', xy=(0.98, 0.02), xycoords='axes fraction',
                        fontsize=7, color='gray', ha='right')
        ax.grid(True, axis='x', alpha=0.3)
        ax.invert_yaxis()

    stem = metric.replace('test_', '')
    fig.suptitle(f'{label} — All Variants', fontsize=12)
    fig.tight_layout()
    out = out_dir / f'combined_{stem}_bar.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def _plot_variant_summary_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar: variant-level mean for top-1, MRR, cosine."""
    metrics  = ['test_top1', 'test_mrr', 'test_mean_cosine']
    m_labels = ['Top-1', 'MRR', 'Cosine']
    variants = [s['variant'] for s in _SOURCES if s['variant'] in df['variant'].values]
    colours  = [_variant_colour(v) for v in variants]

    x    = np.arange(len(metrics))
    w    = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (var, col) in enumerate(zip(variants, colours)):
        grp  = df[df['variant'] == var]
        vals = [grp[m].mean() for m in metrics]
        offset = (i - len(variants) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=var, color=col, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(m_labels, fontsize=11)
    ax.set_ylabel('Mean value across runs', fontsize=10)
    ax.set_title('Per-Variant Mean Performance (top-1 / MRR / cosine)', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    out = out_dir / 'variant_summary_bar.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def _plot_normed_vs_base_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Scatter: non-normed top-1 (x) vs normed top-1 (y) for S1 paired runs.
    Points above the diagonal = normalisation helped.
    """
    base_df   = df[df['variant'] == 'S1-non-normed'].set_index('run_id')
    normed_df = df[df['variant'] == 'S1-normed'].set_index('run_id')

    xs, ys, labels = [], [], []
    for base_id, normed_id in _S1_PAIRS:
        if base_id in base_df.index and normed_id in normed_df.index:
            xs.append(float(base_df.loc[base_id, 'test_top1']))
            ys.append(float(normed_df.loc[normed_id, 'test_top1']))
            labels.append(base_id)

    if not xs:
        print('  [skip] normed_vs_base scatter — no paired S1 data')
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, color='#2196F3', s=60, zorder=3)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords='offset points', xytext=(5, 3), fontsize=8)

    lim = max(max(xs), max(ys)) * 1.1
    ax.plot([0, lim], [0, lim], '--', color='grey', linewidth=1, alpha=0.6, label='y = x')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Top-1 (non-normed)',  fontsize=11)
    ax.set_ylabel('Top-1 (normed)',      fontsize=11)
    ax.set_title('Normalisation Effect on Top-1 (S1 pairs)\n'
                 'Above diagonal = normalisation helped', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / 'normed_vs_base_scatter.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def _plot_stage_architecture_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """
    For S2 runs: side-by-side top-1 bars for non-normed vs normed MSE,
    grouped by architecture.
    """
    arch_map = {'S2a': 'cnn_1layer', 'S2b': 'cnn_3layer', 'S2c': 'resnet18_pt'}
    mse_pairs = [('S2a', 'S2an'), ('S2b', 'S2bn'), ('S2c', 'S2cn')]

    s2_non = df[df['variant'] == 'S2-non-normed'].set_index('run_id')
    s2_nor = df[df['variant'] == 'S2-normed'].set_index('run_id')

    if s2_non.empty and s2_nor.empty:
        return

    archs = [arch_map[b] for b, _ in mse_pairs]
    non_vals = [float(s2_non.loc[b, 'test_top1']) if b in s2_non.index else float('nan')
                for b, _ in mse_pairs]
    nor_vals = [float(s2_nor.loc[n, 'test_top1']) if n in s2_nor.index else float('nan')
                for _, n in mse_pairs]

    # Also add Combined-loss normed
    comb_pairs = [('S2a', 'S2ad'), ('S2b', 'S2bd'), ('S2c', 'S2cd')]
    comb_vals = [float(s2_nor.loc[n, 'test_top1']) if n in s2_nor.index else float('nan')
                 for _, n in comb_pairs]

    x  = np.arange(len(archs))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x - w, non_vals,  w, label='non-normed (MSE)', color='#FF5722', alpha=0.85)
    b2 = ax.bar(x,     nor_vals,  w, label='normed (MSE)',      color='#FF8A65', alpha=0.85)
    b3 = ax.bar(x + w, comb_vals, w, label='normed (Combined)', color='#7E57C2', alpha=0.85)

    all_vals = [v for v in non_vals + nor_vals + comb_vals if not np.isnan(v)]
    y_max = max(all_vals) if all_vals else 0.01
    y_top = y_max * 1.35   # headroom for value labels

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + y_top * 0.02,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=10)
    ax.set_ylabel('Top-1 Accuracy', fontsize=10)
    ax.set_title('Stage-2 Architecture × Loss × Normalisation (Top-1)', fontsize=11)
    ax.set_ylim(0, y_top)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    out = out_dir / 's2_arch_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print('\n' + '=' * 60)
    print('  Type-B Combined Analysis — All Variants')
    print('=' * 60)

    df = _load_all()
    if df.empty:
        print('\n  No data loaded. Run evaluations first:')
        print('    python total_eval_pipeline_all_b.py')
        return

    print(f'\n  Total rows loaded: {len(df)}  '
          f'(variants: {df["variant"].value_counts().to_dict()})')

    _print_combined_table(df)

    # ── Save combined CSV ──────────────────────────────────────────────────────
    PREDICTIONS_B_COMBINED.mkdir(parents=True, exist_ok=True)
    out_csv = PREDICTIONS_B_COMBINED / 'combined_results.csv'
    df.to_csv(out_csv, index=False)
    print(f'  [saved] {out_csv}')

    # ── Figures ────────────────────────────────────────────────────────────────
    FIGURES_EVAL_B_COMBINED.mkdir(parents=True, exist_ok=True)

    print('\n── Per-metric bar charts ──')
    for metric in _METRICS:
        _plot_metric_bar(df, metric, FIGURES_EVAL_B_COMBINED)

    print('\n── Variant summary ──')
    _plot_variant_summary_bar(df, FIGURES_EVAL_B_COMBINED)

    print('\n── Normalisation effect (S1 paired scatter) ──')
    _plot_normed_vs_base_scatter(df, FIGURES_EVAL_B_COMBINED)

    print('\n── Stage-2 architecture comparison ──')
    _plot_stage_architecture_comparison(df, FIGURES_EVAL_B_COMBINED)

    print(f'\nDone → {FIGURES_EVAL_B_COMBINED}\n')


if __name__ == '__main__':
    main()
