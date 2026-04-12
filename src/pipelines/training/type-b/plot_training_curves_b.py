# always run:   python src/pipelines/training/type-b/plot_training_curves_b.py --normalised --compare                
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import METRICS_B_NON_NORMED, METRICS_B_NORMED, FIGURES_DIR  # noqa: E402

FIGURES_TRAIN_B    = FIGURES_DIR / 'train'
FIGURES_TRAIN_NORM = FIGURES_DIR / 'train' / 'normalised'
COMPARISON_DIR     = FIGURES_DIR / 'train' / 'comparison'


# ══════════════════════════════════════════════════════════════════════════════
# Experiment registry  (model_name, embedding_name) → run_id label
# ══════════════════════════════════════════════════════════════════════════════

_EXPERIMENTS: dict[str, dict] = {
    'B0':  {'model_name': 'cnn_1layer', 'embedding_name': 'tfidf_lsa'},
    'E2a': {'model_name': 'cnn_1layer', 'embedding_name': 'sbert'},
    'E2b': {'model_name': 'cnn_1layer', 'embedding_name': 'sbert_finetuned'},
    'E2e': {'model_name': 'cnn_1layer', 'embedding_name': 'tinybert_mean'},
    'E2f': {'model_name': 'cnn_1layer', 'embedding_name': 'tinybert_pooler'},
    'E2g': {'model_name': 'cnn_1layer', 'embedding_name': 'glove'},
    'E2h': {'model_name': 'cnn_1layer', 'embedding_name': 'word2vec_pretrained'},
    'E2i': {'model_name': 'cnn_1layer', 'embedding_name': 'word2vec_skipgram'},
    'E2k': {'model_name': 'cnn_1layer', 'embedding_name': 'tfidf_w2v'},
}

# Colour palette — one colour per run for consistent aggregate plots
_PALETTE: dict[str, str] = {
    'B0':  '#607D8B',   # blue-grey  (baseline)
    'E2a': '#2196F3',   # blue
    'E2b': '#00897B',   # teal (clearly distinct from E2a)
    'E2e': '#4CAF50',   # green
    'E2f': '#F48FB1',   # light pink
    'E2g': '#FF9800',   # orange
    'E2h': '#FF5722',   # deep-orange
    'E2i': '#9C27B0',   # purple
    'E2k': '#E91E63',   # pink
}

# ── Normalised experiment registry ─────────────────────────────────────────────
# Filename format in METRICS_B_NORMED:
#   b_{model}_{embedding}_normed_full_sweep_normed_{timestamp}_training_log.csv
# 'base_run' links back to _EXPERIMENTS for comparison; None = normed-only run.

_EXPERIMENTS_NORMED: dict[str, dict] = {
    'B0n':  {'model_name': 'cnn_1layer', 'embedding_name': 'tfidf_lsa_normed',          'base_run': 'B0'},
    'E2an': {'model_name': 'cnn_1layer', 'embedding_name': 'sbert_normed',              'base_run': 'E2a'},
    'E2bn': {'model_name': 'cnn_1layer', 'embedding_name': 'sbert_finetuned_normed',    'base_run': 'E2b'},
    'E2en': {'model_name': 'cnn_1layer', 'embedding_name': 'tinybert_mean_normed',      'base_run': 'E2e'},
    'E2fn': {'model_name': 'cnn_1layer', 'embedding_name': 'tinybert_pooler_normed',    'base_run': 'E2f'},
    'E2gn': {'model_name': 'cnn_1layer', 'embedding_name': 'glove_normed',              'base_run': 'E2g'},
    'E2hn': {'model_name': 'cnn_1layer', 'embedding_name': 'word2vec_pretrained_normed','base_run': 'E2h'},
    'E2in': {'model_name': 'cnn_1layer', 'embedding_name': 'word2vec_skipgram_normed',  'base_run': 'E2i'},
    'E2kn': {'model_name': 'cnn_1layer', 'embedding_name': 'tfidf_w2v_normed',          'base_run': 'E2k'},
    'E2ln': {'model_name': 'cnn_1layer', 'embedding_name': 'bert_mean_normed',          'base_run': None},
    'E2mn': {'model_name': 'cnn_1layer', 'embedding_name': 'bert_pooler_normed',        'base_run': None},
}

# Same hue as base but slightly desaturated — normed runs shown with dashed line
# in comparison plots, so colour alone doesn't need to encode normed-ness.
_PALETTE_NORMED: dict[str, str] = {
    'B0n':  '#90A4AE',   # lighter blue-grey
    'E2an': '#64B5F6',   # lighter blue
    'E2bn': '#4DB6AC',   # lighter teal
    'E2en': '#81C784',   # lighter green
    'E2fn': '#F8BBD9',   # lighter pink
    'E2gn': '#FFB74D',   # lighter orange
    'E2hn': '#FF8A65',   # lighter deep-orange
    'E2in': '#CE93D8',   # lighter purple
    'E2kn': '#F48FB1',   # lighter pink
    'E2ln': '#80CBC4',   # teal (bert_mean, no base)
    'E2mn': '#A5D6A7',   # green (bert_pooler, no base)
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

_ALL_EMBEDDING_NAMES: set[str] = {
    'tfidf_lsa', 'sbert', 'sbert_finetuned', 'tinybert_mean',
    'tinybert_pooler', 'glove', 'word2vec_pretrained',
    'word2vec_skipgram', 'tfidf_w2v',
}


def _embedding_matches(stem: str, model_name: str, embedding_name: str) -> bool:
    core        = stem.replace('_training_log', '')
    parts       = core.split('_')
    model_parts = model_name.split('_')
    emb_parts   = embedding_name.split('_')
    n_model     = len(model_parts)
    n_emb       = len(emb_parts)
    start       = 1 + n_model

    # Short check: does our embedding match at the expected position?
    if parts[start : start + n_emb] != emb_parts:
        return False

    # Disambiguation: reject if a longer known embedding also matches here
    for other in _ALL_EMBEDDING_NAMES:
        if other == embedding_name:
            continue
        other_parts = other.split('_')
        n_other     = len(other_parts)
        if (n_other > n_emb
                and other_parts[:n_emb] == emb_parts
                and parts[start : start + n_other] == other_parts):
            return False   # file belongs to the longer embedding

    return True


def _resolve_log(model_name: str, embedding_name: str) -> Path | None:
    """
    Return the most recently modified training log for (model, embedding).
    Uses exact embedding name matching to avoid sbert matching sbert_finetuned.
    """
    candidates = sorted(
        (
            p for p in METRICS_B_NON_NORMED.glob(
                f'b_{model_name}_{embedding_name}_*_training_log.csv'
            )
            if _embedding_matches(p.stem, model_name, embedding_name)
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _load_logs(run_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Return {run_id: DataFrame} for runs that have training logs."""
    loaded: dict[str, pd.DataFrame] = {}
    for run_id in run_ids:
        cfg      = _EXPERIMENTS[run_id]
        log_path = _resolve_log(cfg['model_name'], cfg['embedding_name'])
        if log_path is None:
            print(f'  [skip] {run_id} — no training log found')
            continue
        df = pd.read_csv(log_path)
        if df.empty:
            print(f'  [skip] {run_id} — training log is empty')
            continue
        loaded[run_id] = df
        print(f'  [load] {run_id} ← {log_path.name}  ({len(df)} epochs)')
    return loaded


# ── Normalised data loading ────────────────────────────────────────────────────

def _resolve_log_normed(model_name: str, embedding_name: str) -> Path | None:

    candidates = sorted(
        METRICS_B_NORMED.glob(f'b_{model_name}_{embedding_name}_*_training_log.csv'),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _load_logs_normed(run_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Return {run_id: DataFrame} for normalised runs that have training logs."""
    loaded: dict[str, pd.DataFrame] = {}
    for run_id in run_ids:
        cfg      = _EXPERIMENTS_NORMED[run_id]
        log_path = _resolve_log_normed(cfg['model_name'], cfg['embedding_name'])
        if log_path is None:
            print(f'  [skip] {run_id} — no normalised training log found')
            continue
        df = pd.read_csv(log_path)
        if df.empty:
            print(f'  [skip] {run_id} — normalised training log is empty')
            continue
        loaded[run_id] = df
        print(f'  [load] {run_id} ← {log_path.name}  ({len(df)} epochs)')
    return loaded


# ══════════════════════════════════════════════════════════════════════════════
# Individual plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_individual_loss(run_id: str, df: pd.DataFrame, out_dir: Path) -> None:
    """
    Train loss vs val loss for a single run.
    Marks best epoch (min val_loss) with a vertical dotted line.
    Marks LR-reduction epochs with ▼ on the x-axis.
    """
    epochs     = df['epoch'].tolist()
    train_loss = df['train_loss'].tolist()
    val_loss   = df['val_loss'].tolist()
    lrs        = df['lr'].tolist()

    best_epoch = epochs[int(df['val_loss'].idxmin())]
    lr_drops   = [epochs[i] for i in range(1, len(lrs)) if lrs[i] < lrs[i - 1]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, '-',  color='steelblue', linewidth=1.8, label='train_loss')
    ax.plot(epochs, val_loss,   '--', color='tomato',    linewidth=1.8, label='val_loss')
    ax.axvline(best_epoch, linestyle=':', color='gray', linewidth=1.2,
               label=f'best epoch ({best_epoch})')

    y_bot = ax.get_ylim()[0]
    for ep in lr_drops:
        ax.annotate('▼', xy=(ep, y_bot), fontsize=10, color='purple',
                    ha='center', va='bottom')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Curves — {run_id} '
                 f'({_EXPERIMENTS[run_id]["embedding_name"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = out_dir / f'{run_id.lower()}_loss_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_individual_val_metrics(run_id: str, df: pd.DataFrame, out_dir: Path) -> None:
    """
    val_top1, val_mrr, val_mean_rank on dual y-axis for a single run.
    Skipped if columns are absent.
    """
    required = {'val_top1', 'val_mrr', 'val_mean_rank'}
    if not required.issubset(df.columns):
        print(f'  [skip] {run_id} val_metrics — missing columns: '
              f'{required - set(df.columns)}')
        return

    epochs    = df['epoch'].tolist()
    top1      = df['val_top1'].tolist()
    mrr       = df['val_mrr'].tolist()
    mean_rank = df['val_mean_rank'].tolist()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(epochs, top1, '-',  color='steelblue', linewidth=1.8, label='val_top1')
    ax1.plot(epochs, mrr,  '--', color='seagreen',  linewidth=1.8, label='val_mrr')
    ax2.plot(epochs, mean_rank, ':',  color='tomato',    linewidth=1.8, label='val_mean_rank')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('val_top1 / val_mrr', color='steelblue')
    ax2.set_ylabel('val_mean_rank (↓ better)', color='tomato')
    ax2.invert_yaxis()

    ax1.set_title(f'Val Metrics — {run_id} '
                  f'({_EXPERIMENTS[run_id]["embedding_name"]})')

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    ax1.grid(True, alpha=0.3)

    out = out_dir / f'{run_id.lower()}_val_metrics.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate plots
# ══════════════════════════════════════════════════════════════════════════════

def _overlaid_curve(
    logs:    dict[str, pd.DataFrame],
    column:  str,
    ylabel:  str,
    title:   str,
    out:     Path,
    invert:  bool = False,
) -> None:
    """
    Plot one column from every run's training log on a single axis.
    Skips runs that don't have the column.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for run_id, df in logs.items():
        if column not in df.columns:
            continue
        colour = _PALETTE.get(run_id, _PALETTE_NORMED.get(run_id, '#333333'))
        cfg    = _EXPERIMENTS.get(run_id) or _EXPERIMENTS_NORMED.get(run_id, {})
        emb    = cfg.get('embedding_name', run_id)
        ax.plot(
            df['epoch'].tolist(),
            df[column].tolist(),
            color=colour, linewidth=1.8,
            label=f'{run_id} ({emb})',
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print(f'  [skip] aggregate {column} — no data')
        return

    if invert:
        ax.invert_yaxis()

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_aggregate_val_loss(logs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    _overlaid_curve(
        logs, column='val_loss',
        ylabel='val_loss',
        title='Val Loss Convergence — All Embeddings (cnn_1layer)',
        out=out_dir / 'aggregate_val_loss.png',
    )


def plot_aggregate_val_top1(logs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    _overlaid_curve(
        logs, column='val_top1',
        ylabel='val_top1 accuracy',
        title='Val Top-1 Accuracy — All Embeddings (cnn_1layer)',
        out=out_dir / 'aggregate_val_top1.png',
    )


def plot_aggregate_val_mrr(logs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    _overlaid_curve(
        logs, column='val_mrr',
        ylabel='val_mrr',
        title='Val MRR — All Embeddings (cnn_1layer)',
        out=out_dir / 'aggregate_val_mrr.png',
    )


def plot_aggregate_val_mean_rank(logs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    _overlaid_curve(
        logs, column='val_mean_rank',
        ylabel='val_mean_rank  (↓ better)',
        title='Val Mean Rank — All Embeddings (cnn_1layer)',
        out=out_dir / 'aggregate_val_mean_rank.png',
        invert=True,
    )


def plot_aggregate_grid_loss(logs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """
    Grid of subplots — one subplot per run showing train_loss + val_loss.
    Layout: ceil(sqrt(N)) × ceil(sqrt(N)).
    """
    n     = len(logs)
    if n == 0:
        print('  [skip] aggregate grid — no data')
        return

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle('Train vs Val Loss — All Embeddings (cnn_1layer)',
                 fontsize=13, y=1.01)

    for ax_idx, (run_id, df) in enumerate(logs.items()):
        row, col = divmod(ax_idx, ncols)
        ax       = axes[row][col]
        colour   = _PALETTE.get(run_id, _PALETTE_NORMED.get(run_id, '#333333'))
        cfg      = _EXPERIMENTS.get(run_id) or _EXPERIMENTS_NORMED.get(run_id, {})
        emb      = cfg.get('embedding_name', run_id)

        epochs     = df['epoch'].tolist()
        train_loss = df['train_loss'].tolist()
        val_loss   = df['val_loss'].tolist()
        best_ep    = epochs[int(df['val_loss'].idxmin())]

        ax.plot(epochs, train_loss, '-',  color=colour,     linewidth=1.4, label='train')
        ax.plot(epochs, val_loss,   '--', color='tomato',   linewidth=1.4, label='val')
        ax.axvline(best_ep, linestyle=':', color='gray', linewidth=1.0)
        ax.set_title(f'{run_id}  ({emb})', fontsize=9)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Loss',  fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax_idx in range(n, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out = out_dir / 'aggregate_grid_loss.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Normed vs Base comparison
# ══════════════════════════════════════════════════════════════════════════════

def _best_row(df: pd.DataFrame) -> pd.Series:
    """Return the row at the epoch with minimum val_loss."""
    return df.loc[df['val_loss'].idxmin()]


def _get_metric(row: pd.Series, col: str) -> float | None:
    """Return metric value from a best-epoch row, or None if column absent."""
    return float(row[col]) if col in row.index else None


def plot_comparison_per_embedding(
    normed_id: str,
    df_normed:  pd.DataFrame,
    df_base:    pd.DataFrame | None,
    out_dir:    Path,
) -> None:
    """
    3-panel figure for one embedding: val_loss | val_top1 | val_mrr.
    Base (solid) and normed (dashed) are overlaid on the same axes.
    If df_base is None, only the normed curve is plotted.
    """
    cfg_n   = _EXPERIMENTS_NORMED[normed_id]
    emb_raw = cfg_n['embedding_name']                        # e.g. 'sbert_normed'
    emb_base = emb_raw.replace('_normed', '')                # e.g. 'sbert'

    metrics = [
        ('val_loss',      'val_loss',              False),
        ('val_top1',      'val_top1 accuracy',     False),
        ('val_mrr',       'val_mrr',               False),
        ('val_mean_rank', 'val_mean_rank (↓ best)', True),
    ]
    available = [m for m in metrics if m[0] in df_normed.columns]
    if not available:
        print(f'  [skip] comparison {normed_id} — no plottable columns')
        return

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    fig.suptitle(f'Normed vs Base — {emb_base}  (cnn_1layer)', fontsize=12)

    colour_base  = _PALETTE.get(cfg_n['base_run'], '#888888') if cfg_n['base_run'] else '#888888'
    colour_normed = _PALETTE_NORMED.get(normed_id, '#AAAAAA')

    for ax, (col, ylabel, invert) in zip(axes[0], available):
        epochs_n = df_normed['epoch'].tolist()
        ax.plot(epochs_n, df_normed[col].tolist(),
                '--', color=colour_normed, linewidth=1.8, label=f'{emb_raw}')

        if df_base is not None and col in df_base.columns:
            epochs_b = df_base['epoch'].tolist()
            ax.plot(epochs_b, df_base[col].tolist(),
                    '-', color=colour_base, linewidth=1.8, label=emb_base)

        if invert:
            ax.invert_yaxis()

        # Mark best epoch for each series
        best_ep_n = df_normed.loc[df_normed['val_loss'].idxmin(), 'epoch']
        ax.axvline(best_ep_n, linestyle=':', color=colour_normed, linewidth=1.0, alpha=0.7)
        if df_base is not None:
            best_ep_b = df_base.loc[df_base['val_loss'].idxmin(), 'epoch']
            ax.axvline(best_ep_b, linestyle=':', color=colour_base, linewidth=1.0, alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / f'{emb_base}_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def plot_comparison_aggregate_grid(
    logs_normed: dict[str, pd.DataFrame],
    logs_base:   dict[str, pd.DataFrame],
    out_dir:     Path,
    column:      str = 'val_loss',
    ylabel:      str = 'val_loss',
) -> None:
    """
    Grid of subplots — one cell per normed run.
    Each cell overlays base (solid) and normed (dashed) val_loss.
    Runs with no base counterpart show normed only.
    """
    n = len(logs_normed)
    if n == 0:
        print('  [skip] comparison aggregate grid — no normed data')
        return

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle(f'Normed vs Base — {column} — All Embeddings (cnn_1layer)',
                 fontsize=12, y=1.01)

    for ax_idx, (normed_id, df_n) in enumerate(logs_normed.items()):
        row, col = divmod(ax_idx, ncols)
        ax       = axes[row][col]
        cfg_n    = _EXPERIMENTS_NORMED[normed_id]
        emb_base = cfg_n['embedding_name'].replace('_normed', '')

        c_normed = _PALETTE_NORMED.get(normed_id, '#AAAAAA')
        c_base   = _PALETTE.get(cfg_n['base_run'], '#888888') if cfg_n['base_run'] else '#888888'

        if column in df_n.columns:
            ax.plot(df_n['epoch'].tolist(), df_n[column].tolist(),
                    '--', color=c_normed, linewidth=1.4, label='normed')

        base_run_id = cfg_n['base_run']
        if base_run_id and base_run_id in logs_base and column in logs_base[base_run_id].columns:
            df_b = logs_base[base_run_id]
            ax.plot(df_b['epoch'].tolist(), df_b[column].tolist(),
                    '-', color=c_base, linewidth=1.4, label='base')

        ax.set_title(f'{emb_base}', fontsize=9)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel(ylabel,  fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax_idx in range(n, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out = out_dir / 'aggregate_comparison_grid.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] {out.name}')


def save_comparison_csv(
    logs_normed: dict[str, pd.DataFrame],
    logs_base:   dict[str, pd.DataFrame],
    out_dir:     Path,
) -> None:
    """
    Build a summary CSV comparing best-epoch metrics for base vs normed.

    Columns
    -------
    embedding           : base embedding name (e.g. 'sbert')
    normed_run_id       : run ID in _EXPERIMENTS_NORMED (e.g. 'E2an')
    base_run_id         : run ID in _EXPERIMENTS (e.g. 'E2a'), or '' if no base
    {metric}_base       : metric value at best (argmin val_loss) epoch — base run
    {metric}_normed     : same for normed run
    {metric}_delta      : normed − base  (positive = normed better for ↑ metrics,
                          negative = normed better for ↓ metrics)

    Metrics reported: val_loss, val_top1, val_mrr, val_mean_rank, val_mean_cosine
    """
    _METRICS = ['val_loss', 'val_top1', 'val_mrr', 'val_mean_rank', 'val_mean_cosine']

    rows = []
    for normed_id, df_n in logs_normed.items():
        cfg_n        = _EXPERIMENTS_NORMED[normed_id]
        base_run_id  = cfg_n['base_run'] or ''
        emb_base     = cfg_n['embedding_name'].replace('_normed', '')

        best_n = _best_row(df_n)
        df_b   = logs_base.get(base_run_id) if base_run_id else None
        best_b = _best_row(df_b) if df_b is not None else None

        record: dict = {
            'embedding':    emb_base,
            'normed_run_id': normed_id,
            'base_run_id':  base_run_id,
            'best_epoch_normed': int(best_n['epoch']),
            'best_epoch_base':   int(best_b['epoch']) if best_b is not None else None,
        }

        for metric in _METRICS:
            v_normed = _get_metric(best_n, metric)
            v_base   = _get_metric(best_b, metric) if best_b is not None else None
            record[f'{metric}_normed'] = v_normed
            record[f'{metric}_base']   = v_base
            if v_normed is not None and v_base is not None:
                record[f'{metric}_delta'] = round(v_normed - v_base, 6)
            else:
                record[f'{metric}_delta'] = None

        rows.append(record)

    if not rows:
        print('  [skip] comparison CSV — no data')
        return

    df_out = pd.DataFrame(rows)
    # Reorder columns for readability
    fixed_cols = ['embedding', 'normed_run_id', 'base_run_id',
                  'best_epoch_normed', 'best_epoch_base']
    metric_cols = [c for c in df_out.columns if c not in fixed_cols]
    df_out = df_out[fixed_cols + sorted(metric_cols)]

    out = out_dir / 'normed_vs_base_summary.csv'
    df_out.to_csv(out, index=False)
    print(f'  [saved] {out.name}  ({len(df_out)} rows)')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot training curves for Type-B embedding-axis experiments'
    )
    parser.add_argument(
        '--runs', nargs='+', default=list(_EXPERIMENTS),
        metavar='RUN_ID',
        help='Base run IDs to plot (default: all). E.g. --runs B0 E2a E2b',
    )
    parser.add_argument(
        '--normed-runs', nargs='+', default=list(_EXPERIMENTS_NORMED),
        metavar='RUN_ID',
        help='Normalised run IDs to plot (default: all). E.g. --normed-runs E2an E2bn',
    )
    parser.add_argument(
        '--no-individual', action='store_true',
        help='Skip individual per-run plots',
    )
    parser.add_argument(
        '--no-aggregate', action='store_true',
        help='Skip aggregate (all-runs) plots',
    )
    parser.add_argument(
        '--normalised', action='store_true',
        help='Also produce plots for normalised embedding runs',
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Produce normed-vs-base comparison plots and summary CSV',
    )
    args = parser.parse_args()

    unknown_base = [r for r in args.runs if r not in _EXPERIMENTS]
    if unknown_base:
        parser.error(f'Unknown base run IDs: {unknown_base}. Available: {list(_EXPERIMENTS)}')

    unknown_norm = [r for r in args.normed_runs if r not in _EXPERIMENTS_NORMED]
    if unknown_norm:
        parser.error(f'Unknown normed run IDs: {unknown_norm}. Available: {list(_EXPERIMENTS_NORMED)}')

    FIGURES_TRAIN_B.mkdir(parents=True, exist_ok=True)
    if args.normalised or args.compare:
        FIGURES_TRAIN_NORM.mkdir(parents=True, exist_ok=True)
    if args.compare:
        COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    print('\nType-B Training Curve Plotter')
    print(f'Base runs    : {args.runs}')
    print(f'Normed runs  : {args.normed_runs if (args.normalised or args.compare) else "(skipped)"}')
    print(f'Output (base): {FIGURES_TRAIN_B}')

    # ── Base runs ──────────────────────────────────────────────────────────────
    logs = _load_logs(args.runs)
    if not logs:
        print('\nNo base training logs found. Skipping base plots.')
    else:
        if not args.no_individual:
            print('\n── Base individual plots ──')
            for run_id, df in logs.items():
                plot_individual_loss(run_id, df, FIGURES_TRAIN_B)
                plot_individual_val_metrics(run_id, df, FIGURES_TRAIN_B)

        if not args.no_aggregate:
            print('\n── Base aggregate plots ──')
            plot_aggregate_val_loss(logs, FIGURES_TRAIN_B)
            plot_aggregate_val_top1(logs, FIGURES_TRAIN_B)
            plot_aggregate_val_mrr(logs, FIGURES_TRAIN_B)
            plot_aggregate_val_mean_rank(logs, FIGURES_TRAIN_B)
            plot_aggregate_grid_loss(logs, FIGURES_TRAIN_B)

    # ── Normalised runs ────────────────────────────────────────────────────────
    logs_normed: dict[str, pd.DataFrame] = {}
    if args.normalised or args.compare:
        print(f'\n── Loading normalised logs ──')
        logs_normed = _load_logs_normed(args.normed_runs)

        if logs_normed and args.normalised:
            if not args.no_individual:
                print('\n── Normalised individual plots ──')
                for run_id, df in logs_normed.items():
                    # Reuse existing per-run plotters; they use _EXPERIMENTS for labels,
                    # so pass normed metadata via a temporary monkey-patch is avoided —
                    # instead call the generic _overlaid_curve helper directly.
                    cfg    = _EXPERIMENTS_NORMED[run_id]
                    emb    = cfg['embedding_name']
                    colour = _PALETTE_NORMED.get(run_id, '#AAAAAA')
                    epochs = df['epoch'].tolist()

                    # Loss curves
                    fig, ax = plt.subplots(figsize=(9, 5))
                    ax.plot(epochs, df['train_loss'].tolist(), '-',
                            color='steelblue', linewidth=1.8, label='train_loss')
                    ax.plot(epochs, df['val_loss'].tolist(),   '--',
                            color='tomato',    linewidth=1.8, label='val_loss')
                    best_ep = epochs[int(df['val_loss'].idxmin())]
                    ax.axvline(best_ep, linestyle=':', color='gray', linewidth=1.2,
                               label=f'best epoch ({best_ep})')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'Loss Curves — {run_id} ({emb})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    out = FIGURES_TRAIN_NORM / f'{run_id.lower()}_loss_curves.png'
                    fig.savefig(out, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f'  [saved] {out.name}')

                    # Val metrics
                    required = {'val_top1', 'val_mrr', 'val_mean_rank'}
                    if required.issubset(df.columns):
                        fig, ax1 = plt.subplots(figsize=(9, 5))
                        ax2 = ax1.twinx()
                        ax1.plot(epochs, df['val_top1'].tolist(), '-',  color='steelblue', linewidth=1.8, label='val_top1')
                        ax1.plot(epochs, df['val_mrr'].tolist(),  '--', color='seagreen',  linewidth=1.8, label='val_mrr')
                        ax2.plot(epochs, df['val_mean_rank'].tolist(), ':', color='tomato', linewidth=1.8, label='val_mean_rank')
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('val_top1 / val_mrr', color='steelblue')
                        ax2.set_ylabel('val_mean_rank (↓ better)', color='tomato')
                        ax2.invert_yaxis()
                        ax1.set_title(f'Val Metrics — {run_id} ({emb})')
                        lines  = ax1.get_lines() + ax2.get_lines()
                        labels = [l.get_label() for l in lines]
                        ax1.legend(lines, labels, loc='upper right')
                        ax1.grid(True, alpha=0.3)
                        out = FIGURES_TRAIN_NORM / f'{run_id.lower()}_val_metrics.png'
                        fig.savefig(out, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        print(f'  [saved] {out.name}')

            if not args.no_aggregate:
                print('\n── Normalised aggregate plots ──')
                _overlaid_curve(logs_normed, 'val_loss', 'val_loss',
                                'Val Loss — Normalised Runs (cnn_1layer)',
                                FIGURES_TRAIN_NORM / 'aggregate_val_loss.png')
                _overlaid_curve(logs_normed, 'val_top1', 'val_top1 accuracy',
                                'Val Top-1 — Normalised Runs (cnn_1layer)',
                                FIGURES_TRAIN_NORM / 'aggregate_val_top1.png')
                _overlaid_curve(logs_normed, 'val_mrr', 'val_mrr',
                                'Val MRR — Normalised Runs (cnn_1layer)',
                                FIGURES_TRAIN_NORM / 'aggregate_val_mrr.png')
                _overlaid_curve(logs_normed, 'val_mean_rank', 'val_mean_rank (↓ better)',
                                'Val Mean Rank — Normalised Runs (cnn_1layer)',
                                FIGURES_TRAIN_NORM / 'aggregate_val_mean_rank.png',
                                invert=True)
                # Grid loss for normed — inline (avoids _EXPERIMENTS coupling)
                n = len(logs_normed)
                ncols = math.ceil(math.sqrt(n))
                nrows = math.ceil(n / ncols)
                fig, axes = plt.subplots(nrows, ncols,
                                         figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
                fig.suptitle('Train vs Val Loss — Normalised Runs (cnn_1layer)',
                             fontsize=13, y=1.01)
                for ax_idx, (run_id, df) in enumerate(logs_normed.items()):
                    r, c = divmod(ax_idx, ncols)
                    ax   = axes[r][c]
                    emb  = _EXPERIMENTS_NORMED[run_id]['embedding_name']
                    best_ep = df['epoch'].tolist()[int(df['val_loss'].idxmin())]
                    ax.plot(df['epoch'].tolist(), df['train_loss'].tolist(),
                            '-',  color=_PALETTE_NORMED.get(run_id, '#999'), linewidth=1.4, label='train')
                    ax.plot(df['epoch'].tolist(), df['val_loss'].tolist(),
                            '--', color='tomato', linewidth=1.4, label='val')
                    ax.axvline(best_ep, linestyle=':', color='gray', linewidth=1.0)
                    ax.set_title(f'{run_id}  ({emb})', fontsize=9)
                    ax.set_xlabel('Epoch', fontsize=8)
                    ax.set_ylabel('Loss',  fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                for ax_idx in range(n, nrows * ncols):
                    r, c = divmod(ax_idx, ncols)
                    axes[r][c].set_visible(False)
                fig.tight_layout()
                out = FIGURES_TRAIN_NORM / 'aggregate_grid_loss.png'
                fig.savefig(out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'  [saved] {out.name}')

    # ── Comparison (normed vs base) ────────────────────────────────────────────
    if args.compare and logs_normed:
        print('\n── Comparison plots (normed vs base) ──')
        for normed_id, df_n in logs_normed.items():
            base_run_id = _EXPERIMENTS_NORMED[normed_id]['base_run']
            df_base     = logs.get(base_run_id) if base_run_id else None
            plot_comparison_per_embedding(normed_id, df_n, df_base, COMPARISON_DIR)

        print('\n── Comparison aggregate grid ──')
        plot_comparison_aggregate_grid(logs_normed, logs, COMPARISON_DIR)

        print('\n── Comparison CSV ──')
        save_comparison_csv(logs_normed, logs, COMPARISON_DIR)

    print(f'\nDone')
    print(f'  Base figures  → {FIGURES_TRAIN_B}')
    if args.normalised:
        print(f'  Normed figures→ {FIGURES_TRAIN_NORM}')
    if args.compare:
        print(f'  Comparison    → {COMPARISON_DIR}')


if __name__ == '__main__':
    main()
