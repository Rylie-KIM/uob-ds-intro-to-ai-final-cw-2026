"""
src/pipelines/type-b/pipeline_b.py
End-to-end pipeline for the Type-B (coloured MNIST numbers) experiments.

Pipeline stages
---------------
1. [Optional] Generate sentence embeddings (skip if .pt files already exist)
2. Train all model x embedding combinations
3. Evaluate each trained model on the test set
4. Plot loss curves (epoch_log CSV -> matplotlib figure)
5. Plot results heatmap (results_summary CSV -> seaborn heatmap)
6. [Optional] Run Gemini Vision API comparison

Usage
-----
# Full pipeline (all models x all embeddings, auto-detect device)
python src/pipelines/type-b/pipeline_b.py

# Subset run
python src/pipelines/type-b/pipeline_b.py --models cnn cnn_1layer --embeddings sbert tfidf

# Skip embedding generation (already done), also run Gemini comparison
python src/pipelines/type-b/pipeline_b.py --skip-embed --run-gemini

# Quick smoke test (2 epochs)
python src/pipelines/type-b/pipeline_b.py --models cnn --embeddings tfidf --epochs 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import EMBED_RESULTS_B, METRICS_B, FIGURES_DIR

# src/pipelines/training/type-b/ uses a hyphen — not directly importable.
# Load the module from its file path using importlib.
import importlib.util as _ilu

_train_spec = _ilu.spec_from_file_location(
    'train_type_b',
    _ROOT / 'src' / 'pipelines' / 'training' / 'type-b' / 'train_type_b.py',
)
_train_mod = _ilu.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train_mod)

EMBEDDING_CONFIGS = _train_mod.EMBEDDING_CONFIGS
MODEL_CONFIGS     = _train_mod.MODEL_CONFIGS
run_experiment    = _train_mod.run_experiment
_default_device   = _train_mod._default_device


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Embedding generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_embeddings(embedding_names: list[str]) -> None:
    """Generate pre-computed embeddings for the specified methods (skip if exists)."""
    script = _ROOT / 'src' / 'embeddings' / 'computed-embeddings' / 'type-b' / \
             'generate_embeddings_type_b.py'

    for emb in embedding_names:
        out = EMBED_RESULTS_B / f'{emb}_embedding_result_typeb.pt'
        if out.exists():
            print(f'[embed] Skip {emb} — already exists')
            continue
        print(f'[embed] Generating {emb}...')
        subprocess.run(
            [sys.executable, str(script), '--embedding', emb],
            check=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 & 3: Training and evaluation
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    models:     list[str],
    embeddings: list[str],
    epochs:     int,
    device:     str,
) -> None:
    """Run run_experiment() for each model x embedding combination."""
    total = len(models) * len(embeddings)
    done  = 0
    for model_name in models:
        for emb_name in embeddings:
            done += 1
            print(f'\n[pipeline] ({done}/{total}) {model_name} x {emb_name}')
            run_experiment(model_name, emb_name, epochs=epochs, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4: Loss curve plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(models: list[str], embeddings: list[str]) -> None:
    """
    Plot train/val loss curves from epoch_log CSV files.
    Saves one PNG per model, overlaying all embedding methods on the same axes.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        fig.suptitle(f'Loss Curves — {model_name} (Type-B)', fontsize=13)

        for emb_name in embeddings:
            log_path = METRICS_B / f'b_{model_name}_{emb_name}_epoch_log.csv'
            if not log_path.exists():
                continue
            df = pd.read_csv(log_path)
            axes[0].plot(df['epoch'], df['train_loss'], label=emb_name, alpha=0.8)
            axes[1].plot(df['epoch'], df['val_loss'],   label=emb_name, alpha=0.8)

        axes[0].set_title('Training Loss')
        axes[1].set_title('Validation Loss')
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

        out = FIGURES_DIR / f'loss_curves_{model_name}_typeb.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[plot] Loss curves saved → {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5: Results heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_results_heatmap(models: list[str], embeddings: list[str]) -> None:
    """
    Plot top-1 accuracy as a heatmap: rows = embeddings, cols = models.
    Saves to results/figures/results_heatmap_typeb.png.
    """
    summary_path = METRICS_B / 'results_summary.csv'
    if not summary_path.exists():
        print('[plot] results_summary.csv not found — skipping heatmap')
        return

    try:
        import seaborn as sns
    except ImportError:
        print('[plot] seaborn not installed — skipping heatmap (pip install seaborn)')
        return

    df      = pd.read_csv(summary_path)
    df['model']     = df['run_name'].str.split('_').str[1]
    df['embedding'] = df['run_name'].str.split('_', n=2).str[2]

    pivot = df.pivot(index='embedding', columns='model', values='top_1_acc')
    # Reorder to match plan order
    pivot = pivot.reindex(index=[e for e in embeddings if e in pivot.index],
                          columns=[m for m in models if m in pivot.columns])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot, annot=True, fmt='.3f', cmap='YlOrRd',
        vmin=0.0, vmax=1.0, ax=ax, linewidths=0.5,
    )
    ax.set_title('Top-1 Retrieval Accuracy — Type-B\n(rows: embedding, cols: model)')
    ax.set_ylabel('Embedding method')
    ax.set_xlabel('CNN model')

    out = FIGURES_DIR / 'results_heatmap_typeb.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] Heatmap saved → {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Stage 6: Gemini comparison (optional)
# ══════════════════════════════════════════════════════════════════════════════

def run_gemini_comparison() -> None:
    """Run Gemini Vision API comparison on the test set."""
    script = _ROOT / 'src' / 'pipelines' / 'evaluation' / 'type-b' /'gemini_comparison.py'
    if not script.exists():
        print('[gemini] gemini_comparison.py not found — skipping')
        return
    print('[gemini] Running Gemini comparison...')
    subprocess.run([sys.executable, str(script)], check=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    models:      list[str],
    embeddings:  list[str],
    epochs:      int,
    device:      str,
    skip_embed:  bool = False,
    run_gemini:  bool = False,
) -> None:
    print(f'\n{"#"*60}')
    print(f'# Type-B Pipeline')
    print(f'# Models    : {models}')
    print(f'# Embeddings: {embeddings}')
    print(f'# Epochs    : {epochs}    Device: {device}')
    print(f'{"#"*60}\n')

    if not skip_embed:
        print('Stage 1: Generating embeddings...')
        generate_embeddings(embeddings)

    print('\nStage 2-3: Training and evaluating...')
    train_and_evaluate(models, embeddings, epochs, device)

    print('\nStage 4: Plotting loss curves...')
    plot_loss_curves(models, embeddings)

    print('\nStage 5: Plotting results heatmap...')
    plot_results_heatmap(models, embeddings)

    if run_gemini:
        print('\nStage 6: Gemini comparison...')
        run_gemini_comparison()

    print(f'\nPipeline complete.')
    print(f'  Summary : {METRICS_B / "results_summary.csv"}')
    print(f'  Figures : {FIGURES_DIR}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Type-B end-to-end pipeline')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_CONFIGS),
                        default=None, help='Models to run (default: all)')
    parser.add_argument('--embeddings', nargs='+', choices=list(EMBEDDING_CONFIGS),
                        default=None, help='Embeddings to run (default: all)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default=_default_device())
    parser.add_argument('--skip-embed', action='store_true',
                        help='Skip embedding generation stage')
    parser.add_argument('--run-gemini', action='store_true',
                        help='Run Gemini Vision API comparison after training')
    args = parser.parse_args()

    models     = args.models     or list(MODEL_CONFIGS)
    embeddings = args.embeddings or list(EMBEDDING_CONFIGS)

    run_pipeline(
        models=models,
        embeddings=embeddings,
        epochs=args.epochs,
        device=args.device,
        skip_embed=args.skip_embed,
        run_gemini=args.run_gemini,
    )


if __name__ == '__main__':
    main()
