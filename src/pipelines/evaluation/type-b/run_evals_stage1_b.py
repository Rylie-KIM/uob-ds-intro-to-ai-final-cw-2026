# Output paths
#   Predictions  : src/pipelines/results/metrics/type-b/prediction/
#   Test results : src/pipelines/results/metrics/type-b/prediction/test_results.csv
#   Figures      : src/pipelines/results/figures/type-b/evaluation/
#                  (comparison_*.png saved here, same level as normed/ subfolder)

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_ROOT     = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
_EVAL_DIR = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_metrics_b import run_evaluation, plot_cross_run_graphs  # noqa: E402
from src.config.paths import CHECKPOINTS_B, PREDICTIONS_B, FIGURES_EVAL_B  # noqa: E402



# Keys must match the checkpoint naming convention:
#   b_{model_name}_{embedding_name}_{tag}_{timestamp}_best.pt


EXPERIMENTS: dict[str, dict] = {
    # Stage 0 — true baseline (statistical floor)
    'B0': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tfidf_lsa',
        'loss_fn':        'MSE',
        'description':    'Baseline — TF-IDF + LSA (100-dim), MSE',
    },
    # Stage 1 — embedding axis (all use cnn_1layer to isolate embedding effect)
    'E2a': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'sbert',
        'loss_fn':        'Cosine',
        'description':    'SBERT all-MiniLM-L6-v2 (384-dim), CosineLoss',
    },
    'E2b': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'sbert_finetuned',
        'loss_fn':        'Cosine',
        'description':    'SBERT fine-tuned on Type-B corpus (384-dim), CosineLoss',
    },
    'E2e': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'MSE',
        'description':    'TinyBERT mean-pool (312-dim), MSE',
    },
    'E2f': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tinybert_pooler',
        'loss_fn':        'MSE',
        'description':    'TinyBERT [CLS] pooler (312-dim), MSE',
    },
    'E2g': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'glove',
        'loss_fn':        'MSE',
        'description':    'GloVe word-avg (300-dim), MSE',
    },
    'E2h': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'word2vec_pretrained',
        'loss_fn':        'MSE',
        'description':    'Word2Vec Google News pretrained (300-dim), MSE',
    },
    'E2i': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'word2vec_skipgram',
        'loss_fn':        'MSE',
        'description':    'Word2Vec skip-gram trained on Type-B corpus (100-dim), MSE',
    },
    'E2k': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tfidf_w2v',
        'loss_fn':        'MSE',
        'description':    'TF-IDF weighted Word2Vec (100-dim), MSE',
    },
}


def _resolve_checkpoint(model_name: str, embedding_name: str) -> Path | None:
    """
    Return the most recently modified checkpoint for this (model, embedding) pair,
    or None if no checkpoint exists.

    Pattern: b_{model_name}_{embedding_name}_*_best.pt
    """
    candidates = sorted(
        CHECKPOINTS_B.glob(f'b_{model_name}_{embedding_name}_*_best.pt'),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _run_one(
    run_id: str,
    cfg:    dict,
    device: str | None,
) -> dict | None:
    """
    Evaluate one experiment. Returns the metrics dict on success, None otherwise.
    """
    model_name     = cfg['model_name']
    embedding_name = cfg['embedding_name']
    loss_fn        = cfg['loss_fn']

    ckpt_path = _resolve_checkpoint(model_name, embedding_name)
    if ckpt_path is None:
        print(f'\n  [skip] {run_id} — no checkpoint found for '
              f'{model_name} + {embedding_name}')
        return None

    try:
        metrics = run_evaluation(
            run_id=run_id,
            model_name=model_name,
            embedding_name=embedding_name,
            loss_fn=loss_fn,
            device=device,
            ckpt_path=ckpt_path,
        )
        return metrics
    except FileNotFoundError as exc:
        print(f'\n  [skip] {run_id} — {exc}')
        return None
    except Exception:
        print(f'\n  [error] {run_id} — unexpected error:')
        traceback.print_exc()
        return None


def _print_comparison_table(results: dict[str, dict]) -> None:
    """
    Print a ranked comparison table sorted by test_top1 desc, then test_mrr desc.
    """
    if not results:
        print('\n  No results to compare.')
        return

    rows = sorted(
        results.items(),
        key=lambda kv: (kv[1].get('test_top1', -1), kv[1].get('test_mrr', -1)),
        reverse=True,
    )

    header = (
        f"\n{'='*82}\n"
        f"  Stage-1 Embedding Comparison — Type-B (cnn_1layer, test set)\n"
        f"{'='*82}\n"
        f"  {'Run':<6} {'Embedding':<22} {'top-1':>6} {'top-5':>6} "
        f"{'MRR':>7} {'mean_rank':>10} {'cosine':>8}\n"
        f"  {'-'*76}"
    )
    print(header)

    for run_id, m in rows:
        emb = EXPERIMENTS[run_id]['embedding_name']
        print(
            f"  {run_id:<6} {emb:<22} "
            f"{m.get('test_top1', float('nan')):>6.4f} "
            f"{m.get('test_top5', float('nan')):>6.4f} "
            f"{m.get('test_mrr',  float('nan')):>7.4f} "
            f"{m.get('test_mean_rank', float('nan')):>10.1f} "
            f"{m.get('test_mean_cosine', float('nan')):>8.4f}"
        )

    print(f"  {'='*76}")
    print(f"  Sorted by: test_top1 ↓, test_mrr ↓")
    print(f"  Full results → {PREDICTIONS_B / 'test_results.csv'}")
    print(f"{'='*82}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage-1 batch evaluator for Type-B embedding-axis experiments'
    )
    parser.add_argument(
        '--runs', nargs='+', default=list(EXPERIMENTS),
        metavar='RUN_ID',
        help='Experiment IDs to evaluate (default: all). E.g. --runs B0 E2a E2b',
    )
    parser.add_argument(
        '--device', default=None,
        help='Device: cpu | cuda | mps (default: auto-detect)',
    )
    parser.add_argument(
        '--no-cross-graphs', action='store_true',
        help='Skip cross-run comparison graphs after evaluation',
    )
    args = parser.parse_args()

    # Validate
    unknown = [r for r in args.runs if r not in EXPERIMENTS]
    if unknown:
        parser.error(f'Unknown run IDs: {unknown}. Available: {list(EXPERIMENTS)}')

    print(f'\nType-B Stage-1 Evaluation')
    print(f'Runs   : {args.runs}')
    print(f'Device : {args.device or "auto"}')
    print(f'Results: {METRICS_B}')


    completed: dict[str, dict] = {}   # run_id → metrics
    skipped:   list[str]       = []

    for run_id in args.runs:
        cfg     = EXPERIMENTS[run_id]
        metrics = _run_one(run_id, cfg, device=args.device)
        if metrics is not None:
            completed[run_id] = metrics
        else:
            skipped.append(run_id)


    _print_comparison_table(completed)

    if not args.no_cross_graphs and completed:
        print(f'{"="*60}')
        print('  Generating cross-run comparison graphs…')
        print(f'{"="*60}')
        test_results_csv = PREDICTIONS_B / 'test_results.csv'
        if test_results_csv.exists():
            FIGURES_EVAL_B.mkdir(parents=True, exist_ok=True)
            plot_cross_run_graphs(
                test_results_csv=test_results_csv,
                predictions_dir=PREDICTIONS_B,
                figures_dir=FIGURES_EVAL_B,
            )
        else:
            print('  [skip] test_results.csv not found — run evaluations first.')

    print(f'{"="*60}')
    print(f'  Completed : {list(completed)}')
    if skipped:
        print(f'  Skipped   : {skipped}  (checkpoint not found)')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
