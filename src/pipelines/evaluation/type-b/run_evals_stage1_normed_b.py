# Output paths
#   Predictions  : src/pipelines/results/metrics/type-b/prediction-normalised/
#   Test results : src/pipelines/results/metrics/type-b/prediction-normalised/test_results_normed.csv
#   Figures      : src/pipelines/results/figures/type-b/evaluation/normalised/

"""
Batch evaluator for Type-B normalised-embedding experiments.
Output Path: 
  metrics/type-b/prediction-normalised/
  figures/type-b/evaluation/normalised/

Checkpoint naming convention (normalised runs):
  b_{model}_{embedding}_normed_{tag}_{timestamp}_best.pt

Experiments
-----------
  Run   | Embedding              | Loss              | Notes
  ------|------------------------|-------------------|-----------------------
  B0n   | tfidf_lsa              | CosineLoss_normed | Baseline (normed)
  E2an  | sbert                  | CosineLoss_normed |
  E2bn  | sbert_finetuned        | CosineLoss_normed |
  E2en  | tinybert_mean          | CosineLoss_normed |
  E2fn  | tinybert_pooler        | CosineLoss_normed |
  E2gn  | glove                  | CosineLoss_normed |
  E2hn  | word2vec_pretrained    | CosineLoss_normed |
  E2in  | word2vec_skipgram      | CosineLoss_normed |
  E2kn  | tfidf_w2v              | CosineLoss_normed |
  E2ln  | bert_mean              | CosineLoss_normed | bert_mean (normed-only)
  E2mn  | bert_pooler            | CosineLoss_normed | bert_pooler (normed-only)

"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT     = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
_EVAL_DIR = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_metrics_b import run_evaluation_normed, plot_cross_run_graphs  # noqa: E402
from src.config.paths import (                                             # noqa: E402
    CHECKPOINTS_B_NORMED,
    PREDICTIONS_B_NORMED,
    FIGURES_EVAL_NORM_B,
)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment registry
# embedding_name is the BASE name (no '_normed').
# _resolve_checkpoint appends '_normed' to match the checkpoint filename.
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS_NORMED: dict[str, dict] = {
    'B0n': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tfidf_lsa',
        'loss_fn':        'CosineLoss_normed',
        'description':    'Baseline TF-IDF LSA (100-dim), L2-normed targets',
    },
    'E2an': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'sbert',
        'loss_fn':        'CosineLoss_normed',
        'description':    'SBERT all-MiniLM-L6-v2 (384-dim), L2-normed targets',
    },
    'E2bn': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'sbert_finetuned',
        'loss_fn':        'CosineLoss_normed',
        'description':    'SBERT fine-tuned on Type-B corpus (384-dim), L2-normed targets',
    },
    'E2en': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'CosineLoss_normed',
        'description':    'TinyBERT mean-pool (312-dim), L2-normed targets',
    },
    'E2fn': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tinybert_pooler',
        'loss_fn':        'CosineLoss_normed',
        'description':    'TinyBERT [CLS] pooler (312-dim), L2-normed targets',
    },
    'E2gn': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'glove',
        'loss_fn':        'CosineLoss_normed',
        'description':    'GloVe word-avg (300-dim), L2-normed targets',
    },
    'E2hn': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'word2vec_pretrained',
        'loss_fn':        'CosineLoss_normed',
        'description':    'Word2Vec Google News pretrained (300-dim), L2-normed targets',
    },
    'E2in': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'word2vec_skipgram',
        'loss_fn':        'CosineLoss_normed',
        'description':    'Word2Vec skip-gram on Type-B corpus (100-dim), L2-normed targets',
    },
    'E2kn': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tfidf_w2v',
        'loss_fn':        'CosineLoss_normed',
        'description':    'TF-IDF weighted Word2Vec (100-dim), L2-normed targets',
    },
    'E2ln': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'bert_mean',
        'loss_fn':        'CosineLoss_normed',
        'description':    'BERT base mean-pool (768-dim), L2-normed targets',
    },
    'E2mn': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'bert_pooler',
        'loss_fn':        'CosineLoss_normed',
        'description':    'BERT base [CLS] pooler (768-dim), L2-normed targets',
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_checkpoint(model_name: str, embedding_name: str) -> Path | None:
    """
    Return the most recently modified normalised checkpoint for this
    (model, embedding) pair, or None if no checkpoint exists.

    Pattern: b_{model}_{embedding}_normed_*_best.pt
    Searched in CHECKPOINTS_B_NORMED (checkpoints/type-b/normalised/).
    """
    candidates = sorted(
        CHECKPOINTS_B_NORMED.glob(
            f'b_{model_name}_{embedding_name}_normed_*_best.pt'
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _run_one(
    run_id: str,
    cfg:    dict,
    device: str | None,
) -> dict | None:
    """
    Evaluate one normalised experiment.
    Returns the metrics dict on success, None otherwise.
    """
    model_name     = cfg['model_name']
    embedding_name = cfg['embedding_name']
    loss_fn        = cfg['loss_fn']

    ckpt_path = _resolve_checkpoint(model_name, embedding_name)
    if ckpt_path is None:
        print(f'\n  [skip] {run_id} — no normalised checkpoint found for '
              f'{model_name} + {embedding_name}_normed')
        return None

    try:
        metrics = run_evaluation_normed(
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
        f"  Normalised Embedding Comparison — Type-B (cnn_1layer, L2-normed, test set)\n"
        f"{'='*82}\n"
        f"  {'Run':<7} {'Embedding':<24} {'top-1':>6} {'top-5':>6} "
        f"{'MRR':>7} {'mean_rank':>10} {'cosine':>8}\n"
        f"  {'-'*76}"
    )
    print(header)

    for run_id, m in rows:
        emb = EXPERIMENTS_NORMED[run_id]['embedding_name']
        print(
            f"  {run_id:<7} {emb + '_normed':<24} "
            f"{m.get('test_top1',        float('nan')):>6.4f} "
            f"{m.get('test_top5',        float('nan')):>6.4f} "
            f"{m.get('test_mrr',         float('nan')):>7.4f} "
            f"{m.get('test_mean_rank',   float('nan')):>10.1f} "
            f"{m.get('test_mean_cosine', float('nan')):>8.4f}"
        )

    print(f"  {'='*76}")
    print(f"  Sorted by: test_top1 ↓, test_mrr ↓")
    print(f"  Full results → {PREDICTIONS_B_NORMED / 'test_results_normed.csv'}")
    print(f"{'='*82}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Batch evaluator for Type-B normalised-embedding experiments'
    )
    parser.add_argument(
        '--runs', nargs='+', default=list(EXPERIMENTS_NORMED),
        metavar='RUN_ID',
        help='Run IDs to evaluate (default: all). E.g. --runs B0n E2an E2bn',
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
    unknown = [r for r in args.runs if r not in EXPERIMENTS_NORMED]
    if unknown:
        parser.error(f'Unknown run IDs: {unknown}. Available: {list(EXPERIMENTS_NORMED)}')

    print(f'\nType-B Normalised Evaluation')
    print(f'Runs      : {args.runs}')
    print(f'Device    : {args.device or "auto"}')
    print(f'Checkpoints: {CHECKPOINTS_B_NORMED}')
    print(f'Results   : {PREDICTIONS_B_NORMED}')

    # ── Per-experiment evaluation ──────────────────────────────────────────────
    completed: dict[str, dict] = {}
    skipped:   list[str]       = []

    for run_id in args.runs:
        cfg     = EXPERIMENTS_NORMED[run_id]
        metrics = _run_one(run_id, cfg, device=args.device)
        if metrics is not None:
            completed[run_id] = metrics
        else:
            skipped.append(run_id)

    # ── Comparison table ───────────────────────────────────────────────────────
    _print_comparison_table(completed)

    # ── Cross-run graphs ───────────────────────────────────────────────────────
    if not args.no_cross_graphs and completed:
        print(f'{"="*60}')
        print('  Generating cross-run comparison graphs…')
        print(f'{"="*60}')
        results_csv = PREDICTIONS_B_NORMED / 'test_results_normed.csv'
        if results_csv.exists():
            FIGURES_EVAL_NORM_B.mkdir(parents=True, exist_ok=True)
            plot_cross_run_graphs(
                test_results_csv=results_csv,
                predictions_dir=PREDICTIONS_B_NORMED,
                figures_dir=FIGURES_EVAL_NORM_B,
            )
        else:
            print('  [skip] test_results_normed.csv not found — run evaluations first.')

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f'{"="*60}')
    print(f'  Completed : {list(completed)}')
    if skipped:
        print(f'  Skipped   : {skipped}  (checkpoint not found)')
    print(f'  Results   → {PREDICTIONS_B_NORMED / "test_results_normed.csv"}')
    print(f'  Figures   → {FIGURES_EVAL_NORM_B}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
