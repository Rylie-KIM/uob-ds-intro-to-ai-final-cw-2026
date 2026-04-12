"""
Full evaluation pipeline for Type-B normalised-embedding experiments.

Runs the three stages in order:
  1. run_evals_stage1_normed_b    — evaluate normalised checkpoints, save per-sample
                                     predictions and test_results_normed.csv to
                                     metrics/type-b/prediction-normalised/
  2. plot_eval_aggregate_normed_b — generate aggregate cross-run figures
  3. final_analysis_normed        — compute composite ranking, save
                                     final_ranking_normed.csv and final_per_digit_normed.csv
                                     to metrics/type-b/prediction-normalised/

Output paths
------------
  Predictions  : src/pipelines/results/metrics/type-b/prediction-normalised/
  Test results : src/pipelines/results/metrics/type-b/prediction-normalised/test_results_normed.csv
  Final ranking: src/pipelines/results/metrics/type-b/prediction-normalised/final_ranking_normed.csv
  Per-digit    : src/pipelines/results/metrics/type-b/prediction-normalised/final_per_digit_normed.csv
  Figures      : src/pipelines/results/figures/type-b/evaluation/normalised/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT     = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
_EVAL_DIR = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_evals_stage1_normed_b    as _stage1    # noqa: E402
import plot_eval_aggregate_normed_b as _aggregate  # noqa: E402
import final_analysis_normed        as _final      # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Full Type-B evaluation pipeline (normalised runs)'
    )
    parser.add_argument(
        '--runs', nargs='+', default=list(_stage1.EXPERIMENTS_NORMED),
        metavar='RUN_ID',
        help='Run IDs to evaluate (default: all). E.g. --runs B0n E2an E2bn',
    )
    parser.add_argument(
        '--device', default=None,
        help='Device: cpu | cuda | mps (default: auto-detect)',
    )
    parser.add_argument(
        '--no-cross-graphs', action='store_true',
        help='Skip cross-run comparison graphs from stage 1',
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip aggregate plot stage (plot_eval_aggregate_normed_b)',
    )
    parser.add_argument(
        '--tsne', action='store_true',
        help='Generate t-SNE corpus visualisation during aggregate stage (~5 min)',
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

    # Validate run IDs
    unknown = [r for r in args.runs if r not in _stage1.EXPERIMENTS_NORMED]
    if unknown:
        parser.error(f'Unknown run IDs: {unknown}. Available: {list(_stage1.EXPERIMENTS_NORMED)}')

    sep = '=' * 60

    # ── Stage 1: evaluate normalised checkpoints ───────────────────────────────
    print(f'\n{sep}')
    print('  STAGE 1 — Evaluate normalised checkpoints')
    print(sep)

    completed: dict[str, dict] = {}
    skipped:   list[str]       = []

    for run_id in args.runs:
        cfg     = _stage1.EXPERIMENTS_NORMED[run_id]
        metrics = _stage1._run_one(run_id, cfg, device=args.device)
        if metrics is not None:
            completed[run_id] = metrics
        else:
            skipped.append(run_id)

    _stage1._print_comparison_table(completed)

    if not args.no_cross_graphs and completed:
        from src.config.paths import PREDICTIONS_B_NORMED, FIGURES_EVAL_NORM_B
        from eval_metrics_b import plot_cross_run_graphs
        results_csv = PREDICTIONS_B_NORMED / 'test_results_normed.csv'
        if results_csv.exists():
            FIGURES_EVAL_NORM_B.mkdir(parents=True, exist_ok=True)
            plot_cross_run_graphs(
                test_results_csv=results_csv,
                predictions_dir=PREDICTIONS_B_NORMED,
                figures_dir=FIGURES_EVAL_NORM_B,
            )

    if skipped:
        print(f'  Skipped: {skipped}  (checkpoint not found)')

    if not completed:
        print('\n  No experiments completed — stopping pipeline.')
        return

    # ── Stage 2: aggregate plots ───────────────────────────────────────────────
    if not args.no_plots:
        print(f'\n{sep}')
        print('  STAGE 2 — Aggregate cross-run plots (normalised)')
        print(sep)

        from src.config.paths import PREDICTIONS_B_NORMED, FIGURES_EVAL_NORM_B
        FIGURES_EVAL_NORM_B.mkdir(parents=True, exist_ok=True)

        preds = _aggregate._load_predictions(list(completed))
        if preds:
            _aggregate.plot_rank_cdf(preds, FIGURES_EVAL_NORM_B)
            _aggregate.plot_rank_boxplot(preds, FIGURES_EVAL_NORM_B)
            _aggregate.plot_cosine_sim_kde(preds, FIGURES_EVAL_NORM_B)
            _aggregate.plot_top1_bar(preds, FIGURES_EVAL_NORM_B)
            _aggregate.plot_mrr_bar(preds, FIGURES_EVAL_NORM_B)
            _aggregate.plot_rank_cdf_by_ndigits(preds, FIGURES_EVAL_NORM_B)

        if args.tsne and preds:
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
    else:
        print(f'\n  [skip] Aggregate plots (--no-plots)')

    # ── Stage 3: final ranking ─────────────────────────────────────────────────
    print(f'\n{sep}')
    print('  STAGE 3 — Final composite ranking (normalised)')
    print(sep)
    _final.main()

    print(f'\n{sep}')
    print('  Pipeline complete.')
    print(sep)


if __name__ == '__main__':
    main()
