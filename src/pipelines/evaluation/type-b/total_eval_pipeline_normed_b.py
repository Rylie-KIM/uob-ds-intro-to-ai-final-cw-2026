"""
Full evaluation pipeline for Type-B normalised experiments.

Runs in order:
  Stage 1  (--stage s1 or all)
    1a. run_evals_stage1_normed_b    — evaluate S1 normed checkpoints
                                       → prediction-normalised/test_results_normed.csv
    1b. aggregate plots              — cross-run figures
                                       → figures/type-b/evaluation/normalised/

  Stage 2  (--stage s2 or all)
    2a. run_evals_stage2_b           — evaluate S2 normed checkpoints
                                       → prediction-s2-normalised/test_results_s2_normed.csv
    2b. aggregate plots              — cross-run figures
                                       → figures/type-b/evaluation/s2/normalised/

  Final  (always, unless --no-final)
    3.  final_analysis_normed        — merge S1 + S2 normed results, compute ranking
                                       → prediction-normalised/final_ranking_normed.csv
                                       → prediction-normalised/leaderboard_normed.csv

Output paths
------------
  S1 predictions  : src/pipelines/results/metrics/type-b/prediction-normalised/
  S2 predictions  : src/pipelines/results/metrics/type-b/prediction-s2-normalised/
  S1 figures      : src/pipelines/results/figures/type-b/evaluation/normalised/
  S2 figures      : src/pipelines/results/figures/type-b/evaluation/s2/normalised/
  Final ranking   : src/pipelines/results/metrics/type-b/prediction-normalised/final_ranking_normed.csv

Usage
-----
  # Run everything (default)
  python src/pipelines/evaluation/type-b/total_eval_pipeline_normed_b.py

  # Stage 1 only
  python src/pipelines/evaluation/type-b/total_eval_pipeline_normed_b.py --stage s1

  # Stage 2 only
  python src/pipelines/evaluation/type-b/total_eval_pipeline_normed_b.py --stage s2

  # Specific runs (mixes stages freely)
  python src/pipelines/evaluation/type-b/total_eval_pipeline_normed_b.py --runs E2en S2an S2ad
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT     = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
_EVAL_DIR = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_evals_stage1_normed_b as _s1        # noqa: E402
import run_evals_stage2_b        as _s2        # noqa: E402
import plot_eval_aggregate_b     as _aggregate  # noqa: E402
import final_analysis_normed     as _final      # noqa: E402

from src.config.paths import (  # noqa: E402
    PREDICTIONS_B_NORMED,
    PREDICTIONS_B_S2_NORMED,
    FIGURES_EVAL_NORM_B,
    FIGURES_EVAL_B_S2_NORMED,
)


def _run_stage1(runs: list[str], device: str | None, no_cross_graphs: bool) -> dict[str, dict]:
    from eval_metrics_b import plot_cross_run_graphs

    completed: dict[str, dict] = {}
    skipped:   list[str]       = []

    for run_id in runs:
        cfg     = _s1.EXPERIMENTS_NORMED[run_id]
        metrics = _s1._run_one(run_id, cfg, device=device)
        if metrics is not None:
            completed[run_id] = metrics
        else:
            skipped.append(run_id)

    _s1._print_comparison_table(completed)

    if not no_cross_graphs and completed:
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

    return completed


def _run_stage2(runs: list[str], device: str | None, no_cross_graphs: bool) -> dict[str, dict]:
    from eval_metrics_b import plot_cross_run_graphs

    completed: dict[str, dict] = {}
    skipped:   list[str]       = []

    PREDICTIONS_B_S2_NORMED.mkdir(parents=True, exist_ok=True)

    for run_id in runs:
        cfg     = _s2.EXPERIMENTS_NORMED[run_id]
        metrics = _s2._run_one_normed(run_id, cfg, device=device)
        if metrics is not None:
            completed[run_id] = metrics
        else:
            skipped.append(run_id)

    _s2._print_comparison_table(completed, _s2.EXPERIMENTS_NORMED,
                                'Stage-2 Normalised — Architecture Comparison (test set)')

    if not no_cross_graphs and completed:
        results_csv = PREDICTIONS_B_S2_NORMED / 'test_results_s2_normed.csv'
        if results_csv.exists():
            FIGURES_EVAL_B_S2_NORMED.mkdir(parents=True, exist_ok=True)
            plot_cross_run_graphs(
                test_results_csv=results_csv,
                predictions_dir=PREDICTIONS_B_S2_NORMED,
                figures_dir=FIGURES_EVAL_B_S2_NORMED,
            )

    if skipped:
        print(f'  Skipped: {skipped}  (checkpoint not found)')

    return completed


def _run_aggregate_plots(
    completed: dict[str, dict],
    figures_dir: Path,
    tsne: bool,
    tsne_embeddings: list[str],
    tsne_colour_by: str,
) -> None:
    preds = _aggregate._load_predictions(list(completed))
    if not preds:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    _aggregate.plot_rank_cdf(preds, figures_dir)
    _aggregate.plot_rank_boxplot(preds, figures_dir)
    _aggregate.plot_cosine_sim_kde(preds, figures_dir)
    _aggregate.plot_top1_bar(preds, figures_dir)
    _aggregate.plot_mrr_bar(preds, figures_dir)
    _aggregate.plot_rank_cdf_by_ndigits(preds, figures_dir)

    if tsne:
        from eval_metrics_b import plot_tsne_corpus
        colour_by_list = (
            ['n_digits', 'colour', 'size'] if tsne_colour_by == 'all' else [tsne_colour_by]
        )
        print(f'\n── t-SNE corpus visualisation ──')
        for colour_by in colour_by_list:
            plot_tsne_corpus(
                embedding_names=tsne_embeddings,
                colour_by=colour_by,
                figures_dir=figures_dir,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Full Type-B evaluation pipeline (normalised: Stage 1 + Stage 2)'
    )
    parser.add_argument(
        '--stage',
        choices=['s1', 's2', 'all'],
        default='all',
        help='Which stage to evaluate: s1 | s2 | all (default: all)',
    )
    parser.add_argument(
        '--runs', nargs='+', default=None,
        metavar='RUN_ID',
        help=(
            'Explicit run IDs (overrides --stage). '
            'S1: B0n E2an … E2kn E2ln E2mn. '
            'S2: S2an S2bn S2cn S2ad S2bd S2cd S2dn S2en.'
        ),
    )
    parser.add_argument('--device', default=None,
                        help='Device: cpu | cuda | mps (default: auto-detect)')
    parser.add_argument('--no-cross-graphs', action='store_true',
                        help='Skip per-stage cross-run comparison graphs')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip aggregate plot stage')
    parser.add_argument('--no-final', action='store_true',
                        help='Skip final composite ranking (final_analysis_normed.py)')
    parser.add_argument('--tsne', action='store_true',
                        help='Generate t-SNE corpus visualisation (~5 min)')
    parser.add_argument('--tsne-embeddings', nargs='+',
                        default=['sbert', 'sbert_finetuned', 'tinybert_mean', 'tfidf_lsa'],
                        metavar='EMBEDDING')
    parser.add_argument('--tsne-colour-by', default='all',
                        choices=['n_digits', 'colour', 'size', 'all'])
    args = parser.parse_args()

    # Resolve which runs to execute
    all_s1 = list(_s1.EXPERIMENTS_NORMED)
    all_s2 = list(_s2.EXPERIMENTS_NORMED)

    if args.runs:
        unknown = [r for r in args.runs
                   if r not in _s1.EXPERIMENTS_NORMED and r not in _s2.EXPERIMENTS_NORMED]
        if unknown:
            parser.error(f'Unknown run IDs: {unknown}. '
                         f'S1: {all_s1}  S2: {all_s2}')
        s1_runs = [r for r in args.runs if r in _s1.EXPERIMENTS_NORMED]
        s2_runs = [r for r in args.runs if r in _s2.EXPERIMENTS_NORMED]
    else:
        s1_runs = all_s1 if args.stage in ('s1', 'all') else []
        s2_runs = all_s2 if args.stage in ('s2', 'all') else []

    sep = '=' * 60
    print(f'\nType-B Normalised Pipeline  (--stage {args.stage})')
    print(f'S1 runs : {s1_runs}')
    print(f'S2 runs : {s2_runs}')

    completed_s1: dict[str, dict] = {}
    completed_s2: dict[str, dict] = {}

    # ── Stage 1 ────────────────────────────────────────────────────────────────
    if s1_runs:
        print(f'\n{sep}')
        print('  STAGE 1 — Evaluate S1 normed checkpoints (embedding axis)')
        print(sep)
        completed_s1 = _run_stage1(s1_runs, args.device, args.no_cross_graphs)

        if not args.no_plots and completed_s1:
            print(f'\n{sep}')
            print('  STAGE 1 — Aggregate plots (normalised)')
            print(sep)
            _run_aggregate_plots(
                completed_s1, FIGURES_EVAL_NORM_B,
                args.tsne, args.tsne_embeddings, args.tsne_colour_by,
            )

    # ── Stage 2 ────────────────────────────────────────────────────────────────
    if s2_runs:
        print(f'\n{sep}')
        print('  STAGE 2 — Evaluate S2 normed checkpoints (architecture axis)')
        print(sep)
        completed_s2 = _run_stage2(s2_runs, args.device, args.no_cross_graphs)

        if not args.no_plots and completed_s2:
            print(f'\n{sep}')
            print('  STAGE 2 — Aggregate plots (normalised)')
            print(sep)
            _run_aggregate_plots(
                completed_s2, FIGURES_EVAL_B_S2_NORMED,
                args.tsne, args.tsne_embeddings, args.tsne_colour_by,
            )

    all_completed = {**completed_s1, **completed_s2}
    if not all_completed:
        print('\n  No experiments completed — stopping pipeline.')
        return

    # ── Final ranking ──────────────────────────────────────────────────────────
    if not args.no_final:
        print(f'\n{sep}')
        print('  FINAL — Composite ranking (S1 + S2 normed merged)')
        print(sep)
        _final.main()

    print(f'\n{sep}')
    print('  Pipeline complete.')
    print(f'  S1 completed : {list(completed_s1)}')
    print(f'  S2 completed : {list(completed_s2)}')
    print(sep)


if __name__ == '__main__':
    main()
