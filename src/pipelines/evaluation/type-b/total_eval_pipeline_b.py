
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT     = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
_EVAL_DIR = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_evals_stage1_b    as _s1        # noqa: E402
import run_evals_stage2_b    as _s2        # noqa: E402
import plot_eval_aggregate_b as _aggregate  # noqa: E402
import final_analysis        as _final      # noqa: E402

from src.config.paths import (  # noqa: E402
    PREDICTIONS_B,
    PREDICTIONS_B_S2,
    FIGURES_EVAL_B,
    FIGURES_EVAL_B_S2_NON_NORMED,
)


def _run_stage1(runs: list[str], device: str | None, no_cross_graphs: bool) -> dict[str, dict]:
    from eval_metrics_b import plot_cross_run_graphs

    completed: dict[str, dict] = {}
    skipped:   list[str]       = []

    for run_id in runs:
        cfg     = _s1.EXPERIMENTS[run_id]
        metrics = _s1._run_one(run_id, cfg, device=device)
        if metrics is not None:
            completed[run_id] = metrics
        else:
            skipped.append(run_id)

    _s1._print_comparison_table(completed)

    if not no_cross_graphs and completed:
        results_csv = PREDICTIONS_B / 'test_results.csv'
        if results_csv.exists():
            FIGURES_EVAL_B.mkdir(parents=True, exist_ok=True)
            plot_cross_run_graphs(
                test_results_csv=results_csv,
                predictions_dir=PREDICTIONS_B,
                figures_dir=FIGURES_EVAL_B,
            )

    if skipped:
        print(f'  Skipped: {skipped}  (checkpoint not found)')

    return completed


def _run_stage2(runs: list[str], device: str | None, no_cross_graphs: bool) -> dict[str, dict]:
    from eval_metrics_b import plot_cross_run_graphs

    completed: dict[str, dict] = {}
    skipped:   list[str]       = []

    PREDICTIONS_B_S2.mkdir(parents=True, exist_ok=True)

    for run_id in runs:
        cfg     = _s2.EXPERIMENTS_NON_NORMED[run_id]
        metrics = _s2._run_one_non_normed(run_id, cfg, device=device)
        if metrics is not None:
            completed[run_id] = metrics
        else:
            skipped.append(run_id)

    _s2._print_comparison_table(completed, _s2.EXPERIMENTS_NON_NORMED,
                                'Stage-2 Non-Normalised — Architecture Comparison (test set)')

    if not no_cross_graphs and completed:
        results_csv = PREDICTIONS_B_S2 / 'test_results_s2.csv'
        if results_csv.exists():
            FIGURES_EVAL_B_S2_NON_NORMED.mkdir(parents=True, exist_ok=True)
            plot_cross_run_graphs(
                test_results_csv=results_csv,
                predictions_dir=PREDICTIONS_B_S2,
                figures_dir=FIGURES_EVAL_B_S2_NON_NORMED,
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
        description='Full Type-B evaluation pipeline (non-normalised: Stage 1 + Stage 2)'
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
            'S1: B0 E2a … E2k. S2: S2a S2b S2c.'
        ),
    )
    parser.add_argument('--device', default=None,
                        help='Device: cpu | cuda | mps (default: auto-detect)')
    parser.add_argument('--no-cross-graphs', action='store_true',
                        help='Skip per-stage cross-run comparison graphs')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip aggregate plot stage')
    parser.add_argument('--no-final', action='store_true',
                        help='Skip final composite ranking (final_analysis.py)')
    parser.add_argument('--tsne', action='store_true',
                        help='Generate t-SNE corpus visualisation (~5 min)')
    parser.add_argument('--tsne-embeddings', nargs='+',
                        default=['sbert', 'sbert_finetuned', 'tinybert_mean', 'tfidf_lsa'],
                        metavar='EMBEDDING')
    parser.add_argument('--tsne-colour-by', default='all',
                        choices=['n_digits', 'colour', 'size', 'all'])
    args = parser.parse_args()

    # Resolve which runs to execute
    all_s1 = list(_s1.EXPERIMENTS)
    all_s2 = list(_s2.EXPERIMENTS_NON_NORMED)

    if args.runs:
        unknown = [r for r in args.runs
                   if r not in _s1.EXPERIMENTS and r not in _s2.EXPERIMENTS_NON_NORMED]
        if unknown:
            parser.error(f'Unknown run IDs: {unknown}. '
                         f'S1: {all_s1}  S2: {all_s2}')
        s1_runs = [r for r in args.runs if r in _s1.EXPERIMENTS]
        s2_runs = [r for r in args.runs if r in _s2.EXPERIMENTS_NON_NORMED]
    else:
        s1_runs = all_s1 if args.stage in ('s1', 'all') else []
        s2_runs = all_s2 if args.stage in ('s2', 'all') else []

    sep = '=' * 60
    print(f'\nType-B Non-Normalised Pipeline  (--stage {args.stage})')
    print(f'S1 runs : {s1_runs}')
    print(f'S2 runs : {s2_runs}')

    completed_s1: dict[str, dict] = {}
    completed_s2: dict[str, dict] = {}

    if s1_runs:
        print(f'\n{sep}')
        print('  STAGE 1 — Evaluate S1 checkpoints (embedding axis)')
        print(sep)
        completed_s1 = _run_stage1(s1_runs, args.device, args.no_cross_graphs)

        if not args.no_plots and completed_s1:
            print(f'\n{sep}')
            print('  STAGE 1 — Aggregate plots')
            print(sep)
            _run_aggregate_plots(
                completed_s1, FIGURES_EVAL_B,
                args.tsne, args.tsne_embeddings, args.tsne_colour_by,
            )

    if s2_runs:
        print(f'\n{sep}')
        print('  STAGE 2 — Evaluate S2 checkpoints (architecture axis, non-normalised)')
        print(sep)
        completed_s2 = _run_stage2(s2_runs, args.device, args.no_cross_graphs)

        if not args.no_plots and completed_s2:
            print(f'\n{sep}')
            print('  STAGE 2 — Aggregate plots')
            print(sep)
            _run_aggregate_plots(
                completed_s2, FIGURES_EVAL_B_S2_NON_NORMED,
                args.tsne, args.tsne_embeddings, args.tsne_colour_by,
            )

    all_completed = {**completed_s1, **completed_s2}
    if not all_completed:
        print('\n  No experiments completed — stopping pipeline.')
        return

    if not args.no_final:
        print(f'\n{sep}')
        print('  FINAL — Composite ranking (S1 + S2 merged)')
        print(sep)
        _final.main()

    print(f'\n{sep}')
    print('  Pipeline complete.')
    print(f'  S1 completed : {list(completed_s1)}')
    print(f'  S2 completed : {list(completed_s2)}')
    print(sep)


if __name__ == '__main__':
    main()
