"""
src/pipelines/evaluation/type-b/total_eval_pipeline_all_b.py

Master evaluation pipeline — runs ALL Type-B variants in one shot.

Execution order
---------------
  1. S1 non-normed eval  (run_evals_stage1_b)
  2. S1 normed eval      (run_evals_stage1_normed_b)
  3. S2 non-normed eval  (run_evals_stage2_b)
  4. S2 normed eval      (run_evals_stage2_b)
  5. Aggregate plots     (per variant → their own figures dirs)
  6. final_analysis          — S1+S2 non-normed ranking + LLM
  7. final_analysis_normed   — S1+S2 normed ranking
  8. plot_eval_comparison_b  — all cross-variant comparison plots
  9. final_analysis_combined — unified metric table + combined figures

Skip flags
----------
  --skip-eval     skip steps 1-4  (use existing prediction CSVs)
  --skip-plots    skip step 5     (aggregate per-variant figures)
  --skip-final    skip steps 6-7  (per-variant final rankings)
  --skip-combined skip step 9     (combined analysis)
  --stage s1|s2|all   limit eval to one stage (default: all)
  --variant non-normalised|normalised|all  limit normed/non-normed (default: all)

Usage
-----
  # Full run (everything)
  python src/pipelines/evaluation/type-b/total_eval_pipeline_all_b.py

  # Only evaluate + rank (no combined figures)
  python src/pipelines/evaluation/type-b/total_eval_pipeline_all_b.py --skip-combined

  # Reuse existing CSVs, regenerate all plots/rankings
  python src/pipelines/evaluation/type-b/total_eval_pipeline_all_b.py --skip-eval

  # Stage 2 only, all variants
  python src/pipelines/evaluation/type-b/total_eval_pipeline_all_b.py --stage s2
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

# ── Import pipeline helpers (not main() — to avoid argparse conflicts) ─────────
import total_eval_pipeline_b        as _pipe_non   # noqa: E402
import total_eval_pipeline_normed_b as _pipe_nor   # noqa: E402

import final_analysis          as _final_non   # noqa: E402
import final_analysis_normed   as _final_nor   # noqa: E402
import final_analysis_combined as _final_comb  # noqa: E402
import plot_eval_comparison_b  as _cmp         # noqa: E402

import pandas as pd  # noqa: E402

from src.config.paths import (  # noqa: E402
    PREDICTIONS_B,
    PREDICTIONS_B_NORMED,
    PREDICTIONS_B_S2,
    PREDICTIONS_B_S2_NORMED,
    METRICS_B,
    FIGURES_EVAL_B,
    FIGURES_EVAL_NORM_B,
    FIGURES_EVAL_B_S2_NON_NORMED,
    FIGURES_EVAL_B_S2_NORMED,
    FIGURES_EVAL_CMP_B,
)

SEP  = '=' * 65
SEP2 = '-' * 65


def _header(title: str) -> None:
    print(f'\n{SEP}')
    print(f'  {title}')
    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Master Type-B evaluation pipeline (all variants)'
    )
    parser.add_argument(
        '--stage', choices=['s1', 's2', 'all'], default='all',
        help='Which experiment stage to evaluate (default: all)',
    )
    parser.add_argument(
        '--variant', choices=['non-normalised', 'normalised', 'all'], default='all',
        help='Which normalisation variant to evaluate (default: all)',
    )
    parser.add_argument('--device', default=None,
                        help='Device: cpu | cuda | mps (default: auto-detect)')
    parser.add_argument('--skip-eval',     action='store_true',
                        help='Skip evaluation steps — reuse existing prediction CSVs')
    parser.add_argument('--skip-plots',    action='store_true',
                        help='Skip per-variant aggregate plot generation')
    parser.add_argument('--skip-final',    action='store_true',
                        help='Skip per-variant final ranking (final_analysis[_normed])')
    parser.add_argument('--skip-combined', action='store_true',
                        help='Skip combined cross-variant analysis')
    parser.add_argument('--no-cross-graphs', action='store_true',
                        help='Skip cross-run comparison graphs during eval')
    parser.add_argument('--tsne', action='store_true',
                        help='Generate t-SNE corpus visualisation (~5 min)')
    parser.add_argument('--tsne-embeddings', nargs='+',
                        default=['sbert', 'sbert_finetuned', 'tinybert_mean', 'tfidf_lsa'],
                        metavar='EMBEDDING')
    parser.add_argument('--tsne-colour-by', default='all',
                        choices=['n_digits', 'colour', 'size', 'all'])
    args = parser.parse_args()

    run_s1  = args.stage   in ('s1',  'all')
    run_s2  = args.stage   in ('s2',  'all')
    run_non = args.variant in ('non-normalised', 'all')
    run_nor = args.variant in ('normalised',     'all')

    print(f'\nType-B Master Pipeline')
    print(f'  stage   : {args.stage}   variant : {args.variant}')
    print(f'  device  : {args.device or "auto"}')
    print(f'  skip-eval={args.skip_eval}  skip-plots={args.skip_plots}  '
          f'skip-final={args.skip_final}  skip-combined={args.skip_combined}')

    tsne_kwargs = dict(
        tsne=args.tsne,
        tsne_embeddings=args.tsne_embeddings,
        tsne_colour_by=args.tsne_colour_by,
    )

    completed: dict[str, dict] = {}   # variant-label → {run_id: metrics}

    # ── 1. Evaluation ──────────────────────────────────────────────────────────
    if not args.skip_eval:

        if run_s1 and run_non:
            _header('STEP 1a — Stage-1 Non-Normalised Eval')
            c = _pipe_non._run_stage1(
                list(_pipe_non._s1.EXPERIMENTS),
                args.device, args.no_cross_graphs,
            )
            completed['s1-non'] = c
            print(f'  Completed: {list(c)}')

        if run_s1 and run_nor:
            _header('STEP 1b — Stage-1 Normalised Eval')
            c = _pipe_nor._run_stage1(
                list(_pipe_nor._s1.EXPERIMENTS_NORMED),
                args.device, args.no_cross_graphs,
            )
            completed['s1-nor'] = c
            print(f'  Completed: {list(c)}')

        if run_s2 and run_non:
            _header('STEP 2a — Stage-2 Non-Normalised Eval')
            c = _pipe_non._run_stage2(
                list(_pipe_non._s2.EXPERIMENTS_NON_NORMED),
                args.device, args.no_cross_graphs,
            )
            completed['s2-non'] = c
            print(f'  Completed: {list(c)}')

        if run_s2 and run_nor:
            _header('STEP 2b — Stage-2 Normalised Eval')
            c = _pipe_nor._run_stage2(
                list(_pipe_nor._s2.EXPERIMENTS_NORMED),
                args.device, args.no_cross_graphs,
            )
            completed['s2-nor'] = c
            print(f'  Completed: {list(c)}')

    else:
        print(f'\n  [skip-eval] Reusing existing prediction CSVs.')

    # ── 2. Aggregate plots ─────────────────────────────────────────────────────
    if not args.skip_plots:

        plot_cases = []
        if run_s1 and run_non:
            plot_cases.append(('s1-non', PREDICTIONS_B,        FIGURES_EVAL_B))
        if run_s1 and run_nor:
            plot_cases.append(('s1-nor', PREDICTIONS_B_NORMED, FIGURES_EVAL_NORM_B))
        if run_s2 and run_non:
            plot_cases.append(('s2-non', PREDICTIONS_B_S2,     FIGURES_EVAL_B_S2_NON_NORMED))
        if run_s2 and run_nor:
            plot_cases.append(('s2-nor', PREDICTIONS_B_S2_NORMED, FIGURES_EVAL_B_S2_NORMED))

        for label, pred_dir, fig_dir in plot_cases:
            _header(f'Aggregate Plots — {label}')
            # Load predictions from the right dir using the completed run IDs
            # (fall back to all known run IDs if --skip-eval was used)
            run_ids = list(completed.get(label, {}).keys())
            if not run_ids:
                # discover from existing prediction CSVs
                run_ids = [p.stem.replace('_test_predictions', '')
                           for p in pred_dir.glob('*_test_predictions.csv')]
            if run_ids:
                _pipe_non._run_aggregate_plots(
                    {r: {} for r in run_ids}, fig_dir, **tsne_kwargs
                )

    else:
        print(f'\n  [skip-plots] Skipping aggregate plot generation.')

    # ── 3. Per-variant final rankings ──────────────────────────────────────────
    if not args.skip_final:

        if run_non:
            _header('STEP 3a — Final Ranking (non-normalised)')
            _final_non.main()

        if run_nor:
            _header('STEP 3b — Final Ranking (normalised)')
            _final_nor.main()

    else:
        print(f'\n  [skip-final] Skipping per-variant final rankings.')

    # ── 4. Cross-variant comparison plots ─────────────────────────────────────
    _header('STEP 4 — Cross-Variant Comparison Plots')
    FIGURES_EVAL_CMP_B.mkdir(parents=True, exist_ok=True)

    comparison_ran = False
    if run_non and run_nor:
        # s1 base vs normed
        try:
            df_base, df_normed = _cmp._load_s1()
            _cmp.plot_cmp_top1(df_base, df_normed)
            _cmp.plot_cmp_mrr(df_base, df_normed)
            _cmp.plot_cmp_mean_rank(df_base, df_normed)
            _cmp.plot_cmp_top1_by_ndigits(df_base, df_normed)
            comparison_ran = True
        except FileNotFoundError as exc:
            print(f'  [skip] S1 base vs normed: {exc}')

        # s2 non vs normed
        try:
            df_non, df_nor = _cmp._load_s2()
            _cmp.plot_s2_cmp_top1(df_non, df_nor)
            _cmp.plot_s2_cmp_mrr(df_non, df_nor)
            _cmp.plot_s2_cmp_mean_rank(df_non, df_nor)
            _cmp.plot_s2_cmp_top1_by_ndigits(df_non, df_nor)
            comparison_ran = True
        except FileNotFoundError as exc:
            print(f'  [skip] S2 non vs normed: {exc}')

    if run_s1 and run_s2 and run_non:
        try:
            df_s1, df_s2 = _cmp._load_s1_vs_s2()
            _cmp.plot_s1_vs_s2_cmp_top1(df_s1, df_s2)
            _cmp.plot_s1_vs_s2_cmp_mrr(df_s1, df_s2)
            _cmp.plot_s1_vs_s2_cmp_mean_rank(df_s1, df_s2)
            comparison_ran = True
        except FileNotFoundError as exc:
            print(f'  [skip] S1 vs S2: {exc}')

    if not comparison_ran:
        print('  [skip] No comparison plots generated '
              '(need both normed and non-normed results).')

    # ── 5. Combined analysis ───────────────────────────────────────────────────
    if not args.skip_combined:
        _header('STEP 5 — Combined Cross-Variant Analysis')
        _final_comb.main()
    else:
        print(f'\n  [skip-combined] Skipping combined analysis.')

    # ── 6. Superman leaderboard (merge non-normed + normed) ───────────────────
    _header('STEP 6 — Superman Leaderboard')
    _build_superman_leaderboard()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('  Master pipeline complete.')
    for label, c in completed.items():
        print(f'  {label:<10}: {list(c)}')
    print(SEP)


_W_MRR         = 0.35
_W_MEDIAN_RANK = 0.25
_W_TOP1        = 0.20
_W_TOP5        = 0.15
_W_COSINE      = 0.05
_CORPUS_SIZE   = 10008
_COLLAPSE_THRESHOLD = 0.95


def _build_superman_leaderboard() -> None:
    non_normed_path = PREDICTIONS_B / 'leaderboard.csv'
    normed_path     = PREDICTIONS_B_NORMED / 'leaderboard_normed.csv'
    out_path        = METRICS_B / 'superman_leaderboard.csv'

    frames = []
    for path, label in [(non_normed_path, 'non-normed'), (normed_path, 'normed')]:
        if not path.exists():
            print(f'  [skip] {label} leaderboard not found: {path}')
            continue
        df = pd.read_csv(path)
        df.insert(0, 'variant', label)
        frames.append(df)

    if not frames:
        print('  [skip] No leaderboard files found — superman_leaderboard.csv not created.')
        return

    combined = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    combined.drop(columns=['rank', 'mrr_norm', 'median_rank_norm', 'top1_norm',
                           'top5_norm', 'cosine_norm', 'composite_score'],
                  errors='ignore', inplace=True)

    # Recompute composite using a single shared CNN-only max across both leaderboards
    is_llm    = combined['run_id'].str.startswith('LLM-')
    collapsed = combined['collapsed'].fillna(False)
    combined['collapsed'] = collapsed

    cnn_mask = ~is_llm

    def _norm(col: str) -> pd.Series:
        mx = combined.loc[cnn_mask, col].max()
        return combined[col] / mx if mx > 0 else combined[col] * 0.0

    combined['mrr_norm']         = _norm('test_mrr')
    combined['median_rank_norm'] = 1.0 - (combined['test_median_rank'] / _CORPUS_SIZE)
    combined['top1_norm']        = _norm('test_top1')
    combined['top5_norm']        = _norm('test_top5')
    combined['cosine_norm']      = combined['test_mean_cosine'] if 'test_mean_cosine' in combined.columns else 0.0

    raw = (
        _W_MRR         * combined['mrr_norm']
        + _W_MEDIAN_RANK * combined['median_rank_norm']
        + _W_TOP1        * combined['top1_norm']
        + _W_TOP5        * combined['top5_norm']
        + _W_COSINE      * combined['cosine_norm']
    )
    combined['composite_score'] = raw.where(~collapsed, other=0.0)

    combined = combined.sort_values('composite_score', ascending=False).reset_index(drop=True)
    combined.insert(0, 'overall_rank', combined.index + 1)

    METRICS_B.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f'  [ok] superman_leaderboard.csv saved → {out_path}')
    print(f'       {len(combined)} rows, composite recomputed on shared CNN-only max')


if __name__ == '__main__':
    main()
