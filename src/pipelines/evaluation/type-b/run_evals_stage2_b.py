# Output paths
#   Non-normalised predictions : src/pipelines/results/metrics/type-b/prediction-s2/
#   Normalised predictions     : src/pipelines/results/metrics/type-b/prediction-s2-normalised/
#   Non-normalised figures     : src/pipelines/results/figures/type-b/evaluation/s2/non-normalised/
#   Normalised figures         : src/pipelines/results/figures/type-b/evaluation/s2/normalised/

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

from eval_metrics_b import run_evaluation, run_evaluation_normed, plot_cross_run_graphs  # noqa: E402
from src.config.paths import (  # noqa: E402
    CHECKPOINTS_B_S2_NON_NORMED,
    CHECKPOINTS_B_S2_NORMED,
    METRICS_B_S2_NON_NORMED,
    METRICS_B_S2_NORMED,
    PREDICTIONS_B_S2,
    PREDICTIONS_B_S2_NORMED,
    FIGURES_EVAL_B_S2_NON_NORMED,
    FIGURES_EVAL_B_S2_NORMED,
)


# Stage 2 — architecture axis (fixed: tinybert_mean, varying: CNN architecture)
# Checkpoint naming: b_s2_{model}_{loss}_{embedding}[_normed]_{tag}_{ts}_best.pt

EXPERIMENTS_NON_NORMED: dict[str, dict] = {
    'S2a': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'MSE',
        'glob_pattern':   'b_s2_cnn_1layer_mse_tinybert_mean_*_best.pt',
        'description':    'Stage-2 cnn_1layer + TinyBERT-mean (312-dim), MSE',
    },
    'S2b': {
        'model_name':     'cnn_3layer',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'MSE',
        'glob_pattern':   'b_s2_cnn_3layer_mse_tinybert_mean_*_best.pt',
        'description':    'Stage-2 cnn_3layer + TinyBERT-mean (312-dim), MSE',
    },
    'S2c': {
        'model_name':     'resnet18_pt',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'MSE',
        'glob_pattern':   'b_s2_resnet18_pt_mse_tinybert_mean_*_best.pt',
        'description':    'Stage-2 ResNet18-pretrained + TinyBERT-mean (312-dim), MSE',
    },
}

EXPERIMENTS_NORMED: dict[str, dict] = {
    'S2an': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'MSE',
        'glob_pattern':   'b_s2_cnn_1layer_mse_tinybert_mean_normed_*_best.pt',
        'description':    'Stage-2 cnn_1layer + TinyBERT-mean normed (312-dim), MSE',
    },
    'S2bn': {
        'model_name':     'cnn_3layer',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'MSE',
        'glob_pattern':   'b_s2_cnn_3layer_mse_tinybert_mean_normed_*_best.pt',
        'description':    'Stage-2 cnn_3layer + TinyBERT-mean normed (312-dim), MSE',
    },
    'S2cn': {
        'model_name':     'resnet18_pt',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'MSE',
        'glob_pattern':   'b_s2_resnet18_pt_mse_tinybert_mean_normed_*_best.pt',
        'description':    'Stage-2 ResNet18-pretrained + TinyBERT-mean normed (312-dim), MSE',
    },
    'S2ad': {
        'model_name':     'cnn_1layer',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'Combined',
        'glob_pattern':   'b_s2_cnn_1layer_combined_tinybert_mean_normed_*_best.pt',
        'description':    'Stage-2 cnn_1layer + TinyBERT-mean normed (312-dim), Combined',
    },
    'S2bd': {
        'model_name':     'cnn_3layer',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'Combined',
        'glob_pattern':   'b_s2_cnn_3layer_combined_tinybert_mean_normed_*_best.pt',
        'description':    'Stage-2 cnn_3layer + TinyBERT-mean normed (312-dim), Combined',
    },
    'S2cd': {
        'model_name':     'resnet18_pt',
        'embedding_name': 'tinybert_mean',
        'loss_fn':        'Combined',
        'glob_pattern':   'b_s2_resnet18_pt_combined_tinybert_mean_normed_*_best.pt',
        'description':    'Stage-2 ResNet18-pretrained + TinyBERT-mean normed (312-dim), Combined',
    },
    'S2dn': {
        'model_name':     'cnn_3layer',
        'embedding_name': 'sbert',
        'loss_fn':        'Cosine',
        'glob_pattern':   'b_s2_cnn_3layer_cosine_sbert_normed_*_best.pt',
        'description':    'Stage-2 cnn_3layer + SBERT normed (384-dim), Cosine',
    },
    'S2en': {
        'model_name':     'resnet18_pt',
        'embedding_name': 'sbert',
        'loss_fn':        'Cosine',
        'glob_pattern':   'b_s2_resnet18_pt_cosine_sbert_normed_*_best.pt',
        'description':    'Stage-2 ResNet18-pretrained + SBERT normed (384-dim), Cosine',
    },
}


def _resolve_checkpoint(glob_pattern: str, ckpt_dir: Path) -> Path | None:
    candidates = sorted(
        ckpt_dir.glob(glob_pattern),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _run_one_non_normed(run_id: str, cfg: dict, device: str | None) -> dict | None:
    ckpt_path = _resolve_checkpoint(cfg['glob_pattern'], CHECKPOINTS_B_S2_NON_NORMED)
    if ckpt_path is None:
        print(f'\n  [skip] {run_id} — no checkpoint found in {CHECKPOINTS_B_S2_NON_NORMED}'
              f'\n         pattern: {cfg["glob_pattern"]}')
        return None

    try:
        return run_evaluation(
            run_id=run_id,
            model_name=cfg['model_name'],
            embedding_name=cfg['embedding_name'],
            loss_fn=cfg['loss_fn'],
            device=device,
            ckpt_path=ckpt_path,
            pred_dir=PREDICTIONS_B_S2,
            results_path=PREDICTIONS_B_S2 / 'test_results_s2.csv',
            log_dir=METRICS_B_S2_NON_NORMED,
        )
    except FileNotFoundError as exc:
        print(f'\n  [skip] {run_id} — {exc}')
        return None
    except Exception:
        print(f'\n  [error] {run_id}:')
        traceback.print_exc()
        return None


def _run_one_normed(run_id: str, cfg: dict, device: str | None) -> dict | None:
    ckpt_path = _resolve_checkpoint(cfg['glob_pattern'], CHECKPOINTS_B_S2_NORMED)
    if ckpt_path is None:
        print(f'\n  [skip] {run_id} — no checkpoint found in {CHECKPOINTS_B_S2_NORMED}'
              f'\n         pattern: {cfg["glob_pattern"]}')
        return None

    try:
        return run_evaluation_normed(
            run_id=run_id,
            model_name=cfg['model_name'],
            embedding_name=cfg['embedding_name'],
            loss_fn=cfg['loss_fn'],
            device=device,
            ckpt_path=ckpt_path,
            pred_dir=PREDICTIONS_B_S2_NORMED,
            results_path=PREDICTIONS_B_S2_NORMED / 'test_results_s2_normed.csv',
            log_dir=METRICS_B_S2_NORMED,
        )
    except FileNotFoundError as exc:
        print(f'\n  [skip] {run_id} — {exc}')
        return None
    except Exception:
        print(f'\n  [error] {run_id}:')
        traceback.print_exc()
        return None


def _print_comparison_table(
    results:     dict[str, dict],
    experiments: dict[str, dict],
    title:       str,
) -> None:
    if not results:
        print(f'\n  No results to compare for {title}.')
        return

    rows = sorted(
        results.items(),
        key=lambda kv: (kv[1].get('test_top1', -1), kv[1].get('test_mrr', -1)),
        reverse=True,
    )

    header = (
        f"\n{'='*84}\n"
        f"  {title}\n"
        f"{'='*84}\n"
        f"  {'Run':<6} {'Model':<14} {'Loss':<10} {'top-1':>6} {'top-5':>6} "
        f"{'MRR':>7} {'mean_rank':>10} {'cosine':>8}\n"
        f"  {'-'*78}"
    )
    print(header)

    for run_id, m in rows:
        cfg = experiments[run_id]
        print(
            f"  {run_id:<6} {cfg['model_name']:<14} {cfg['loss_fn']:<10} "
            f"{m.get('test_top1', float('nan')):>6.4f} "
            f"{m.get('test_top5', float('nan')):>6.4f} "
            f"{m.get('test_mrr',  float('nan')):>7.4f} "
            f"{m.get('test_mean_rank', float('nan')):>10.1f} "
            f"{m.get('test_mean_cosine', float('nan')):>8.4f}"
        )

    print(f"  {'='*78}")
    print(f"  Sorted by: test_top1 ↓, test_mrr ↓")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Stage-2 batch evaluator for Type-B architecture-axis experiments'
    )
    parser.add_argument(
        '--variant',
        choices=['non-normalised', 'normalised', 'all'],
        default='all',
        help='Which checkpoint variant to evaluate (default: all)',
    )
    parser.add_argument(
        '--runs', nargs='+', default=None,
        metavar='RUN_ID',
        help=(
            'Specific run IDs to evaluate. '
            'Non-normed: S2a S2b S2c. '
            'Normed: S2an S2bn S2cn S2ad S2bd S2cd S2dn S2en.'
        ),
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

    run_non_normed = args.variant in ('non-normalised', 'all')
    run_normed     = args.variant in ('normalised', 'all')

    # Validate run IDs
    all_valid = set(EXPERIMENTS_NON_NORMED) | set(EXPERIMENTS_NORMED)
    if args.runs:
        unknown = [r for r in args.runs if r not in all_valid]
        if unknown:
            parser.error(f'Unknown run IDs: {unknown}. Available: {sorted(all_valid)}')

    print(f'\nType-B Stage-2 Evaluation — Architecture Axis')
    print(f'Variant : {args.variant}')
    print(f'Device  : {args.device or "auto"}')

    completed_nn:  dict[str, dict] = {}
    skipped_nn:    list[str]       = []
    completed_n:   dict[str, dict] = {}
    skipped_n:     list[str]       = []

    # ── Non-normalised ─────────────────────────────────────────────────────────
    if run_non_normed:
        nn_runs = (
            [r for r in args.runs if r in EXPERIMENTS_NON_NORMED]
            if args.runs else list(EXPERIMENTS_NON_NORMED)
        )
        print(f'\nNon-normalised runs : {nn_runs}')
        PREDICTIONS_B_S2.mkdir(parents=True, exist_ok=True)

        for run_id in nn_runs:
            metrics = _run_one_non_normed(run_id, EXPERIMENTS_NON_NORMED[run_id], args.device)
            if metrics is not None:
                completed_nn[run_id] = metrics
            else:
                skipped_nn.append(run_id)

        _print_comparison_table(
            completed_nn, EXPERIMENTS_NON_NORMED,
            'Stage-2 Non-Normalised — Architecture Comparison (test set)',
        )
        print(f'  Results → {PREDICTIONS_B_S2 / "test_results_s2.csv"}')

        if not args.no_cross_graphs and completed_nn:
            test_csv = PREDICTIONS_B_S2 / 'test_results_s2.csv'
            if test_csv.exists():
                FIGURES_EVAL_B_S2_NON_NORMED.mkdir(parents=True, exist_ok=True)
                plot_cross_run_graphs(
                    test_results_csv=test_csv,
                    predictions_dir=PREDICTIONS_B_S2,
                    figures_dir=FIGURES_EVAL_B_S2_NON_NORMED,
                )

    if run_normed:
        n_runs = (
            [r for r in args.runs if r in EXPERIMENTS_NORMED]
            if args.runs else list(EXPERIMENTS_NORMED)
        )
        print(f'\nNormalised runs : {n_runs}')
        PREDICTIONS_B_S2_NORMED.mkdir(parents=True, exist_ok=True)

        for run_id in n_runs:
            metrics = _run_one_normed(run_id, EXPERIMENTS_NORMED[run_id], args.device)
            if metrics is not None:
                completed_n[run_id] = metrics
            else:
                skipped_n.append(run_id)

        _print_comparison_table(
            completed_n, EXPERIMENTS_NORMED,
            'Stage-2 Normalised — Architecture Comparison (test set)',
        )
        print(f'  Results → {PREDICTIONS_B_S2_NORMED / "test_results_s2_normed.csv"}')

        if not args.no_cross_graphs and completed_n:
            test_csv = PREDICTIONS_B_S2_NORMED / 'test_results_s2_normed.csv'
            if test_csv.exists():
                FIGURES_EVAL_B_S2_NORMED.mkdir(parents=True, exist_ok=True)
                plot_cross_run_graphs(
                    test_results_csv=test_csv,
                    predictions_dir=PREDICTIONS_B_S2_NORMED,
                    figures_dir=FIGURES_EVAL_B_S2_NORMED,
                )

    print(f'\n{"="*60}')
    if run_non_normed:
        print(f'  Non-normed completed : {list(completed_nn)}')
        if skipped_nn:
            print(f'  Non-normed skipped   : {skipped_nn}')
    if run_normed:
        print(f'  Normed completed     : {list(completed_n)}')
        if skipped_n:
            print(f'  Normed skipped       : {skipped_n}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
