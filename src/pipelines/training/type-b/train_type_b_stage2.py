from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util as _ilu


def _load_hyphen_module(name: str, rel_path: str):
    _spec = _ilu.spec_from_file_location(name, Path(__file__).resolve().parent / rel_path)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


_shared = _load_hyphen_module('_shared_typeb',  'shared.py')
_cnn1   = _load_hyphen_module('_cnn1layer',     '../../../models/type-b/cnn_1layer.py')
_cnn3   = _load_hyphen_module('_cnn3layer',     '../../../models/type-b/cnn_3layer.py')
_resnet = _load_hyphen_module('_resnet18pt',    '../../../models/type-b/resnet18_pt.py')

train_one_epoch  = _shared.train_one_epoch
run_validation   = _shared.run_validation
run_val_retrieval = _shared.run_val_retrieval
set_seed         = _shared.set_seed
CombinedLoss     = _shared.CombinedLoss

CNN1Layer          = _cnn1.CNN1Layer
CNN3Layer          = _cnn3.CNN3Layer
ResNet18Pretrained = _resnet.ResNet18Pretrained

from src.config.paths import EMBED_RESULTS_B, CHECKPOINTS_B, METRICS_B
from src.pipelines.data_loaders.type_b_loader import make_splits


# ── Constants ────────────────────────────────────────────────────────────────

EMBEDDING      = 'tinybert_mean'
EMBEDDING_DIM  = 312

MODEL_CONFIGS: dict[str, callable] = {
    'cnn_1layer':  lambda dim: CNN1Layer(embedding_dim=dim),
    'cnn_3layer':  lambda dim: CNN3Layer(embedding_dim=dim),
    'resnet18_pt': lambda dim: ResNet18Pretrained(embedding_dim=dim),
}

# Per-model epoch budget (resnet18 converges faster due to pretrained backbone)
MODEL_EPOCHS: dict[str, int] = {
    'cnn_1layer':  30,
    'cnn_3layer':  30,
    'resnet18_pt': 20,
}

LOSS_OPTIONS = ('mse', 'combined')

BATCH_SIZE        = 64
LR                = 1e-4
WEIGHT_DECAY      = 1e-4
SEED              = 42
EARLY_STOP_PAT    = 7



# loss class 
class _MSELoss(nn.MSELoss):
    """Thin wrapper so __class__.__name__ reads 'MSELoss' in logs."""
    pass


def _build_criterion(loss_key: str) -> nn.Module:
    """
    Explicitly construct the requested loss, ignoring the default model/embedding
    heuristics used in Stage 1. This ensures Stage 2 is a controlled comparison.
    """
    if loss_key == 'mse':
        return _MSELoss()
    if loss_key == 'combined':
        return CombinedLoss(alpha=0.5)
    raise ValueError(f'Unknown loss_key: {loss_key!r}. Choose from {LOSS_OPTIONS}.')


# train 
def run_experiment_stage2(
    model_name:     str,
    loss_key:       str,
    epochs:         int | None = None,
    device:         str        = 'cpu',
    run_tag:        str        = '',
) -> None:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f'Unknown model: {model_name!r}')
    if loss_key not in LOSS_OPTIONS:
        raise ValueError(f'Unknown loss: {loss_key!r}')

    set_seed(SEED)

    n_epochs = epochs if epochs is not None else MODEL_EPOCHS[model_name]
    run_name = f'b_s2_{model_name}_{loss_key}_{EMBEDDING}' + (f'_{run_tag}' if run_tag else '')

    print(f"\n{'='*64}")
    print(f"  Stage 2 Run : {run_name}")
    print(f"  model       : {model_name}")
    print(f"  loss        : {loss_key}")
    print(f"  embedding   : {EMBEDDING}  (dim={EMBEDDING_DIM})")
    print(f"  epochs      : {n_epochs}   device: {device}")
    print(f"{'='*64}")

    cache_path = EMBED_RESULTS_B / f'{EMBEDDING}_embedding_result_typeb.pt'
    if not cache_path.exists():
        print(f'[skip] Embedding file not found: {cache_path}')
        print(f'  Place tinybert_mean_embedding_result_typeb.pt in {EMBED_RESULTS_B}')
        return

    train_set, val_set, _ = make_splits(
        embedding_cache=cache_path,
        device=device,
        seed=SEED,
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    full_dataset   = train_set.dataset
    all_embeddings = torch.stack([rec[2] for rec in full_dataset.records])
    all_sentences  = [rec[1] for rec in full_dataset.records]

    model     = MODEL_CONFIGS[model_name](EMBEDDING_DIM).to(device)
    criterion = _build_criterion(loss_key)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  trainable params : {total_params:,}')
    print(f'  loss function    : {criterion.__class__.__name__}')
    print(f'  optimizer lr     : {LR}   weight_decay: {WEIGHT_DECAY}')
    print(f'  train={len(train_set)}  val={len(val_set)}')

    CHECKPOINTS_B.mkdir(parents=True, exist_ok=True)
    METRICS_B.mkdir(parents=True, exist_ok=True)

    ckpt_path          = CHECKPOINTS_B / f'{run_name}_best.pt'
    epoch_log_path     = METRICS_B     / f'{run_name}_training_log.csv'
    best_val_loss      = float('inf')
    best_epoch         = 0
    early_stop_counter = 0
    epoch_log: list[dict] = []
    train_start = time.time()

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = run_validation(model, val_loader, criterion, device)
        val_ret    = run_val_retrieval(model, val_loader, all_embeddings, all_sentences, device)
        scheduler.step(val_loss)
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        epoch_log.append({
            'epoch':           epoch,
            'model':           model_name,
            'loss_fn':         loss_key,
            'train_loss':      round(train_loss, 6),
            'val_loss':        round(val_loss, 6),
            'val_top1':        val_ret.get('val_top1',        float('nan')),
            'val_top5':        val_ret.get('val_top5',        float('nan')),
            'val_mrr':         val_ret.get('val_mrr',         float('nan')),
            'val_mean_cosine': val_ret.get('val_mean_cosine', float('nan')),
            'val_mean_rank':   val_ret.get('val_mean_rank',   float('nan')),
            'val_median_rank': val_ret.get('val_median_rank', float('nan')),
            'lr':              current_lr,
            'epoch_time_s':    round(elapsed, 2),
        })
        pd.DataFrame(epoch_log).to_csv(epoch_log_path, index=False)

        print(f'  Epoch {epoch:03d}/{n_epochs}'
              f'  train={train_loss:.6f}  val={val_loss:.6f}'
              f'  top1={val_ret.get("val_top1", float("nan")):.4f}'
              f'  mrr={val_ret.get("val_mrr", float("nan")):.4f}'
              f'  lr={current_lr:.2e}  {elapsed:.1f}s', end='')

        if val_loss < best_val_loss - 1e-5:
            best_val_loss      = val_loss
            best_epoch         = epoch
            early_stop_counter = 0
            torch.save({
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss':       val_loss,
                'embedding_dim':  EMBEDDING_DIM,
                'embedding_name': EMBEDDING,
                'model_name':     model_name,
                'loss_fn':        loss_key,
                'stage':          2,
                'dataset':        'b',
                'train_size':     len(train_set),
                'val_size':       len(val_set),
                'seed':           SEED,
            }, ckpt_path)
            print('  ✓ best', end='')
        else:
            early_stop_counter += 1
            print(f'  [patience {early_stop_counter}/{EARLY_STOP_PAT}]', end='')
        print()

        if early_stop_counter >= EARLY_STOP_PAT:
            print(f'  Early stopping at epoch {epoch}')
            break

    total_time = time.time() - train_start
    print(f'\n  Training complete: best_epoch={best_epoch}'
          f'  best_val_loss={best_val_loss:.6f}  total_time={total_time:.1f}s')
    print(f'  Log        → {epoch_log_path.name}')
    print(f'  Checkpoint → {ckpt_path.name}')


# ── CLI ──────────────────────────────────────────────────────────────────────

def _default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main() -> None:
    parser = argparse.ArgumentParser(description='Type-B Stage 2: loss function comparison')
    parser.add_argument('--model',  choices=list(MODEL_CONFIGS), default=None,
                        help='Single model to run (default: all)')
    parser.add_argument('--loss',   choices=list(LOSS_OPTIONS),  default=None,
                        help='Single loss to run (default: both)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count')
    parser.add_argument('--device', default=_default_device())
    args = parser.parse_args()

    models  = [args.model] if args.model else list(MODEL_CONFIGS)
    losses  = [args.loss]  if args.loss  else list(LOSS_OPTIONS)
    total   = len(models) * len(losses)

    print(f'Stage 2 — {total} run(s): {models} × {losses}')
    print(f'Embedding : {EMBEDDING}  Device: {args.device}')

    for model_name in models:
        for loss_key in losses:
            run_experiment_stage2(
                model_name=model_name,
                loss_key=loss_key,
                epochs=args.epochs,
                device=args.device,
            )

    print('\nAll Stage 2 runs complete.')


if __name__ == '__main__':
    main()
