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
    """Load a module from a path that contains hyphens (not valid Python identifiers)."""
    _spec = _ilu.spec_from_file_location(name, Path(__file__).resolve().parent / rel_path)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod

_models  = _load_hyphen_module('_models_typeb',   '../../../models/type-b/__init__.py') if False else None
_shared  = _load_hyphen_module('_shared_typeb',   'shared.py')

train_one_epoch             = _shared.train_one_epoch
train_one_epoch_normalised  = _shared.train_one_epoch_normalised
run_validation              = _shared.run_validation
run_validation_normalised   = _shared.run_validation_normalised
run_val_retrieval           = _shared.run_val_retrieval
set_seed                    = _shared.set_seed
CombinedLoss                = _shared.CombinedLoss

_cnn1      = _load_hyphen_module('_cnn1layer',   '../../../models/type-b/cnn_1layer.py')
_cnn3      = _load_hyphen_module('_cnn3layer',   '../../../models/type-b/cnn_3layer.py')
_resnet    = _load_hyphen_module('_resnet18pt',  '../../../models/type-b/resnet18_pt.py')

CNN1Layer          = _cnn1.CNN1Layer
CNN3Layer          = _cnn3.CNN3Layer
ResNet18Pretrained = _resnet.ResNet18Pretrained

from src.config.paths import EMBED_RESULTS_B, CHECKPOINTS_B, METRICS_B
from src.pipelines.data_loaders.type_b_loader import make_splits, IMAGENET_TRANSFORM


EMBEDDING_CONFIGS: dict[str, dict] = {
    'sbert':               {'dim': 384},
    'sbert_finetuned':     {'dim': 384},
    'bert_mean':           {'dim': 768},
    'bert_pooler':         {'dim': 768},
    'tinybert_mean':       {'dim': 312},
    'tinybert_pooler':     {'dim': 312},
    'word2vec_skipgram':   {'dim': 100},
    'word2vec_pretrained': {'dim': 300},
    'glove':               {'dim': 300},
    'tfidf_lsa':           {'dim': 100},
    'tfidf_w2v':           {'dim': 100},
}

MODEL_CONFIGS: dict[str, callable] = {
    'cnn_1layer':   lambda dim: CNN1Layer(embedding_dim=dim),
    'cnn_3layer':   lambda dim: CNN3Layer(embedding_dim=dim),
    'resnet18_pt':  lambda dim: ResNet18Pretrained(embedding_dim=dim),
}

# ── Default hyperparameters (from src/config/training.json + hyperparams.json) ─
BATCH_SIZE   = 64
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 0   # overridden per-device in run_experiment (0 for MPS, 4 for CUDA)
SEED         = 42

# AlexNet uses a lower LR due to its larger parameter count (~20M vs ~1.2M)
_ALEXNET_LR           = 5e-5
_ALEXNET_WEIGHT_DECAY = 5e-4

# Loss function selection per model (from src/config/loss.json)
# cnn_3layer uses CombinedLoss; simpler/pretrained models use MSELoss
_COMBINED_LOSS_MODELS = {'cnn_3layer'}
# SBERT-family embeddings prefer CosineLoss
_COSINE_LOSS_EMBEDDINGS = {'sbert', 'sbert_finetuned'}


# Models that require ImageNet-normalised input (pretrained on ImageNet weights)
_IMAGENET_NORM_MODELS = {'resnet18_pt'}


def _build_criterion(model_name: str, embedding_name: str) -> nn.Module:
    """Select loss function based on model and embedding combination."""
    if embedding_name in _COSINE_LOSS_EMBEDDINGS:
        # 1 - cosine_similarity, averaged over batch
        class CosineLoss(nn.Module):
            def forward(self, y_hat, y):
                return (1.0 - torch.nn.functional.cosine_similarity(y_hat, y, dim=1)).mean()
        return CosineLoss()
    if model_name in _COMBINED_LOSS_MODELS:
        return CombinedLoss(alpha=0.5)
    return nn.MSELoss()


def run_experiment(
    model_name:     str,
    embedding_name: str,
    epochs:         int  = EPOCHS,
    device:         str  = 'cpu',
    run_tag:        str  = '',
) -> None:
    set_seed(SEED)

    run_name  = f'b_{model_name}_{embedding_name}' + (f'_{run_tag}' if run_tag else '')
    embed_dim = EMBEDDING_CONFIGS[embedding_name]['dim']

    print(f"\n{'='*60}")
    print(f"  Run       : {run_name}")
    print(f"  model     : {model_name}    embedding: {embedding_name}")
    print(f"  embed_dim : {embed_dim}     device: {device}")
    print(f"{'='*60}")


    cache_path = EMBED_RESULTS_B / f'{embedding_name}_embedding_result_typeb.pt'
    if not cache_path.exists():
        print(f'[skip] Embedding file not found: {cache_path}')
        print(f'  Run: python src/embeddings/computed-embeddings/type-b/'
              f'generate_embeddings_type_b.py --embedding {embedding_name}')
        return

    img_transform = IMAGENET_TRANSFORM if model_name in _IMAGENET_NORM_MODELS else None
    train_set, val_set, _ = make_splits(
        embedding_cache=cache_path,
        device=device,
        seed=SEED,
        transform=img_transform,
    )

    # num_workers=0 is faster in Colab: small dataset + limited CPU cores mean
    # worker process spawn/teardown overhead exceeds data loading benefit.
    num_workers = 0
    pin = False  # pin_memory only helps with num_workers > 0
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers)

    # Full corpus for per-epoch retrieval evaluation on the validation set
    full_dataset   = train_set.dataset
    all_embeddings = torch.stack([rec[2] for rec in full_dataset.records])  # (N, dim) on CPU
    all_sentences  = [rec[1] for rec in full_dataset.records]

    model = MODEL_CONFIGS[model_name](embed_dim).to(device)

    lr           = _ALEXNET_LR           if model_name == 'alexnet' else LR
    weight_decay = _ALEXNET_WEIGHT_DECAY if model_name == 'alexnet' else WEIGHT_DECAY

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = _build_criterion(model_name, embedding_name)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  trainable params : {total_params:,}')
    print(f'  loss function    : {criterion.__class__.__name__}')
    print(f'  optimizer lr     : {lr}   weight_decay: {weight_decay}')
    print(f'  train={len(train_set)}  val={len(val_set)}')

    # training loop 
    CHECKPOINTS_B.mkdir(parents=True, exist_ok=True)
    METRICS_B.mkdir(parents=True, exist_ok=True)

    ckpt_path     = CHECKPOINTS_B / f'{run_name}_best.pt'
    best_val_loss = float('inf')
    best_epoch    = 0
    early_stop_counter = 0
    EARLY_STOP_PATIENCE = 7

    epoch_log: list[dict] = []
    epoch_log_path = METRICS_B / f'{run_name}_training_log.csv'
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss    = run_validation(model, val_loader, criterion, device)
        val_ret     = run_val_retrieval(model, val_loader, all_embeddings, all_sentences, device)
        scheduler.step(val_loss)
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        epoch_log.append({
            'epoch':           epoch,
            'loss_fn':         criterion.__class__.__name__,
            'train_loss':      round(train_loss, 6),
            'val_loss':        round(val_loss, 6),
            'val_top1':        val_ret.get('val_top1', float('nan')),
            'val_top2':        val_ret.get('val_top2', float('nan')),
            'val_top3':        val_ret.get('val_top3', float('nan')),
            'val_top4':        val_ret.get('val_top4', float('nan')),
            'val_top5':        val_ret.get('val_top5', float('nan')),
            'val_mrr':         val_ret.get('val_mrr', float('nan')),
            'val_mean_cosine': val_ret.get('val_mean_cosine', float('nan')),
            'val_mean_rank':   val_ret.get('val_mean_rank', float('nan')),
            'val_median_rank': val_ret.get('val_median_rank', float('nan')),
            'lr':              current_lr,
            'epoch_time_s':    round(elapsed, 2),
        })
        # Write after every epoch so progress survives a Colab disconnect
        pd.DataFrame(epoch_log).to_csv(epoch_log_path, index=False)

        print(f'  Epoch {epoch:03d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}'
              f'  top1={val_ret.get("val_top1", float("nan")):.4f}'
              f'  lr={current_lr:.2e}  {elapsed:.1f}s', end='')

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch    = epoch
            early_stop_counter = 0
            torch.save({
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss':       val_loss,
                'embedding_dim':  embed_dim,
                'embedding_name': embedding_name,
                'model_name':     model_name,
                'loss_fn':        criterion.__class__.__name__,
                'dataset':        'b',
                'train_size':     len(train_set),
                'val_size':       len(val_set),
                'seed':           SEED,
            }, ckpt_path)
            print('  ✓ best', end='')
        else:
            early_stop_counter += 1
            print(f'  [patience {early_stop_counter}/{EARLY_STOP_PATIENCE}]', end='')
        print()

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f'  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)')
            break

    total_train_time = time.time() - train_start
    print(f'\n  Training complete: best_epoch={best_epoch}  best_val_loss={best_val_loss:.6f}'
          f'  total_time={total_train_time:.1f}s')

    print(f'  Training log saved → {epoch_log_path.name}')
    print(f'  Checkpoint saved → {ckpt_path.name}')
    print('  Run local evaluation scripts to get test metrics.')


def run_experiment_normalised(
    model_name:     str,
    embedding_name: str,
    epochs:         int  = EPOCHS,
    device:         str  = 'cpu',
    run_tag:        str  = '',
) -> None:
    """Same as run_experiment but L2-normalises target embeddings before loss.

    Normalising targets to unit norm makes MSELoss equivalent to
    (1 - cosine_similarity), directly aligning the training objective with the
    cosine-based retrieval metric used at evaluation.  Run name gets a '_normed'
    suffix so checkpoints and logs are kept separate from the raw-target runs.
    """
    set_seed(SEED)

    run_name  = f'b_{model_name}_{embedding_name}_normed' + (f'_{run_tag}' if run_tag else '')
    embed_dim = EMBEDDING_CONFIGS[embedding_name]['dim']

    print(f"\n{'='*60}")
    print(f"  Run       : {run_name}")
    print(f"  model     : {model_name}    embedding: {embedding_name}")
    print(f"  embed_dim : {embed_dim}     device: {device}")
    print(f"  targets   : L2-normalised before loss")
    print(f"{'='*60}")

    cache_path = EMBED_RESULTS_B / f'{embedding_name}_embedding_result_typeb.pt'
    if not cache_path.exists():
        print(f'[skip] Embedding file not found: {cache_path}')
        print(f'  Run: python src/embeddings/computed-embeddings/type-b/'
              f'generate_embeddings_type_b.py --embedding {embedding_name}')
        return

    img_transform = IMAGENET_TRANSFORM if model_name in _IMAGENET_NORM_MODELS else None
    train_set, val_set, _ = make_splits(
        embedding_cache=cache_path,
        device=device,
        seed=SEED,
        transform=img_transform,
    )

    num_workers = 0
    pin = False
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers)

    full_dataset   = train_set.dataset
    all_embeddings = torch.stack([rec[2] for rec in full_dataset.records])
    all_sentences  = [rec[1] for rec in full_dataset.records]

    model = MODEL_CONFIGS[model_name](embed_dim).to(device)

    lr           = _ALEXNET_LR           if model_name == 'alexnet' else LR
    weight_decay = _ALEXNET_WEIGHT_DECAY if model_name == 'alexnet' else WEIGHT_DECAY

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = _build_criterion(model_name, embedding_name)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  trainable params : {total_params:,}')
    print(f'  loss function    : {criterion.__class__.__name__} (on normalised targets)')
    print(f'  optimizer lr     : {lr}   weight_decay: {weight_decay}')
    print(f'  train={len(train_set)}  val={len(val_set)}')

    CHECKPOINTS_B.mkdir(parents=True, exist_ok=True)
    METRICS_B.mkdir(parents=True, exist_ok=True)

    ckpt_path          = CHECKPOINTS_B / f'{run_name}_best.pt'
    best_val_loss      = float('inf')
    best_epoch         = 0
    early_stop_counter = 0
    EARLY_STOP_PATIENCE = 7

    epoch_log: list[dict] = []
    epoch_log_path = METRICS_B / f'{run_name}_training_log.csv'
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch_normalised(model, train_loader, optimizer, criterion, device)
        val_loss   = run_validation_normalised(model, val_loader, criterion, device)
        val_ret    = run_val_retrieval(model, val_loader, all_embeddings, all_sentences, device)
        scheduler.step(val_loss)
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        epoch_log.append({
            'epoch':           epoch,
            'loss_fn':         criterion.__class__.__name__ + '_normed',
            'train_loss':      round(train_loss, 6),
            'val_loss':        round(val_loss, 6),
            'val_top1':        val_ret.get('val_top1', float('nan')),
            'val_top2':        val_ret.get('val_top2', float('nan')),
            'val_top3':        val_ret.get('val_top3', float('nan')),
            'val_top4':        val_ret.get('val_top4', float('nan')),
            'val_top5':        val_ret.get('val_top5', float('nan')),
            'val_mrr':         val_ret.get('val_mrr', float('nan')),
            'val_mean_cosine': val_ret.get('val_mean_cosine', float('nan')),
            'val_mean_rank':   val_ret.get('val_mean_rank', float('nan')),
            'val_median_rank': val_ret.get('val_median_rank', float('nan')),
            'lr':              current_lr,
            'epoch_time_s':    round(elapsed, 2),
        })
        pd.DataFrame(epoch_log).to_csv(epoch_log_path, index=False)

        print(f'  Epoch {epoch:03d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}'
              f'  top1={val_ret.get("val_top1", float("nan")):.4f}'
              f'  lr={current_lr:.2e}  {elapsed:.1f}s', end='')

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch    = epoch
            early_stop_counter = 0
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss':        val_loss,
                'embedding_dim':   embed_dim,
                'embedding_name':  embedding_name,
                'model_name':      model_name,
                'loss_fn':         criterion.__class__.__name__ + '_normed',
                'normalised':      True,
                'dataset':         'b',
                'train_size':      len(train_set),
                'val_size':        len(val_set),
                'seed':            SEED,
            }, ckpt_path)
            print('  ✓ best', end='')
        else:
            early_stop_counter += 1
            print(f'  [patience {early_stop_counter}/{EARLY_STOP_PATIENCE}]', end='')
        print()

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f'  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)')
            break

    total_train_time = time.time() - train_start
    print(f'\n  Training complete: best_epoch={best_epoch}  best_val_loss={best_val_loss:.6f}'
          f'  total_time={total_train_time:.1f}s')
    print(f'  Training log saved → {epoch_log_path.name}')
    print(f'  Checkpoint saved → {ckpt_path.name}')
    print('  Run local evaluation scripts to get test metrics.')


def _default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main() -> None:
    parser = argparse.ArgumentParser(description='Type-B local training')
    parser.add_argument('--model',     choices=list(MODEL_CONFIGS),     default=None,
                        help='Model architecture (default: all)')
    parser.add_argument('--embedding', choices=list(EMBEDDING_CONFIGS), default=None,
                        help='Embedding method (default: all)')
    parser.add_argument('--epochs',    type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--device',    default=_default_device(),
                        help='Device: cpu | cuda | mps (default: auto-detect)')
    args = parser.parse_args()

    models     = [args.model]     if args.model     else list(MODEL_CONFIGS)
    embeddings = [args.embedding] if args.embedding else list(EMBEDDING_CONFIGS)

    print(f'Running {len(models)} model(s) x {len(embeddings)} embedding(s) '
          f'= {len(models)*len(embeddings)} experiment(s)')
    print(f'Device: {args.device}')

    for m in models:
        for e in embeddings:
            run_experiment(m, e, epochs=args.epochs, device=args.device)

    print(f'\nAll done. Summary → {METRICS_B / "results_summary.csv"}')


if __name__ == '__main__':
    main()
