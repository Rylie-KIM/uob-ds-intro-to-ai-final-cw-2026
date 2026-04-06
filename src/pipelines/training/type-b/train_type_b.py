"""
src/pipelines/training/type-b/train_type_b.py
Local training script for Type-B (coloured MNIST numbers).

Pre-requisite
-------------
Generate sentence embeddings before training:
    python src/embeddings/computed-embeddings/type-b/generate_embeddings_type_b.py

Usage
-----
# Train a single model x embedding combination
python src/pipelines/training/type-b/train_type_b.py --model cnn --embedding sbert

# Train all combinations (4 models x 7 embeddings = 28 runs)
python src/pipelines/training/type-b/train_type_b.py

# Custom options
python src/pipelines/training/type-b/train_type_b.py --model alexnet --embedding bert_mean --epochs 20 --device cpu

Available models     : alexnet | cnn | cnn_1layer | cnn_2layer
Available embeddings : sbert | bert_mean | bert_pooler | tinybert_mean | tinybert_pooler
                       word2vec_skipgram | word2vec_pretrained | tfidf | tfidf_w2v

Outputs (per run)
-----------------
Checkpoints  : src/pipelines/results/checkpoints/b_{model}_{embedding}_best.pt
Epoch log    : src/pipelines/results/metrics/b_{model}_{embedding}_epoch_log.csv
Predictions  : src/pipelines/results/metrics/b_{model}_{embedding}_predictions.csv
Summary      : src/pipelines/results/metrics/results_summary.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util as _ilu

from src.models.alexnet    import AlexNet128
from src.models.CNN        import CNN
from src.models.cnn_1layer import CNN1Layer
from src.models.cnn_2layer import CNN2Layer
from src.evaluation.evaluate import evaluate, save_results
from src.pipelines.shared   import train_one_epoch, run_validation, set_seed, CombinedLoss
from src.config.paths import EMBED_RESULTS_B, CHECKPOINTS_B, METRICS_B

# src/pipelines/data_loaders/type-b/ has a hyphen — not directly importable.
_loader_spec = _ilu.spec_from_file_location(
    'type_b_loader',
    _ROOT / 'src' / 'pipelines' / 'data_loaders' / 'type-b' / 'type_b_loader.py',
)
_loader_mod = _ilu.module_from_spec(_loader_spec)
_loader_spec.loader.exec_module(_loader_mod)
make_splits = _loader_mod.make_splits


# ── Embedding configurations ───────────────────────────────────────────────────
EMBEDDING_CONFIGS: dict[str, dict] = {
    'sbert':               {'dim': 384},
    'sbert_finetuned':     {'dim': 384},
    'bert_mean':           {'dim': 768},
    'bert_pooler':         {'dim': 768},
    'tinybert_mean':       {'dim': 312},
    'tinybert_pooler':     {'dim': 312},
    'word2vec_skipgram':   {'dim': 100},
    'word2vec_pretrained': {'dim': 300},
    'tfidf':               {'dim': 100},
    'tfidf_lsa':           {'dim': 100},
    'tfidf_w2v':           {'dim': 100},
}

# ── Model configurations ───────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, callable] = {
    'alexnet':    lambda dim: AlexNet128(embedding_dim=dim),
    'cnn':        lambda dim: CNN(embedding_dim=dim),
    'cnn_1layer': lambda dim: CNN1Layer(embedding_dim=dim),
    'cnn_2layer': lambda dim: CNN2Layer(embedding_dim=dim),
}

# ── Default hyperparameters (from src/config/training.json + hyperparams.json) ─
BATCH_SIZE   = 64
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
SEED         = 42

# AlexNet uses a lower LR due to its larger parameter count (~20M vs ~1.2M)
_ALEXNET_LR           = 5e-5
_ALEXNET_WEIGHT_DECAY = 5e-4

# Loss function selection per model (from src/config/loss.json)
# cnn_3layer and alexnet use CombinedLoss; simpler models use MSELoss
_COMBINED_LOSS_MODELS = {'cnn', 'alexnet'}
# SBERT-family embeddings prefer CosineLoss
_COSINE_LOSS_EMBEDDINGS = {'sbert', 'sbert_finetuned'}


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


# ══════════════════════════════════════════════════════════════════════════════
# Single experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    model_name:     str,
    embedding_name: str,
    epochs:         int  = EPOCHS,
    device:         str  = 'cpu',
) -> None:
    set_seed(SEED)

    run_name  = f'b_{model_name}_{embedding_name}'
    embed_dim = EMBEDDING_CONFIGS[embedding_name]['dim']

    print(f"\n{'='*60}")
    print(f"  Run       : {run_name}")
    print(f"  model     : {model_name}    embedding: {embedding_name}")
    print(f"  embed_dim : {embed_dim}     device: {device}")
    print(f"{'='*60}")

    # ── Embedding cache ────────────────────────────────────────────────────────
    cache_path = EMBED_RESULTS_B / f'{embedding_name}_embedding_result_typeb.pt'
    if not cache_path.exists():
        print(f'[skip] Embedding file not found: {cache_path}')
        print(f'  Run: python src/embeddings/computed-embeddings/type-b/'
              f'generate_embeddings_type_b.py --embedding {embedding_name}')
        return

    # ── Data splits ────────────────────────────────────────────────────────────
    train_set, val_set, test_set = make_splits(
        embedding_cache=cache_path,
        device=device,
        seed=SEED,
    )

    pin = device == 'cuda'
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Full corpus for retrieval evaluation (all sentences + embeddings)
    full_dataset   = train_set.dataset
    all_embeddings = torch.stack([rec[2] for rec in full_dataset.records])
    all_sentences  = [rec[1] for rec in full_dataset.records]

    # ── Model, optimiser, loss ─────────────────────────────────────────────────
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
    print(f'  train={len(train_set)}  val={len(val_set)}  test={len(test_set)}')

    # ── Training loop ──────────────────────────────────────────────────────────
    CHECKPOINTS_B.mkdir(parents=True, exist_ok=True)
    METRICS_B.mkdir(parents=True, exist_ok=True)

    ckpt_path     = CHECKPOINTS_B / f'{run_name}_best.pt'
    best_val_loss = float('inf')
    best_epoch    = 0
    early_stop_counter = 0
    EARLY_STOP_PATIENCE = 7

    epoch_log: list[dict] = []
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = run_validation(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        epoch_log.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': current_lr})

        print(f'  Epoch {epoch:03d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}'
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
                'dataset':        'b',
                'train_size':     len(train_set),
                'val_size':       len(val_set),
                'test_size':      len(test_set),
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

    # ── Save full epoch log ────────────────────────────────────────────────────
    epoch_log_path = METRICS_B / f'{run_name}_epoch_log.csv'
    pd.DataFrame(epoch_log).to_csv(epoch_log_path, index=False)
    print(f'  Epoch log saved  → {epoch_log_path.name}')

    # ── Evaluate best checkpoint on test set ───────────────────────────────────
    print('\n  Evaluating best checkpoint on test set...')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    metrics, results_df = evaluate(
        model=model,
        test_loader=test_loader,
        all_embeddings=all_embeddings,
        all_sentences=all_sentences,
        device=device,
        top_k=(1, 5),
    )

    print(f'  top-1 accuracy   : {metrics["top_1_acc"]:.4f}')
    print(f'  top-5 accuracy   : {metrics["top_5_acc"]:.4f}')
    print(f'  mean cosine sim  : {metrics["mean_cosine_sim"]:.4f}')
    print(f'  mean rank        : {metrics["mean_rank"]:.1f}')

    save_results(metrics, results_df, METRICS_B, run_name)


# ── Device auto-detection ──────────────────────────────────────────────────────

def _default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ── CLI ────────────────────────────────────────────────────────────────────────

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
