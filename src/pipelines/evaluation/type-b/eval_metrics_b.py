from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import (
    TYPE_B_SENTENCES,
    TYPE_B_IMAGE_MAP,
    TYPE_B_IMAGES,
    TYPE_B_SPLITS,
    EMBED_RESULTS_B,
    CHECKPOINTS_B,
    CHECKPOINTS_B_NORMED,
    METRICS_B_NON_NORMED,
    METRICS_B_NORMED,
    PREDICTIONS_B,
    PREDICTIONS_B_NORMED,
    FIGURES_DIR,
    FIGURES_EVAL_NORM_B,
)

# Default transform: matches training config for scratch CNNs (CNN1Layer, CNN3Layer)
_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# ImageNet transform: required for ResNet18Pretrained (ImageNet-calibrated weights)
_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_IMAGENET_NORM_MODELS = {'resnet18_pt'}


def _get_transform(model_name: str):
    return _IMAGENET_TRANSFORM if model_name in _IMAGENET_NORM_MODELS else _TRANSFORM

# SHARED EVALUATION LOGIC METHODS 
# test_results.csv column order, matches performance_metrics.md
_RESULT_COLUMNS = [
    'run_id', 'cnn', 'embedding', 'dim', 'loss_fn',
    'best_epoch', 'total_epochs',
    'test_top1', 'test_top2', 'test_top3', 'test_top4', 'test_top5',
    'test_mrr', 'test_mean_cosine', 'test_mean_rank', 'test_median_rank',
    'colour_size_correct',
    'top1_1d', 'top1_2d', 'top1_3d', 'top1_4d', 'top1_5d', 'top1_6d',
    'mean_rank_1d', 'mean_rank_2d', 'mean_rank_3d',
    'mean_rank_4d', 'mean_rank_5d', 'mean_rank_6d',
    'mean_cosine_1d', 'mean_cosine_2d', 'mean_cosine_3d',
    'mean_cosine_4d', 'mean_cosine_5d', 'mean_cosine_6d',
]


def _auto_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def parse_sentence(s: str) -> dict[str, str]:
    parts = s.strip().split(maxsplit=2)
    if len(parts) < 3:
        return {'size': '', 'colour': '', 'number': ''}
    return {'size': parts[0], 'colour': parts[1], 'number': parts[2]}


def _build_model_factory() -> dict[str, Any]:
    import importlib.util as _ilu
    _here = Path(__file__).resolve().parent

    def _load(name, rel):
        _spec = _ilu.spec_from_file_location(name, (_here / rel).resolve())
        _mod  = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod

    _root     = next(p for p in _here.parents if (p / '.git').exists())
    _models_b = _root / 'src' / 'models' / 'type-b'

    CNN1Layer          = _load('_cnn1layer',  _models_b / 'cnn_1layer.py').CNN1Layer
    CNN3Layer          = _load('_cnn3layer',  _models_b / 'cnn_3layer.py').CNN3Layer
    ResNet18Pretrained = _load('_resnet18pt', _models_b / 'resnet18_pt.py').ResNet18Pretrained

    factory: dict[str, Any] = {
        'cnn_1layer':  lambda dim: CNN1Layer(embedding_dim=dim),
        'cnn_3layer':  lambda dim: CNN3Layer(embedding_dim=dim),
        'resnet18_pt': lambda dim: ResNet18Pretrained(embedding_dim=dim),
    }

    # Legacy ResNet18TextAlign (old script) — kept for backward compat with old checkpoints
    try:
        from src.models.resnet18 import ResNet18TextAlign
        factory['resnet18'] = lambda dim: ResNet18TextAlign(output_dim=dim)
    except SyntaxError:
        # resnet18.py has unfilled constants (TEXT_PT_PATH = , etc.)
        # Build the class manually using torchvision
        import torchvision.models as tv_models
        from torchvision.models import ResNet18_Weights

        class ResNet18TextAlign(nn.Module):   # type: ignore[no-redef]
            def __init__(self, output_dim: int) -> None:
                super().__init__()
                backbone = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT)
                backbone.fc = nn.Linear(backbone.fc.in_features, output_dim)
                self.model = backbone

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model(x)

        factory['resnet18'] = lambda dim: ResNet18TextAlign(output_dim=dim)

    return factory


def load_corpus(embedding_name: str) -> tuple[torch.Tensor, list[str]]:
    """
    Load pre-computed corpus embeddings.

    Returns
    -------
    all_embeddings : Tensor (N_corpus, dim)
    all_sentences  : list[str] — same order as embedding rows
    """
    cache_path = EMBED_RESULTS_B / f'{embedding_name}_embedding_result_typeb.pt'
    if not cache_path.exists():
        raise FileNotFoundError(
            f'Embedding cache not found: {cache_path}\n'
            f'Generate it first:\n'
            f'  python src/embeddings/computed-embeddings/type-b/'
            f'generate_embeddings_type_b.py --embedding {embedding_name}'
        )
    cache = torch.load(cache_path, map_location='cpu')
    return cache['embeddings'].float(), list(cache['sentences'])


def load_test_records(
    embedding_name: str,
    seed:           int = 42,
) -> tuple[list[tuple[Path, str, int, int]], torch.Tensor, list[str]]:
    """
    Load test-split image records and corpus data.

    Returns
    -------
    test_records   : list of (img_path, sentence, corpus_idx, n_digits)
    all_embeddings : Tensor (N_corpus, dim)
    all_sentences  : list[str]
    """
    all_embeddings, all_sentences = load_corpus(embedding_name)
    sentence_to_cidx = {s: i for i, s in enumerate(all_sentences)}

    # Build merged dataframe (same merge order as type_b_loader.py)
    image_map   = pd.read_csv(TYPE_B_IMAGE_MAP)    # filename, sentence_id
    sentences_df = pd.read_csv(TYPE_B_SENTENCES)   # sentence_id, sentence, n_digits
    df = image_map.merge(sentences_df, on='sentence_id')
    # df columns: filename, sentence_id, sentence, n_digits

    # Load split CSV
    split_csv = TYPE_B_SPLITS / f'type_b_splits_seed{seed}.csv'
    if not split_csv.exists():
        raise FileNotFoundError(
            f'Split CSV not found: {split_csv}\n'
            'Run training first to generate the split.'
        )
    split_df  = pd.read_csv(split_csv)
    test_idxs = split_df[split_df['split'] == 'test']['idx'].tolist()

    test_records: list[tuple[Path, str, int, int]] = []
    for idx in test_idxs:
        row      = df.iloc[idx]
        sentence = str(row['sentence'])
        cidx     = sentence_to_cidx.get(sentence)
        if cidx is None:
            raise KeyError(
                f"Sentence '{sentence}' not found in embedding corpus.\n"
                f"Re-run: python src/embeddings/computed-embeddings/type-b/"
                f"generate_embeddings_type_b.py --embedding {embedding_name}"
            )
        n_digits = int(row['n_digits'])
        img_path = TYPE_B_IMAGES / row['filename']
        test_records.append((img_path, sentence, cidx, n_digits))

    return test_records, all_embeddings, all_sentences


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: Path) -> tuple[nn.Module, dict[str, Any]]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    factory = _build_model_factory()
    ckpt    = torch.load(ckpt_path, map_location='cpu')

    if 'model_state' in ckpt:
        # Standard format from train_type_b.py
        model_name     = str(ckpt['model_name'])
        embedding_name = str(ckpt['embedding_name'])
        embedding_dim  = int(ckpt['embedding_dim'])
        best_epoch     = int(ckpt.get('epoch', -1))
        state_dict     = ckpt['model_state']
    elif 'model_state_dict' in ckpt:
        # Legacy format from resnet18.py
        model_name     = 'resnet18'
        embedding_dim  = int(ckpt.get('target_dim', 384))
        embedding_name = str(ckpt.get('embedding_name', 'sbert'))
        best_epoch     = int(ckpt.get('epoch', -1))
        state_dict     = ckpt['model_state_dict']
    else:
        raise ValueError(
            f'Unrecognised checkpoint format. Keys: {list(ckpt.keys())}'
        )

    if model_name not in factory:
        raise ValueError(
            f'Model "{model_name}" not in registry. '
            f'Available: {list(factory)}'
        )

    model = factory[model_name](embedding_dim)
    # load weights saved in .pt files
    model.load_state_dict(state_dict)
    # turn on every neurons and use accumulated statistical info from training
    model.eval()

    return model, {
        'model_name':     model_name,
        'embedding_name': embedding_name,
        'embedding_dim':  embedding_dim,
        'best_epoch':     best_epoch,
        'loss_fn':        str(ckpt.get('loss_fn', '')),
    }

# predicition 
class _TestImageDataset(Dataset):
    def __init__(self, img_paths: list[Path], transform) -> None:
        self.paths     = img_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


# cnn forward pass on all test sets. >> return: predicted embeddings
def predict_all(
    model:        nn.Module,
    test_records: list[tuple[Path, str, int, int]],
    device:       str,
    batch_size:   int = 64,
    transform=None,
) -> torch.Tensor:
    img_paths = [r[0] for r in test_records]
    dataset   = _TestImageDataset(img_paths, transform or _TRANSFORM)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    model.eval()
    batches: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batches.append(model(batch.to(device)).cpu())

    return torch.cat(batches, dim=0)   # (N_test, dim)


def compute_all_metrics(
    pred_embs:     torch.Tensor,
    test_records:  list[tuple[Path, str, int, int]],
    all_embeddings: torch.Tensor,
    all_sentences:  list[str],
    top_k:          tuple[int, ...] = (1, 2, 3, 4, 5),
    chunk_size:     int = 256,
) -> tuple[dict[str, float], pd.DataFrame]:
    N_test = len(test_records)

    # L2-normalise for cosine similarity via matrix multiply
    pred_norm   = F.normalize(pred_embs.float(),      dim=1)  # (N_test, dim)
    corpus_norm = F.normalize(all_embeddings.float(), dim=1)  # (N_corpus, dim)

    ranks:          list[int]   = []
    cosine_sims:    list[float] = []
    top1_cidxs:     list[int]   = []   # corpus_idx of top-1 retrieved sentence

    for start in range(0, N_test, chunk_size):
        end  = min(start + chunk_size, N_test)
        sims = pred_norm[start:end] @ corpus_norm.T   # (chunk, N_corpus)

        # Sort descending by similarity
        sorted_idx = sims.argsort(dim=1, descending=True)  # (chunk, N_corpus)

        for local_i, global_i in enumerate(range(start, end)):
            true_cidx = test_records[global_i][2]
            row       = sorted_idx[local_i]

            # 1-indexed rank of the true sentence
            pos  = (row == true_cidx).nonzero(as_tuple=True)[0]
            rank = int(pos[0].item()) + 1

            cos_sim = float(sims[local_i, true_cidx].item())
            top1_cidx = int(row[0].item())

            ranks.append(rank)
            cosine_sims.append(cos_sim)
            top1_cidxs.append(top1_cidx)

    ranks_arr = np.array(ranks, dtype=np.float64)
    N = len(ranks)

    # eval metrics 
    topk_accs   = {k: float(np.mean(ranks_arr <= k)) for k in top_k}
    mrr         = float(np.mean(1.0 / ranks_arr))
    mean_cos    = float(np.mean(cosine_sims))
    mean_rank   = float(np.mean(ranks_arr))
    median_rank = float(np.median(ranks_arr))

    # colour, size matching mterics 
    colour_size_hits = 0
    for i in range(N):
        pred_sent = all_sentences[top1_cidxs[i]]
        true_sent = test_records[i][1]
        p = parse_sentence(pred_sent)
        t = parse_sentence(true_sent)
        if p['size'] == t['size'] and p['colour'] == t['colour']:
            colour_size_hits += 1

    colour_size_correct = colour_size_hits / N

    # divide into digit lengths 
    groups: dict[int, list[int]] = {nd: [] for nd in range(1, 7)}
    for i in range(N):
        nd = test_records[i][3]
        if nd in groups:
            groups[nd].append(i)

    per_digits: dict[str, float] = {}
    for nd in range(1, 7):
        idxs = groups[nd]
        if not idxs:
            per_digits[f'top1_{nd}d']        = float('nan')
            per_digits[f'mean_rank_{nd}d']   = float('nan')
            per_digits[f'mean_cosine_{nd}d'] = float('nan')
        else:
            nd_ranks = [ranks[i] for i in idxs]
            nd_cos   = [cosine_sims[i] for i in idxs]
            per_digits[f'top1_{nd}d']        = float(np.mean(np.array(nd_ranks) <= 1))
            per_digits[f'mean_rank_{nd}d']   = float(np.mean(nd_ranks))
            per_digits[f'mean_cosine_{nd}d'] = float(np.mean(nd_cos))

    # full metrics 
    metrics: dict[str, float] = {
        'test_top1':           topk_accs.get(1, float('nan')),
        'test_top2':           topk_accs.get(2, float('nan')),
        'test_top3':           topk_accs.get(3, float('nan')),
        'test_top4':           topk_accs.get(4, float('nan')),
        'test_top5':           topk_accs.get(5, float('nan')),
        'test_mrr':            mrr,
        'test_mean_cosine':    mean_cos,
        'test_mean_rank':      mean_rank,
        'test_median_rank':    median_rank,
        'colour_size_correct': colour_size_correct,
        **per_digits,
        'n_test':              float(N),
    }

    
    per_sample = pd.DataFrame([
        {
            'true_sentence': test_records[i][1],
            'pred_sentence': all_sentences[top1_cidxs[i]],
            'n_digits':      test_records[i][3],
            'true_rank':     ranks[i],
            'cosine_sim':    cosine_sims[i],
            **{f'top_{k}_correct': int(ranks[i] <= k) for k in top_k},
        }
        for i in range(N)
    ])

    return metrics, per_sample


# result prediciton
def save_evaluation_results(
    run_id:         str,
    model_name:     str,
    embedding_name: str,
    embedding_dim:  int,
    loss_fn:        str,
    best_epoch:     int,
    total_epochs:   int,
    metrics:        dict[str, float],
    per_sample:     pd.DataFrame,
    pred_dir:       Path | None = None,
    results_path:   Path | None = None,
) -> None:
    pred_dir     = pred_dir     or PREDICTIONS_B
    results_path = results_path or (PREDICTIONS_B / 'test_results.csv')

    pred_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f'{run_id.lower()}_test_predictions.csv'
    per_sample.to_csv(pred_path, index=False)
    print(f'  [saved] predictions  → {pred_path}')

    row = {
        'run_id':         run_id,
        'cnn':            model_name,
        'embedding':      embedding_name,
        'dim':            embedding_dim,
        'loss_fn':        loss_fn,
        'best_epoch':     best_epoch,
        'total_epochs':   total_epochs,
        **metrics,
    }
    row_df = pd.DataFrame([{c: row.get(c, float('nan')) for c in _RESULT_COLUMNS}])

    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing[existing['run_id'] != run_id]
        pd.concat([existing, row_df], ignore_index=True).to_csv(results_path, index=False)
    else:
        row_df.to_csv(results_path, index=False)

    print(f'  [saved] test_results → {results_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Plotting — cross-run (Graphs 3–7)
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_run_graphs(
    test_results_csv: Path,
    predictions_dir:  Path,
    figures_dir:      Path | None = None,
) -> None:
    """
    Generate Graphs 3–7 comparing all runs in test_results.csv.

    Requires test_results.csv to contain at least one completed run.
    """
    if not test_results_csv.exists():
        print(f'  [skip] test_results.csv not found: {test_results_csv}')
        return

    df = pd.read_csv(test_results_csv)
    if df.empty:
        print('  [skip] test_results.csv is empty.')
        return

    figures_dir = figures_dir or FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_graph3_grouped_bar(df, figures_dir)
    _plot_graph4_mrr_median_rank(df, figures_dir)
    _plot_graph5_topk_by_ndigits(df, figures_dir)
    _plot_graph6_mean_rank_by_ndigits(df, figures_dir)
    _plot_graph7_cosine_distribution(df, predictions_dir, figures_dir)


def _plot_graph3_grouped_bar(df: pd.DataFrame, figures_dir: Path) -> None:
    """Graph 3: Grouped bar chart — Top-1 through Top-5 accuracy by run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_ids   = df['run_id'].tolist()
    top_k_cols = ['test_top1', 'test_top2', 'test_top3', 'test_top4', 'test_top5']
    labels     = ['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5']
    colours    = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    n_runs = len(run_ids)
    n_bars = len(top_k_cols)
    width  = 0.15
    x      = np.arange(n_runs)

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_runs), 5))
    for i, (col, label, colour) in enumerate(zip(top_k_cols, labels, colours)):
        vals = df[col].fillna(0).tolist()
        ax.bar(x + i * width, vals, width, label=label, color=colour, alpha=0.85)

    ax.set_xlabel('Run')
    ax.set_ylabel('Accuracy (0–1)')
    ax.set_title('Top-1 through Top-5 Retrieval Accuracy by Run')
    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(run_ids, rotation=20, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    out = figures_dir / 'comparison_topk_accuracy.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] Graph 3 → {out.name}')


def _plot_graph4_mrr_median_rank(df: pd.DataFrame, figures_dir: Path) -> None:
    """Graph 4: MRR and Median Rank by run (dual y-axis)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_ids     = df['run_id'].tolist()
    mrrs        = df['test_mrr'].fillna(0).tolist()
    median_ranks = df['test_median_rank'].fillna(float('nan')).tolist()

    x = np.arange(len(run_ids))
    fig, ax1 = plt.subplots(figsize=(max(8, 2 * len(run_ids)), 5))
    ax2 = ax1.twinx()

    ax1.bar(x - 0.2, mrrs,         0.4, color='steelblue', alpha=0.8, label='MRR')
    ax2.bar(x + 0.2, median_ranks, 0.4, color='tomato',    alpha=0.8, label='Median Rank')

    ax1.set_xlabel('Run')
    ax1.set_ylabel('MRR (0–1)',          color='steelblue')
    ax2.set_ylabel('Median Rank (lower = better)', color='tomato')
    ax1.set_xticks(x)
    ax1.set_xticklabels(run_ids, rotation=20, ha='right')
    ax1.set_title('MRR and Median Rank by Run')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(True, axis='y', alpha=0.3)

    out = figures_dir / 'comparison_mrr_median_rank.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] Graph 4 → {out.name}')


def _plot_graph5_topk_by_ndigits(df: pd.DataFrame, figures_dir: Path) -> None:
    """Graph 5: Top-1 accuracy by n_digits — one line per run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    digit_groups = list(range(1, 7))
    cols         = [f'top1_{nd}d' for nd in digit_groups]
    cmap         = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in df.iterrows():
        vals = [row.get(c, float('nan')) for c in cols]
        ax.plot(digit_groups, vals, marker='o', label=str(row['run_id']),
                color=cmap(i), linewidth=1.8)

    ax.set_xlabel('Number of Digits (n_digits)')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Top-1 Accuracy by Number of Digits')
    ax.set_xticks(digit_groups)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    out = figures_dir / 'comparison_top1_by_ndigits.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] Graph 5 → {out.name}')


def _plot_graph6_mean_rank_by_ndigits(df: pd.DataFrame, figures_dir: Path) -> None:
    """Graph 6: Mean rank by n_digits — one line per run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    digit_groups = list(range(1, 7))
    cols         = [f'mean_rank_{nd}d' for nd in digit_groups]
    cmap         = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in df.iterrows():
        vals = [row.get(c, float('nan')) for c in cols]
        ax.plot(digit_groups, vals, marker='o', label=str(row['run_id']),
                color=cmap(i), linewidth=1.8)

    ax.set_xlabel('Number of Digits (n_digits)')
    ax.set_ylabel('Mean Rank (lower = better)')
    ax.set_title('Mean Retrieval Rank by Number of Digits')
    ax.set_xticks(digit_groups)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    out = figures_dir / 'comparison_mean_rank_by_ndigits.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] Graph 6 → {out.name}')


def _plot_graph7_cosine_distribution(
    df:              pd.DataFrame,
    predictions_dir: Path,
    figures_dir:     Path,
) -> None:
    """
    Graph 7: Cosine similarity distribution per run.

    Overlays two histograms:
      - green: samples where rank == 1 (exact retrieval)
      - red:   samples where rank > 1 (missed)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_ids = df['run_id'].tolist()
    ncols = min(3, len(run_ids))
    nrows = (len(run_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for ax_idx, run_id in enumerate(run_ids):
        ax  = axes[ax_idx // ncols][ax_idx % ncols]
        pred_csv = predictions_dir / f'{run_id.lower()}_test_predictions.csv'

        if not pred_csv.exists():
            ax.set_title(f'{run_id}\n(predictions not found)')
            ax.axis('off')
            continue

        preds = pd.read_csv(pred_csv)
        correct  = preds[preds['true_rank'] == 1]['cosine_sim']
        incorrect = preds[preds['true_rank'] > 1]['cosine_sim']

        ax.hist(incorrect, bins=40, color='tomato',    alpha=0.6,
                label=f'rank>1 (n={len(incorrect)})', density=True)
        ax.hist(correct,  bins=40, color='mediumseagreen', alpha=0.7,
                label=f'rank=1 (n={len(correct)})',   density=True)

        ax.set_xlabel('Cosine Similarity to True Sentence')
        ax.set_ylabel('Density')
        ax.set_title(f'{run_id} — Cosine Sim Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax_idx in range(len(run_ids), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].axis('off')

    fig.tight_layout()
    out = figures_dir / 'comparison_cosine_distribution.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] Graph 7 → {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# Plotting — t-SNE corpus visualisation (Graph 8, optional)
# ══════════════════════════════════════════════════════════════════════════════

def plot_tsne_corpus(
    embedding_names: list[str] | None = None,
    colour_by:       str = 'colour',   # 'colour' | 'n_digits'
    figures_dir:     Path | None = None,
    n_iter:          int = 1000,
    perplexity:      float = 50.0,
) -> None:

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print('  [skip] Graph 8 requires scikit-learn: pip install scikit-learn')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    embedding_names = embedding_names or ['tfidf_lsa', 'sbert']
    figures_dir     = figures_dir or FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata for colouring
    sentences_df = pd.read_csv(TYPE_B_SENTENCES)   # sentence_id, sentence, n_digits
    sentence_to_nd = dict(zip(sentences_df['sentence'], sentences_df['n_digits']))

    ncols = len(embedding_names)
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7), squeeze=False)

    for col_idx, emb_name in enumerate(embedding_names):
        ax = axes[0][col_idx]

        try:
            embs, sentences = load_corpus(emb_name)
        except FileNotFoundError as exc:
            ax.set_title(f'{emb_name}\n(cache not found)')
            ax.text(0.5, 0.5, str(exc), ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, wrap=True)
            continue

        print(f'  [t-SNE] fitting {emb_name} ({embs.shape[0]} × {embs.shape[1]})…')
        import sklearn
        _tsne_kwargs = dict(n_components=2, perplexity=perplexity,
                            random_state=42, verbose=0)
        if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) >= (1, 2):
            _tsne_kwargs['max_iter'] = n_iter   # renamed in sklearn 1.2
        else:
            _tsne_kwargs['n_iter'] = n_iter
        tsne = TSNE(**_tsne_kwargs)
        coords = tsne.fit_transform(embs.numpy())   # (N, 2)

        parsed = [parse_sentence(s) for s in sentences]

        if colour_by == 'n_digits':
            labels = [sentence_to_nd.get(s, -1) for s in sentences]
            unique = sorted(set(labels))
            cmap   = plt.get_cmap('tab10')
            for lbl in unique:
                mask = np.array(labels) == lbl
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           s=2, alpha=0.4, label=f'{lbl}d',
                           color=cmap(lbl - 1))
            ax.legend(markerscale=5, fontsize=8, title='n_digits')

        elif colour_by == 'colour':
            colour_tokens  = [p['colour'] for p in parsed]
            unique_colours = sorted(set(colour_tokens))
            palette = {'red': 'red', 'blue': 'royalblue',
                       'green': 'forestgreen', 'yellow': 'gold'}
            for tok in unique_colours:
                mask = np.array(colour_tokens) == tok
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           s=2, alpha=0.4, label=tok,
                           color=palette.get(tok, 'gray'))
            ax.legend(markerscale=5, fontsize=8, title='colour')

        elif colour_by == 'size':
            size_tokens  = [p['size'] for p in parsed]
            unique_sizes = sorted(set(size_tokens))
            palette      = {'large': '#E53935', 'small': '#1E88E5'}
            for tok in unique_sizes:
                mask = np.array(size_tokens) == tok
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           s=2, alpha=0.4, label=tok,
                           color=palette.get(tok, 'gray'))
            ax.legend(markerscale=5, fontsize=8, title='size')

        else:
            raise ValueError(f'colour_by must be one of: n_digits, colour, size. Got: {colour_by!r}')

        ax.set_title(f't-SNE: {emb_name} (coloured by {colour_by})')
        ax.set_xlabel('t-SNE dim 1')
        ax.set_ylabel('t-SNE dim 2')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Corpus Embedding Space (t-SNE projection)', fontsize=13)
    fig.tight_layout()
    out = figures_dir / f'tsne_corpus_{colour_by}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] Graph 8 → {out.name}')


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    run_id:         str,
    model_name:     str,
    embedding_name: str,
    loss_fn:        str,
    device:         str | None = None,
    ckpt_path:      Path | None = None,
    seed:           int = 42,
    batch_size:     int = 64,
) -> dict[str, float]:
    device    = device or _auto_device()
    run_label = run_id.lower()

    # ── Checkpoint path ────────────────────────────────────────────────────────
    if ckpt_path is None:
        # Files are named: b_{model}_{embedding}_{run_tag}_{timestamp}_best.pt
        # Pick the most recently modified match.
        candidates = sorted(
            CHECKPOINTS_B.glob(f'b_{model_name}_{embedding_name}_*_best.pt'),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f'No checkpoint found for {model_name} + {embedding_name} in {CHECKPOINTS_B}.\n'
                f'  Pattern: b_{model_name}_{embedding_name}_*_best.pt\n'
                f'  Run training first.'
            )
        ckpt_path = candidates[-1]   # most recent

    print(f'\n{"="*60}')
    print(f'  Evaluating: {run_id}')
    print(f'  CNN       : {model_name}')
    print(f'  Embedding : {embedding_name}')
    print(f'  Loss      : {loss_fn}')
    print(f'  Device    : {device}')
    print(f'  Checkpoint: {ckpt_path.name}')
    print(f'{"="*60}')

    # ── Load model ─────────────────────────────────────────────────────────────
    model, meta = load_model(ckpt_path)
    best_epoch  = meta['best_epoch']
    embed_dim   = meta['embedding_dim']
    print(f'  best_epoch    : {best_epoch}')
    print(f'  embedding_dim : {embed_dim}')

    # ── Load test data ─────────────────────────────────────────────────────────
    print('  Loading test records…')
    test_records, all_embeddings, all_sentences = load_test_records(
        embedding_name=embedding_name, seed=seed
    )
    print(f'  test samples  : {len(test_records)}')
    print(f'  corpus size   : {len(all_sentences)}')

    # ── Predict ────────────────────────────────────────────────────────────────
    print('  Running CNN inference…')
    pred_embs = predict_all(model, test_records, device=device, batch_size=batch_size,
                            transform=_get_transform(model_name))
    print(f'  pred_embs shape: {tuple(pred_embs.shape)}')

    # ── Compute metrics ────────────────────────────────────────────────────────
    print('  Computing retrieval metrics…')
    metrics, per_sample = compute_all_metrics(
        pred_embs=pred_embs,
        test_records=test_records,
        all_embeddings=all_embeddings,
        all_sentences=all_sentences,
    )

    # Determine total_epochs from training log (if available)
    # Derive the training_log path from the checkpoint stem:
    #   b_{model}_{embedding}_{run_tag}_{timestamp}_best.pt
    #   → b_{model}_{embedding}_{run_tag}_{timestamp}_training_log.csv
    log_stem       = ckpt_path.stem.replace('_best', '_training_log')
    epoch_log_path = METRICS_B_NON_NORMED / f'{log_stem}.csv'
    total_epochs   = best_epoch   # fallback
    if epoch_log_path.exists():
        try:
            log_df       = pd.read_csv(epoch_log_path)
            total_epochs = int(log_df['epoch'].max())
        except Exception:
            pass

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f'\n  === Test Results ({run_id}) ===')
    print(f'  top-1  : {metrics["test_top1"]:.4f}')
    print(f'  top-5  : {metrics["test_top5"]:.4f}')
    print(f'  MRR    : {metrics["test_mrr"]:.4f}')
    print(f'  cosine : {metrics["test_mean_cosine"]:.4f}')
    print(f'  mean rank   : {metrics["test_mean_rank"]:.1f}')
    print(f'  median rank : {metrics["test_median_rank"]:.1f}')
    print(f'  colour+size : {metrics["colour_size_correct"]:.4f}')

    # ── Save results ───────────────────────────────────────────────────────────
    save_evaluation_results(
        run_id=run_id,
        model_name=model_name,
        embedding_name=embedding_name,
        embedding_dim=embed_dim,
        loss_fn=loss_fn,
        best_epoch=best_epoch,
        total_epochs=total_epochs,
        metrics=metrics,
        per_sample=per_sample,
        pred_dir=PREDICTIONS_B,
        results_path=PREDICTIONS_B / 'test_results.csv',
    )

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end evaluation — normalised runs
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation_normed(
    run_id:         str,
    model_name:     str,
    embedding_name: str,
    loss_fn:        str,
    device:         str | None = None,
    ckpt_path:      Path | None = None,
    seed:           int = 42,
    batch_size:     int = 64,
) -> dict[str, float]:
    device    = device or _auto_device()

    # ── Checkpoint path ────────────────────────────────────────────────────────
    # Normalised checkpoint filenames:
    #   b_{model}_{embedding}_normed_{run_tag}_{timestamp}_best.pt
    if ckpt_path is None:
        candidates = sorted(
            CHECKPOINTS_B_NORMED.glob(
                f'b_{model_name}_{embedding_name}_normed_*_best.pt'
            ),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f'No normalised checkpoint found for {model_name} + {embedding_name} '
                f'in {CHECKPOINTS_B_NORMED}.\n'
                f'  Pattern: b_{model_name}_{embedding_name}_normed_*_best.pt\n'
                f'  Run training first (run_experiment_normalised).'
            )
        ckpt_path = candidates[-1]

    print(f'\n{"="*60}')
    print(f'  Evaluating (normed): {run_id}')
    print(f'  CNN               : {model_name}')
    print(f'  Embedding         : {embedding_name} (L2-normed targets)')
    print(f'  Loss              : {loss_fn}')
    print(f'  Device            : {device}')
    print(f'  Checkpoint        : {ckpt_path.name}')
    print(f'{"="*60}')

    # ── Load model ─────────────────────────────────────────────────────────────
    model, meta = load_model(ckpt_path)
    best_epoch  = meta['best_epoch']
    embed_dim   = meta['embedding_dim']
    # embedding_name in checkpoint is the BASE name (e.g. 'sbert'), not 'sbert_normed'
    ckpt_emb    = meta['embedding_name']
    # Use the loss_fn stored in the checkpoint so the CSV reflects the actual
    # criterion used during training, not the (potentially wrong) registry value.
    ckpt_loss_fn = meta.get('loss_fn', '')
    if ckpt_loss_fn:
        loss_fn = ckpt_loss_fn
    print(f'  best_epoch    : {best_epoch}')
    print(f'  embedding_dim : {embed_dim}')
    print(f'  ckpt emb_name : {ckpt_emb}')

    # ── Load test data ─────────────────────────────────────────────────────────
    print('  Loading test records…')
    test_records, all_embeddings, all_sentences = load_test_records(
        embedding_name=ckpt_emb, seed=seed
    )
    print(f'  test samples  : {len(test_records)}')
    print(f'  corpus size   : {len(all_sentences)}')

    # ── Predict ────────────────────────────────────────────────────────────────
    print('  Running CNN inference…')
    pred_embs = predict_all(model, test_records, device=device, batch_size=batch_size,
                            transform=_get_transform(model_name))
    print(f'  pred_embs shape: {tuple(pred_embs.shape)}')

    # ── Compute metrics ────────────────────────────────────────────────────────
    print('  Computing retrieval metrics…')
    metrics, per_sample = compute_all_metrics(
        pred_embs=pred_embs,
        test_records=test_records,
        all_embeddings=all_embeddings,
        all_sentences=all_sentences,
    )

    # ── Resolve training log ───────────────────────────────────────────────────
    # Normed log: METRICS_B_NORMED / b_{model}_{embedding}_normed_{tag}_{ts}_training_log.csv
    log_stem       = ckpt_path.stem.replace('_best', '_training_log')
    epoch_log_path = METRICS_B_NORMED / f'{log_stem}.csv'
    total_epochs   = best_epoch
    if epoch_log_path.exists():
        try:
            log_df       = pd.read_csv(epoch_log_path)
            total_epochs = int(log_df['epoch'].max())
        except Exception:
            pass

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f'\n  === Test Results ({run_id} — normed) ===')
    print(f'  top-1  : {metrics["test_top1"]:.4f}')
    print(f'  top-5  : {metrics["test_top5"]:.4f}')
    print(f'  MRR    : {metrics["test_mrr"]:.4f}')
    print(f'  cosine : {metrics["test_mean_cosine"]:.4f}')
    print(f'  mean rank   : {metrics["test_mean_rank"]:.1f}')
    print(f'  median rank : {metrics["test_median_rank"]:.1f}')
    print(f'  colour+size : {metrics["colour_size_correct"]:.4f}')

    # ── Save results ───────────────────────────────────────────────────────────
    save_evaluation_results(
        run_id=run_id,
        model_name=model_name,
        embedding_name=f'{ckpt_emb}_normed',   # distinguish from base in summary CSV
        embedding_dim=embed_dim,
        loss_fn=loss_fn,
        best_epoch=best_epoch,
        total_epochs=total_epochs,
        metrics=metrics,
        per_sample=per_sample,
        pred_dir=PREDICTIONS_B_NORMED,
        results_path=PREDICTIONS_B_NORMED / 'test_results_normed.csv',
    )

    return metrics
