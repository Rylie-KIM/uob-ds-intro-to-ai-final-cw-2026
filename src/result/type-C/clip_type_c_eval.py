import csv
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

import clip  # pip install git+https://github.com/openai/CLIP.git


# =========================
# CONFIG
# =========================
SEED = 42
BATCH_SIZE = 32

PROJECT_ROOT = Path(r"D:\Users\git-repo\1")

IMAGE_DIR = PROJECT_ROOT / "src" / "data" / "images" / "type-c"
IMAGE_MAP_PATH = PROJECT_ROOT / "src" / "data" / "type-c" / "image_map_c.csv"
SENTENCE_CSV_PATH = PROJECT_ROOT / "src" / "data" / "type-c" / "sentences_c.csv"

OUTPUT_DIR = PROJECT_ROOT / "results22" / "CLIP"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ViT-B/32"
RUN_ID = "clip_vitb32"

TEST_RESULTS_CSV = OUTPUT_DIR / "test_results.csv"
BY_MOVES_CSV = OUTPUT_DIR / "by_moves_results.csv"
TEST_PREDICTIONS_CSV = OUTPUT_DIR / "test_predictions.csv"
SPLIT_MANIFEST_CSV = OUTPUT_DIR / "split_manifest.csv"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# =========================
# SEED
# =========================
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# HELPERS
# =========================
def save_csv_rows(rows: List[Dict], csv_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(row: Dict, csv_path: Path):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_csv_rows(rows: List[Dict], csv_path: Path):
    if not rows:
        return
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def load_sentence_meta(sentence_csv_path: Path) -> Dict[str, Dict]:
    meta = {}
    with open(sentence_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["sentence_id"]
            meta[sid] = {
                "sentence": row.get("sentence", ""),
                "notation": row.get("notation", ""),
                "n_moves": int(row.get("n_moves", 0)),
                "winner": row.get("winner", ""),
            }
    return meta


# =========================
# DATASET
# =========================
class TypeCTestDataset(Dataset):
    def __init__(self, image_dir: Path, image_map_path: Path, sentence_csv_path: Path, preprocess):
        self.image_dir = image_dir
        self.map_df = pd.read_csv(image_map_path)
        self.sentence_meta = load_sentence_meta(sentence_csv_path)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.map_df)

    def __getitem__(self, idx):
        row = self.map_df.iloc[idx]
        image_name = row["filename"]
        sentence_id = row["sentence_id"]

        image_path = self.image_dir / image_name
        if not image_path.exists() and image_name.startswith("c_"):
            alt_name = "type_" + image_name
            image_path = self.image_dir / alt_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image)

        meta_row = self.sentence_meta[sentence_id]
        sample_meta = {
            "index": idx,
            "filename": image_name,
            "sentence_id": sentence_id,
            "sentence": meta_row["sentence"],
            "n_moves": meta_row["n_moves"],
        }

        return image_tensor, sample_meta


def save_split_manifest(dataset, train_set, val_set, test_set, csv_path: Path):
    rows = []

    for split_name, subset in [("train", train_set), ("val", val_set), ("test", test_set)]:
        assert isinstance(subset, Subset)
        for idx in subset.indices:
            row = dataset.map_df.iloc[idx]
            sid = row["sentence_id"]
            meta = dataset.sentence_meta[sid]

            rows.append({
                "run_id": RUN_ID,
                "split": split_name,
                "index": int(idx),
                "filename": row["filename"],
                "sentence_id": sid,
                "n_moves": meta["n_moves"],
                "sentence": meta["sentence"],
            })

    save_csv_rows(rows, csv_path)


# =========================
# ENCODE
# =========================
@torch.no_grad()
def encode_test_images(model, loader, device):
    model.eval()

    all_img_feats = []
    all_meta = []

    for images, meta in loader:
        images = images.to(device)
        img_feats = model.encode_image(images)
        img_feats = F.normalize(img_feats, dim=1)

        all_img_feats.append(img_feats.cpu())

        batch_size = images.size(0)
        for i in range(batch_size):
            all_meta.append({
                "index": int(meta["index"][i]),
                "filename": meta["filename"][i],
                "sentence_id": meta["sentence_id"][i],
                "sentence": meta["sentence"][i],
                "n_moves": int(meta["n_moves"][i]),
            })

    all_img_feats = torch.cat(all_img_feats, dim=0)
    return all_img_feats, all_meta


@torch.no_grad()
def encode_test_texts(model, meta, device):
    sentences = [m["sentence"] for m in meta]
    text_tokens = clip.tokenize(sentences, truncate=True).to(device)
    txt_feats = model.encode_text(text_tokens)
    txt_feats = F.normalize(txt_feats, dim=1)
    return txt_feats.cpu()


# =========================
# EVAL
# =========================
@torch.no_grad()
def evaluate_retrieval(img_feats: torch.Tensor, txt_feats: torch.Tensor, meta: List[Dict]):
    sim_matrix = img_feats @ txt_feats.T
    num_samples = sim_matrix.size(0)
    true_indices = torch.arange(num_samples)

    ranked_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    ranks = []
    correct_cosines = []
    prediction_rows = []

    for i in range(num_samples):
        rank_pos = (ranked_indices[i] == true_indices[i]).nonzero(as_tuple=False).item() + 1
        ranks.append(rank_pos)

        correct_cos = sim_matrix[i, true_indices[i]].item()
        correct_cosines.append(correct_cos)

        top1_idx = ranked_indices[i, 0].item()
        top1_cos = sim_matrix[i, top1_idx].item()

        top5_indices = ranked_indices[i, :5].tolist()
        top5_sentence_ids = [meta[j]["sentence_id"] for j in top5_indices]

        prediction_rows.append({
            "run_id": RUN_ID,
            "filename": meta[i]["filename"],
            "sentence_id": meta[i]["sentence_id"],
            "sentence": meta[i]["sentence"],
            "n_moves": meta[i]["n_moves"],
            "is_top1_correct": int(top1_idx == i),
            "true_rank": int(rank_pos),
            "correct_cosine": float(correct_cos),
            "pred_top1_sentence_id": meta[top1_idx]["sentence_id"],
            "pred_top1_sentence": meta[top1_idx]["sentence"],
            "pred_top1_cosine": float(top1_cos),
            "pred_top5_sentence_ids": " | ".join(top5_sentence_ids),
        })

    ranks = np.array(ranks, dtype=np.int32)
    correct_cosines = np.array(correct_cosines, dtype=np.float32)

    overall = {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_cosine": float(np.mean(correct_cosines)),
    }

    by_moves_rows = []
    n_moves_values = sorted(set(m["n_moves"] for m in meta))

    for m in n_moves_values:
        mask = np.array([sample["n_moves"] == m for sample in meta], dtype=bool)
        group_ranks = ranks[mask]
        group_cos = correct_cosines[mask]

        by_moves_rows.append({
            "run_id": RUN_ID,
            "cnn": "clip_vitb32",
            "embedding": "clip",
            "n_moves": int(m),
            "count": int(mask.sum()),
            "top1": float(np.mean(group_ranks <= 1)),
            "top5": float(np.mean(group_ranks <= 5)),
            "mean_rank": float(np.mean(group_ranks)),
            "median_rank": float(np.median(group_ranks)),
            "mrr": float(np.mean(1.0 / group_ranks)),
            "mean_cosine": float(np.mean(group_cos)),
        })

    return overall, by_moves_rows, prediction_rows


# =========================
# MAIN
# =========================
def main():
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model, preprocess = clip.load(MODEL_NAME, device=device)
    print("Loaded CLIP model:", MODEL_NAME)

    dataset = TypeCTestDataset(
        image_dir=IMAGE_DIR,
        image_map_path=IMAGE_MAP_PATH,
        sentence_csv_path=SENTENCE_CSV_PATH,
        preprocess=preprocess,
    )

    total_size = len(dataset)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    save_split_manifest(dataset, train_set, val_set, test_set, SPLIT_MANIFEST_CSV)

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Total samples: {len(dataset)}")
    print(f"Test samples: {len(test_set)}")

    img_feats, meta = encode_test_images(model, test_loader, device)
    txt_feats = encode_test_texts(model, meta, device)

    overall, by_moves_rows, prediction_rows = evaluate_retrieval(img_feats, txt_feats, meta)

    print("\n=== CLIP TEST ===")
    print(f"Top-1 Accuracy: {overall['top1']:.6f}")
    print(f"Top-5 Accuracy: {overall['top5']:.6f}")
    print(f"Mean Rank: {overall['mean_rank']:.4f}")
    print(f"Median Rank: {overall['median_rank']:.4f}")
    print(f"MRR: {overall['mrr']:.6f}")
    print(f"Mean Cosine Similarity: {overall['mean_cosine']:.6f}")

    result_row = {
        "run_id": RUN_ID,
        "cnn": "clip_vitb32",
        "embedding": "clip",
        "raw_text_dim": 512,   # ViT-B/32 CLIP output dim
        "shared_dim": 512,
        "loss_fn": "none_zero_shot",
        "optimizer": "none_zero_shot",
        "learning_rate": 0,
        "batch_size": BATCH_SIZE,
        "best_epoch": 0,
        "total_epochs": 0,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "image_size": 224,
        "test_mse_loss": np.nan,
        "test_mean_cosine": overall["mean_cosine"],
        "test_top1": overall["top1"],
        "test_top5": overall["top5"],
        "test_mean_rank": overall["mean_rank"],
        "test_median_rank": overall["median_rank"],
        "test_mrr": overall["mrr"],
    }

    append_csv_row(result_row, TEST_RESULTS_CSV)
    append_csv_rows(by_moves_rows, BY_MOVES_CSV)
    append_csv_rows(prediction_rows, TEST_PREDICTIONS_CSV)

    print("Saved:", TEST_RESULTS_CSV)
    print("Saved:", BY_MOVES_CSV)
    print("Saved:", TEST_PREDICTIONS_CSV)
    print("Saved:", SPLIT_MANIFEST_CSV)


if __name__ == "__main__":
    main()