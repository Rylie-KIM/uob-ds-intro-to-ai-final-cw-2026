import os
import csv
from typing import List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from torchvision.models import ResNet18_Weights


# CONFIG: 

TEXT_PT_PATH = 
# TEXT_PT_PATH =

IMAGE_MAP_CSV = 
IMAGE_DIR = 

MODEL_SAVE_PATH =

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Dataset
def load_image_map(image_map_csv: str) -> List[Dict]:
    if not os.path.exists(image_map_csv):
        raise FileNotFoundError(f"image_map csv not found: {image_map_csv}")

    rows = []
    with open(image_map_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return rows


class ImageTextAlignDataset(Dataset):
    def __init__(self, image_map_csv: str, image_dir: str, text_pt_path: str, transform):
        self.rows = load_image_map(image_map_csv)
        self.image_dir = image_dir
        self.transform = transform

        pt_obj = torch.load(text_pt_path, map_location="cpu")

        # Support two scenarios:
        # 1) Directly save the tensor
        # 2) Save as {"embeddings": tensor, ... }
        if isinstance(pt_obj, dict) and "embeddings" in pt_obj:
            self.text_embeddings = pt_obj["embeddings"].float()
        elif isinstance(pt_obj, torch.Tensor):
            self.text_embeddings = pt_obj.float()
        else:
            raise ValueError("Unsupported PT format. Expected tensor or dict with key 'embeddings'.")

        if len(self.rows) != len(self.text_embeddings):
            raise ValueError(
                f"Length mismatch: image_map has {len(self.rows)} rows but text embeddings have {len(self.text_embeddings)} rows."
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_filename = row["image"]
        image_path = os.path.join(self.image_dir, image_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text_embedding = self.text_embeddings[idx]

        return image, text_embedding


# Model

class ResNet18TextAlign(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)

     
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, output_dim)

        self.model = backbone

    def forward(self, x):
        return self.model(x)



# Train / Eval
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def cosine_match_score(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    outputs = nn.functional.normalize(outputs, dim=1)
    targets = nn.functional.normalize(targets, dim=1)
    sims = torch.sum(outputs * targets, dim=1)
    return sims.mean().item()


@torch.no_grad()
def evaluate_cosine(model, loader):
    model.eval()
    all_scores = []

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(images)
        score = cosine_match_score(outputs, targets)
        all_scores.append(score)

    return float(np.mean(all_scores))


# main
def main():
    print(f"Using device: {DEVICE}")
    print(f"Text PT: {TEXT_PT_PATH}")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    dataset = ImageTextAlignDataset(
        image_map_csv=IMAGE_MAP_CSV,
        image_dir=IMAGE_DIR,
        text_pt_path=TEXT_PT_PATH,
        transform=transform,
    )

    target_dim = dataset.text_embeddings.shape[1]
    print(f"Dataset size: {len(dataset)}")
    print(f"Target text embedding dim: {target_dim}")

    # split
    total_size = len(dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet18TextAlign(output_dim=target_dim).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_dim": target_dim,
                    "text_pt_path": TEXT_PT_PATH,
                },
                MODEL_SAVE_PATH,
            )

    print("\nBest model saved to:", MODEL_SAVE_PATH)

    # load best model
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(model, test_loader, criterion)
    test_cos = evaluate_cosine(model, test_loader)

    print("\nTest Results")
    print(f"Test MSE Loss: {test_loss:.6f}")
    print(f"Mean Cosine Similarity: {test_cos:.6f}")


if __name__ == "__main__":
    main()
