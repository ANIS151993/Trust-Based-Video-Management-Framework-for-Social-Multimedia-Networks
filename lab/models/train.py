"""Transfer-learning training pipeline for VGG16 and ResNet50.

Mirrors the methodology described in the paper (Section III-C): ImageNet-
pretrained convolutional base frozen, a custom head (Flatten -> Dense(128,
ReLU) -> Dropout(0.5) -> Dense(num_classes)) trained with Adam
(lr=1e-3), categorical cross-entropy, batch size 32, on a stratified
70/15/15 train/val/test split -- run here against the synthetic silhouette
proxy dataset (see data/synthetic_dataset.py) rather than a real weapon
image corpus.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ResNet50_Weights,
    VGG16_Weights,
    resnet50,
    vgg16,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, DATA_DIR, RANDOM_SEED, RESULTS_DIR  # noqa: E402


class SilhouetteDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], train: bool, img_size: int):
        self.paths = paths
        self.labels = labels
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2),
                    transforms.RandomAffine(0, scale=(0.9, 1.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), self.labels[idx]


def stratified_split(data_dir: Path, seed: int = RANDOM_SEED):
    rng = np.random.RandomState(seed)
    train_p, train_l, val_p, val_l, test_p, test_l = [], [], [], [], [], []
    for label, cls in enumerate(CLASSES):
        files = sorted((data_dir / cls).glob("*.png"))
        idx = rng.permutation(len(files))
        n = len(files)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        for i, j in enumerate(idx):
            f = files[j]
            if i < n_train:
                train_p.append(f)
                train_l.append(label)
            elif i < n_train + n_val:
                val_p.append(f)
                val_l.append(label)
            else:
                test_p.append(f)
                test_l.append(label)
    return (train_p, train_l), (val_p, val_l), (test_p, test_l)


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "vgg16":
        base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for p in base.features.parameters():
            p.requires_grad = False
        in_features = base.classifier[0].in_features  # 25088 at 224x224, adaptive pool keeps this fixed
        base.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        return base
    elif name == "resnet50":
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for p in base.parameters():
            p.requires_grad = False
        in_features = base.fc.in_features  # 2048
        base.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        return base
    raise ValueError(name)


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    torch.set_grad_enabled(True)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_preds.append(out.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    return {
        "accuracy": acc,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def train_model(
    name: str,
    data_dir: Path = DATA_DIR,
    img_size: int = 128,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = RANDOM_SEED,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (train_p, train_l), (val_p, val_l), (test_p, test_l) = stratified_split(data_dir, seed)
    train_ds = SilhouetteDataset(train_p, train_l, train=True, img_size=img_size)
    val_ds = SilhouetteDataset(val_p, val_l, train=False, img_size=img_size)
    test_ds = SilhouetteDataset(test_p, test_l, train=False, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_model(name, len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable_params(model), lr=lr)

    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc}
        )
        print(
            f"[{name}] epoch {epoch}/{epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
    train_time = time.time() - t0

    test_metrics = evaluate(model, test_loader, device)
    print(f"[{name}] TEST accuracy={test_metrics['accuracy']:.4f} f1={test_metrics['f1_macro']:.4f}")

    result = {
        "model": name,
        "img_size": img_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_train": len(train_p),
        "n_val": len(val_p),
        "n_test": len(test_p),
        "train_time_seconds": train_time,
        "history": history,
        "test": test_metrics,
        "classes": CLASSES,
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train VGG16 and/or ResNet50 on the synthetic testbed.")
    parser.add_argument("--models", nargs="+", default=["vgg16", "resnet50"], choices=["vgg16", "resnet50"])
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name in args.models:
        res = train_model(name, img_size=args.img_size, epochs=args.epochs, batch_size=args.batch_size)
        all_results[name] = res
        with open(RESULTS_DIR / f"{name}_results.json", "w") as f:
            json.dump(res, f, indent=2)
    print(f"Saved results to {RESULTS_DIR}")
