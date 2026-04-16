#!/usr/bin/env python3
"""
Train a baseline CNN on the extracted patch dataset and saved split indices.

Expected inputs inside --dataset-dir:
- X.npy
- y.npy
- train_idx.npy
- val_idx.npy
- test_idx.npy

Outputs inside --output-dir:
- best_model.pt
- metrics.json
- history.csv
- test_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for train_cnn_baseline.py. "
        "Install torch in your training environment first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline CNN on extracted patches")
    parser.add_argument("--dataset-dir", default="patch_dataset")
    parser.add_argument("--output-dir", default="cnn_baseline_runs/default")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-samples", type=int, default=50000)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NpyPatchDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, idx_path: str, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = np.load(idx_path)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = np.array(self.X[idx], dtype=np.float32, copy=True)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), idx


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def compute_channel_stats(x_path: str, train_idx_path: str, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path, mmap_mode="r")
    train_idx = np.load(train_idx_path)
    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(train_idx) > max_samples:
        train_idx = rng.choice(train_idx, size=max_samples, replace=False)

    channel_sum = None
    channel_sq_sum = None
    total_pixels = 0

    batch_size = 256
    for start in range(0, len(train_idx), batch_size):
        batch_idx = train_idx[start : start + batch_size]
        batch = np.array(X[batch_idx], dtype=np.float64, copy=False)
        batch_sum = batch.sum(axis=(0, 2, 3))
        batch_sq_sum = np.square(batch).sum(axis=(0, 2, 3))
        pixels = batch.shape[0] * batch.shape[2] * batch.shape[3]
        if channel_sum is None:
            channel_sum = batch_sum
            channel_sq_sum = batch_sq_sum
        else:
            channel_sum += batch_sum
            channel_sq_sum += batch_sq_sum
        total_pixels += pixels

    mean = channel_sum / total_pixels
    var = channel_sq_sum / total_pixels - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def make_class_weights(y_path: str, train_idx_path: str, num_classes: int) -> torch.Tensor:
    y = np.load(y_path, mmap_mode="r")
    idx = np.load(train_idx_path)
    counts = np.bincount(np.array(y[idx], dtype=np.int64), minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    preds_all = []
    labels_all = []
    sample_ids_all = []

    with torch.no_grad():
        for x, y, sample_ids in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
            preds_all.append(pred.cpu().numpy())
            labels_all.append(y.cpu().numpy())
            sample_ids_all.append(sample_ids.numpy())

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "preds": np.concatenate(preds_all) if preds_all else np.array([], dtype=np.int64),
        "labels": np.concatenate(labels_all) if labels_all else np.array([], dtype=np.int64),
        "sample_ids": np.concatenate(sample_ids_all) if sample_ids_all else np.array([], dtype=np.int64),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> list[list[int]]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        matrix[int(y_true), int(y_pred)] += 1
    return matrix.tolist()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    x_path = os.path.join(args.dataset_dir, "X.npy")
    y_path = os.path.join(args.dataset_dir, "y.npy")
    train_idx_path = os.path.join(args.dataset_dir, "train_idx.npy")
    val_idx_path = os.path.join(args.dataset_dir, "val_idx.npy")
    test_idx_path = os.path.join(args.dataset_dir, "test_idx.npy")

    mean, std = compute_channel_stats(x_path, train_idx_path, args.stats_samples, args.seed)
    np.save(os.path.join(args.output_dir, "channel_mean.npy"), mean)
    np.save(os.path.join(args.output_dir, "channel_std.npy"), std)

    train_ds = NpyPatchDataset(x_path, y_path, train_idx_path, mean=mean, std=std)
    val_ds = NpyPatchDataset(x_path, y_path, val_idx_path, mean=mean, std=std)
    test_ds = NpyPatchDataset(x_path, y_path, test_idx_path, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_classes = int(np.load(y_path, mmap_mode="r").max()) + 1
    in_channels = int(np.load(x_path, mmap_mode="r").shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(in_channels=in_channels, num_classes=num_classes).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        class_weights = make_class_weights(y_path, train_idx_path, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["acc"])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print("Early stopping triggered")
                break

    with open(os.path.join(args.output_dir, "history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    with open(os.path.join(args.output_dir, "test_predictions.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "y_true", "y_pred"])
        writer.writeheader()
        for sample_id, y_true, y_pred in zip(test_metrics["sample_ids"], test_metrics["labels"], test_metrics["preds"]):
            writer.writerow(
                {"sample_id": int(sample_id), "y_true": int(y_true), "y_pred": int(y_pred)}
            )

    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "in_channels": in_channels,
        "num_classes": num_classes,
        "device": str(device),
        "confusion_matrix": confusion_matrix(test_metrics["labels"], test_metrics["preds"], num_classes),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Best val acc:", best_val_acc)
    print("Test acc:", test_metrics["acc"])
    print("Saved outputs to", args.output_dir)


if __name__ == "__main__":
    main()
