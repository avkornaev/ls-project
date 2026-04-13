from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from loss import get_loss
from model import build_model

ALLOWED_SEEDS = [42, 0, 17]
NUM_EPOCHS = 20
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_MILESTONES = [10, 15]
LR_GAMMA = 0.1
NUM_CLASSES = 10
DEFAULT_EPSILON = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 ResNet18 experiment.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "ls"],
        help="Experiment mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Single run seed. Must be one of 42, 0, 17.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Results root directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (cuda/cpu).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(seed: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root="data", train=False, download=True, transform=eval_transform
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: SGD | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def ensure_dirs(output_dir: Path, mode: str, seed: int) -> Dict[str, Path]:
    run_dir = output_dir / "runs" / f"{mode}_seed{seed}"
    ckpt_dir = output_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "ckpt": ckpt_dir / f"{mode}_seed{seed}_best.pt",
        "metrics_csv": run_dir / "metrics.csv",
        "metrics_jsonl": run_dir / "metrics.jsonl",
        "config": run_dir / "config.json",
    }


def save_config(path: Path, config: Dict[str, object]) -> None:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def save_seed_manifest(output_dir: Path) -> None:
    manifest = {
        "allowed_seeds": ALLOWED_SEEDS,
        "experiments": ["baseline", "ls"],
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
    }
    (output_dir / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def append_metrics(
    csv_path: Path,
    jsonl_path: Path,
    rows: List[Dict[str, object]],
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "seed", "epoch", "train_loss", "val_loss", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def train_once(args: argparse.Namespace) -> None:
    if args.seed not in ALLOWED_SEEDS:
        raise ValueError(f"Seed {args.seed} is not allowed. Use one of {ALLOWED_SEEDS}.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_seed_manifest(output_dir)

    set_seed(args.seed)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(args.seed, args.num_workers)

    model = build_model(num_classes=NUM_CLASSES).to(device)
    criterion = get_loss(args.mode, epsilon=DEFAULT_EPSILON).to(device)
    optimizer = SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

    paths = ensure_dirs(output_dir, args.mode, args.seed)
    config = {
        "experiment": args.mode,
        "seed": args.seed,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "SGD",
        "lr": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "lr_schedule": {
            "name": "MultiStepLR",
            "milestones": LR_MILESTONES,
            "gamma": LR_GAMMA,
        },
        "loss": {
            "name": "CrossEntropyLoss",
            "label_smoothing": DEFAULT_EPSILON if args.mode == "ls" else 0.0,
        },
        "augmentations": ["RandomCrop(32,padding=4)", "RandomHorizontalFlip()"],
        "device": str(device),
    }
    save_config(paths["config"], config)

    history: List[Dict[str, object]] = []
    best_val = float("inf")
    start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        row = {
            "experiment": args.mode,
            "seed": args.seed,
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "accuracy": round(float(val_acc), 6),
        }
        history.append(row)
        print(
            f"[{args.mode}][seed={args.seed}] epoch={epoch:02d}/{NUM_EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_loss": float(best_val),
                    "epoch": epoch,
                },
                paths["ckpt"],
            )

    append_metrics(paths["metrics_csv"], paths["metrics_jsonl"], history)
    elapsed = time.time() - start
    print(
        f"Completed mode={args.mode}, seed={args.seed}. "
        f"Best val_loss={best_val:.4f}. Runtime={elapsed:.1f}s"
    )


if __name__ == "__main__":
    train_once(parse_args())
