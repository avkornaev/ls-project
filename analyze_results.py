from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model

ALLOWED_SEEDS = [42, 0, 17]
EXPERIMENTS = ["baseline", "ls"]
TSNE_SAMPLE_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment results and plot.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Input results directory (contains runs/checkpoints).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for aggregated files and plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (cuda/cpu).",
    )
    return parser.parse_args()


def get_device(device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all_run_metrics(results_dir: Path) -> pd.DataFrame:
    all_dfs: List[pd.DataFrame] = []
    for exp in EXPERIMENTS:
        for seed in ALLOWED_SEEDS:
            csv_path = results_dir / "runs" / f"{exp}_seed{seed}" / "metrics.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing metrics file: {csv_path}")
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def aggregate_epoch_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df.groupby(["experiment", "epoch"], as_index=False).agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
        val_loss_mean=("val_loss", "mean"),
        val_loss_std=("val_loss", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    )


def summarize_final_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    final_epoch = metrics_df["epoch"].max()
    final_df = metrics_df[metrics_df["epoch"] == final_epoch]
    summary = final_df.groupby("experiment", as_index=False).agg(
        final_val_loss_mean=("val_loss", "mean"),
        final_val_loss_std=("val_loss", "std"),
        final_accuracy_mean=("accuracy", "mean"),
        final_accuracy_std=("accuracy", "std"),
    )

    best_rows = (
        metrics_df.sort_values("val_loss")
        .groupby(["experiment", "seed"], as_index=False)
        .first()[["experiment", "seed", "epoch", "val_loss", "accuracy"]]
    )
    best_summary = best_rows.groupby("experiment", as_index=False).agg(
        best_val_loss_mean=("val_loss", "mean"),
        best_val_loss_std=("val_loss", "std"),
        best_accuracy_mean=("accuracy", "mean"),
        best_accuracy_std=("accuracy", "std"),
    )
    return summary.merge(best_summary, on="experiment", how="inner")


def plot_loss_curves(agg_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for exp in EXPERIMENTS:
        subset = agg_df[agg_df["experiment"] == exp]
        plt.plot(subset["epoch"], subset["val_loss_mean"], label=f"{exp} val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss (Average Across Seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "validation_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for exp in EXPERIMENTS:
        subset = agg_df[agg_df["experiment"] == exp]
        plt.plot(subset["epoch"], subset["train_loss_mean"], label=f"{exp} train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss (Average Across Seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "train_loss.png", dpi=180)
    plt.close()


def select_best_checkpoint(results_dir: Path, experiment: str) -> Path:
    ckpt_dir = results_dir / "checkpoints"
    best_ckpt = None
    best_val = float("inf")
    for seed in ALLOWED_SEEDS:
        ckpt_path = ckpt_dir / f"{experiment}_seed{seed}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        val = float(state.get("best_val_loss", float("inf")))
        if val < best_val:
            best_val = val
            best_ckpt = ckpt_path
    if best_ckpt is None:
        raise RuntimeError(f"Could not find a best checkpoint for experiment={experiment}")
    return best_ckpt


def get_eval_subset() -> Subset:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=eval_transform)
    if TSNE_SAMPLE_SIZE > len(dataset):
        raise ValueError("TSNE_SAMPLE_SIZE is larger than test dataset size.")
    generator = torch.Generator().manual_seed(12345)
    indices = torch.randperm(len(dataset), generator=generator)[:TSNE_SAMPLE_SIZE].tolist()
    return Subset(dataset, indices)


@torch.no_grad()
def run_tsne(
    results_dir: Path,
    output_dir: Path,
    experiment: str,
    device: torch.device,
) -> None:
    ckpt_path = select_best_checkpoint(results_dir, experiment)
    model = build_model(num_classes=10).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    subset = get_eval_subset()
    loader = DataLoader(
        subset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats = model.extract_features(images)
        all_features.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    embedding = TSNE(
        n_components=2,
        random_state=0,
        init="pca",
        learning_rate="auto",
        perplexity=30,
    ).fit_transform(features)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10", 10)
    for class_idx in range(10):
        mask = labels == class_idx
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=8,
            alpha=0.7,
            color=cmap(class_idx),
            label=str(class_idx),
        )
    plt.title(f"t-SNE ({experiment}) - best validation checkpoint")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=2, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / f"tsne_{experiment}.png", dpi=180)
    plt.close()


def write_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_rows: Dict[str, Dict[str, float]] = {}
    for _, row in summary_df.iterrows():
        summary_rows[row["experiment"]] = {
            "final_val_loss_mean": float(row["final_val_loss_mean"]),
            "final_val_loss_std": float(row["final_val_loss_std"]),
            "final_accuracy_mean": float(row["final_accuracy_mean"]),
            "final_accuracy_std": float(row["final_accuracy_std"]),
            "best_val_loss_mean": float(row["best_val_loss_mean"]),
            "best_val_loss_std": float(row["best_val_loss_std"]),
            "best_accuracy_mean": float(row["best_accuracy_mean"]),
            "best_accuracy_std": float(row["best_accuracy_std"]),
        }

    baseline = summary_rows.get("baseline")
    ls = summary_rows.get("ls")
    if baseline is None or ls is None:
        raise ValueError("Summary is missing baseline or ls rows.")

    lines = [
        "# Label Smoothing Study Report",
        "",
        "## Methods",
        "- Dataset: CIFAR-10",
        "- Model: ResNet18 adapted for CIFAR-10 (3x3 stem, no maxpool)",
        "- Experiments: baseline CE vs CE with label smoothing (epsilon=0.1)",
        "- Seeds: 42, 0, 17",
        "- Epochs: 20",
        "",
        "## Quantitative Results",
        (
            f"- Baseline final val loss: {baseline['final_val_loss_mean']:.4f} "
            f"+/- {baseline['final_val_loss_std']:.4f}"
        ),
        (
            f"- Label smoothing final val loss: {ls['final_val_loss_mean']:.4f} "
            f"+/- {ls['final_val_loss_std']:.4f}"
        ),
        (
            f"- Baseline final accuracy: {baseline['final_accuracy_mean']:.4f} "
            f"+/- {baseline['final_accuracy_std']:.4f}"
        ),
        (
            f"- Label smoothing final accuracy: {ls['final_accuracy_mean']:.4f} "
            f"+/- {ls['final_accuracy_std']:.4f}"
        ),
        "",
        "## Figures",
        "- results/validation_loss.png",
        "- results/train_loss.png",
        "- results/tsne_baseline.png",
        "- results/tsne_ls.png",
        "",
        "## Limitations",
        "- Single dataset and architecture.",
        "- Runtime and convergence can vary by hardware.",
        "- Results should be interpreted as a controlled educational study.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    metrics_df = load_all_run_metrics(args.results_dir)
    agg_df = aggregate_epoch_metrics(metrics_df)
    summary_df = summarize_final_metrics(metrics_df)

    metrics_df.to_csv(args.output_dir / "all_run_metrics.csv", index=False)
    agg_df.to_csv(args.output_dir / "aggregated_epoch_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary_metrics.csv", index=False)

    plot_loss_curves(agg_df, args.output_dir)
    run_tsne(args.results_dir, args.output_dir, "baseline", device=device)
    run_tsne(args.results_dir, args.output_dir, "ls", device=device)

    report_path = Path("report.md")
    write_report(summary_df, report_path)

    metadata = {
        "experiments": EXPERIMENTS,
        "seeds": ALLOWED_SEEDS,
        "tsne_sample_size": TSNE_SAMPLE_SIZE,
        "results_dir": str(args.results_dir),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Analysis complete. Plots, summaries, and report generated.")


if __name__ == "__main__":
    main(parse_args())
