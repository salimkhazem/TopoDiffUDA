"""Plotting utilities for figures."""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_curve(precision: Sequence[float], recall: Sequence[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reliability_diagram(confidences: Sequence[float], accuracies: Sequence[float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.plot(confidences, accuracies, marker="o", color="black")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bar(values: Dict[str, float], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(values.keys())
    vals = list(values.values())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_qualitative_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    output_path: Path,
    n_cols: int = 5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    for idx, (img, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_reliability_bins(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> Tuple[List[float], List[float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = []
    bin_acc = []
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf.append(float(confidences[mask].mean()))
        bin_acc.append(float(accuracies[mask].mean()))
    return bin_conf, bin_acc
