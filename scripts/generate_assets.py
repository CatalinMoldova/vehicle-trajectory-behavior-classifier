"""Generate README assets: architecture, trajectories, confusion matrix, training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from vehicle_behavior.config import load_config
from vehicle_behavior.data import extract_vehicle_labels, load_features_and_labels_from_csv
from vehicle_behavior.features import parse_array_string, parse_coordinate_string, prepare_speed_position_data
from vehicle_behavior.model import build_cnn_lstm_model


def plot_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("CNN-LSTM Architecture", fontsize=14, fontweight="bold")

    boxes = [
        (0.5, 5.5, "Speed\n(60×1)"),
        (0.5, 3.5, "Position\n(60×2)"),
        (3.0, 6.5, "Acceleration"),
        (3.0, 4.5, "Lateral\nmovement"),
        (5.5, 5.5, "Feature\nfusion\n(60×3)"),
        (8.0, 6.5, "CNN branch\nConv1D + Pool"),
        (8.0, 4.5, "LSTM branch\n128 → 64"),
        (10.0, 5.5, "Dense +\nSoftmax\n(3 classes)"),
    ]
    for x, y, text in boxes:
        patch = FancyBboxPatch(
            (x, y), 1.8, 1.0, boxstyle="round,pad=0.05",
            linewidth=1.2, edgecolor="#2c3e50", facecolor="#ecf0f1",
        )
        ax.add_patch(patch)
        ax.text(x + 0.9, y + 0.5, text, ha="center", va="center", fontsize=9)

    arrows = [
        ((2.3, 6.0), (3.0, 6.5)), ((2.3, 4.0), (3.0, 4.5)),
        ((4.8, 6.5), (5.5, 6.0)), ((4.8, 4.5), (5.5, 5.0)),
        ((7.3, 6.0), (8.0, 6.5)), ((7.3, 5.0), (8.0, 4.5)),
        ((9.8, 6.5), (10.0, 6.0)), ((9.8, 4.5), (10.0, 5.0)),
        ((11.8, 5.5), (11.8, 5.5)),
    ]
    for start, end in arrows[:-1]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, color="#34495e"))

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_examples(csv_path: Path, out: Path) -> None:
    df = extract_vehicle_labels(pd.read_csv(csv_path))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ["aggressive", "cooperative", "normal"]

    for ax, label in zip(axes, labels):
        row = df[df["behavior_label"] == label].iloc[0]
        speeds = parse_array_string(row["Speeds"])
        coords = parse_coordinate_string(row["VehFrontCoords"])
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax2 = ax.twinx()
        ax.plot(speeds, color="#e74c3c", label="speed")
        ax2.plot(xs, color="#3498db", alpha=0.7, label="x-position")
        ax.set_title(label.capitalize())
        ax.set_xlabel("timestep")
        ax.set_ylabel("speed")
        ax2.set_ylabel("x-position")

    fig.suptitle("Sample Trajectory Examples (synthetic data)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm_path: Path, out: Path) -> None:
    cm = pd.read_csv(cm_path, index_col=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (CNN-LSTM, synthetic sample)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(config_path: str, out: Path) -> None:
    config = load_config(config_path)
    features, labels = load_features_and_labels_from_csv(config.sample_path)
    train_features, _, train_labels, _ = train_test_split(
        features, labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    model = build_cnn_lstm_model(
        sequence_length=config.sequence_length,
        num_classes=config.num_classes,
        learning_rate=config.learning_rate,
    )
    x_speed, x_pos = prepare_speed_position_data(train_features, config.sequence_length)
    history = model.fit(
        [x_speed, x_pos], y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
    )

    hist = history.history
    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "cnn_lstm_history.json", "w", encoding="utf-8") as handle:
        json.dump(hist, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist["loss"], marker="o", color="#e74c3c")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(hist["accuracy"], marker="o", color="#27ae60")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    fig.suptitle("CNN-LSTM Training Curves (synthetic sample)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs" / "default.yaml"
    sample_path = root / "data" / "sample" / "sample_trajectories.csv"
    cm_path = root / "results" / "confusion_matrix.csv"

    plot_architecture(assets / "architecture.png")
    plot_trajectory_examples(sample_path, assets / "trajectory_examples.png")
    plot_confusion_matrix(cm_path, assets / "confusion_matrix.png")
    plot_training_curves(str(config_path), assets / "training_curves.png")
    print(f"Wrote assets to {assets}")


if __name__ == "__main__":
    main()
