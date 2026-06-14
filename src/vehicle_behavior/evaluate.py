from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from vehicle_behavior.features import prepare_speed_position_data, prepare_tabular_data
from vehicle_behavior.model import is_keras_model


def predict_labels(
    model: object,
    features_list: list[dict[str, Any]],
    label_encoder: Any,
    sequence_length: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer predictions and class probabilities/confidence."""
    y_true_names = None

    if is_keras_model(model):
        x_speed, x_pos = prepare_speed_position_data(
            features_list, max_sequence=sequence_length
        )
        probabilities = model.predict([x_speed, x_pos], verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        return predictions, confidence

    x_tabular = prepare_tabular_data(features_list)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_tabular)
        predictions = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
    else:
        predictions = model.predict(x_tabular)
        confidence = np.ones(len(predictions), dtype=np.float32)

    del y_true_names
    return predictions.astype(int), confidence.astype(np.float32)


def evaluate_model(
    model: object,
    features_list: list[dict[str, Any]],
    labels: list[str],
    label_encoder: Any,
    sequence_length: int = 60,
) -> dict[str, Any]:
    """Compute metrics for a trained model."""
    y_true = label_encoder.transform(labels)
    y_pred, confidence = predict_labels(
        model, features_list, label_encoder, sequence_length=sequence_length
    )

    class_names = list(label_encoder.classes_)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "predictions": label_encoder.inverse_transform(y_pred).tolist(),
        "actual": labels,
        "confidence": confidence.tolist(),
    }


def save_evaluation_results(
    metrics: dict[str, Any],
    results_dir: str | Path,
    model_name: str,
) -> None:
    """Persist metrics, report, and confusion matrix."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
            }
        ]
    )
    summary_path = out / "metrics.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing["model"] != model_name]
        summary = pd.concat([existing, summary], ignore_index=True)
    summary.to_csv(summary_path, index=False)

    with open(out / "classification_report.json", "w", encoding="utf-8") as handle:
        json.dump(metrics["classification_report"], handle, indent=2)

    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["class_names"],
        columns=metrics["class_names"],
    )
    cm_df.to_csv(out / "confusion_matrix.csv")


def main() -> None:
    import argparse
    import joblib

    from vehicle_behavior.config import load_config
    from vehicle_behavior.data import load_features_and_labels_from_csv
    from vehicle_behavior.model import load_model

    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", required=True, help="Path to .keras or .joblib model")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(args.model)
    label_encoder = joblib.load(
        Path(config.artifacts_dir) / "label_encoder.joblib"
    )

    features, labels = load_features_and_labels_from_csv(config.sample_path)
    metrics = evaluate_model(
        model,
        features,
        labels,
        label_encoder,
        sequence_length=config.sequence_length,
    )
    model_name = Path(args.model).stem
    save_evaluation_results(metrics, config.results_dir, model_name)

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 macro: {metrics['f1_macro']:.3f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.3f}")


if __name__ == "__main__":
    main()
