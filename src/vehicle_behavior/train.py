from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from vehicle_behavior.config import load_config
from vehicle_behavior.data import load_features_and_labels_from_csv
from vehicle_behavior.evaluate import evaluate_model, save_evaluation_results
from vehicle_behavior.features import prepare_speed_position_data, prepare_tabular_data
from vehicle_behavior.model import (
    build_cnn_lstm_model,
    build_logistic_regression_baseline,
    build_lstm_model,
    build_random_forest_baseline,
    is_keras_model,
    save_artifacts,
)


SUPPORTED_MODELS = {
    "cnn_lstm": build_cnn_lstm_model,
    "lstm": build_lstm_model,
    "logistic_regression": build_logistic_regression_baseline,
    "random_forest": build_random_forest_baseline,
}


def train_model(
    model_name: str,
    train_features: list,
    train_labels: list[str],
    config,
) -> object:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    if model_name in {"cnn_lstm", "lstm"}:
        builder = SUPPORTED_MODELS[model_name]
        model = builder(
            sequence_length=config.sequence_length,
            num_classes=config.num_classes,
            learning_rate=config.learning_rate,
        )
        x_speed, x_pos = prepare_speed_position_data(
            train_features, max_sequence=config.sequence_length
        )
        model.fit(
            [x_speed, x_pos],
            y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=1,
        )
        model._label_encoder = label_encoder  # type: ignore[attr-defined]
        return model

    if model_name == "logistic_regression":
        model = build_logistic_regression_baseline()
    else:
        model = build_random_forest_baseline()

    x_tabular = prepare_tabular_data(train_features)
    model.fit(x_tabular, y_train)
    model._label_encoder = label_encoder  # type: ignore[attr-defined]
    return model


def run_training(config_path: str, model_name: str | None = None) -> str:
    config = load_config(config_path)
    selected_model = model_name or config.model
    if selected_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{selected_model}'. "
            f"Choose from: {', '.join(SUPPORTED_MODELS)}"
        )

    features, labels = load_features_and_labels_from_csv(config.sample_path)
    if len(features) < 6:
        raise ValueError(
            "Need at least 6 labeled trajectories to train. "
            "Regenerate sample data with scripts/generate_sample_data.py"
        )

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    model = train_model(selected_model, train_features, train_labels, config)
    label_encoder = getattr(model, "_label_encoder", None)
    if label_encoder is None:
        label_encoder = LabelEncoder().fit(train_labels)

    model_path = save_artifacts(
        model,
        label_encoder,
        config.artifacts_dir,
        selected_model,
    )

    metrics = evaluate_model(
        model,
        test_features,
        test_labels,
        label_encoder,
        sequence_length=config.sequence_length,
    )
    save_evaluation_results(metrics, config.results_dir, selected_model)

    print(f"Model saved to {model_path}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 macro: {metrics['f1_macro']:.3f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.3f}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vehicle behavior classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--model",
        choices=list(SUPPORTED_MODELS),
        default=None,
        help="Override model from config",
    )
    args = parser.parse_args()
    run_training(args.config, args.model)


if __name__ == "__main__":
    main()
