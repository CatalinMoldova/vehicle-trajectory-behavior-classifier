from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from vehicle_behavior.config import load_config
from vehicle_behavior.data import extract_vehicle_labels, load_features_and_labels_from_csv
from vehicle_behavior.features import extract_speed_position_features
from vehicle_behavior.model import load_model, is_keras_model
from vehicle_behavior.features import prepare_speed_position_data, prepare_tabular_data


def run_prediction(
    input_path: str,
    model_path: str,
    config_path: str = "configs/default.yaml",
    output_path: str | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    model = load_model(model_path)
    label_encoder = joblib.load(Path(config.artifacts_dir) / "label_encoder.joblib")

    df = pd.read_csv(input_path)
    df = extract_vehicle_labels(df)

    rows: list[dict] = []
    for _, row in df.iterrows():
        features = extract_speed_position_features(row)
        if features is None:
            continue

        if is_keras_model(model):
            x_speed, x_pos = prepare_speed_position_data(
                [features], max_sequence=config.sequence_length
            )
            probs = model.predict([x_speed, x_pos], verbose=0)[0]
        else:
            x_tabular = prepare_tabular_data([features])
            probs = model.predict_proba(x_tabular)[0]

        pred_idx = int(probs.argmax())
        rows.append(
            {
                "VehNr": row.get("VehNr", row.get("vehicle_id")),
                "actual_label": row["behavior_label"],
                "predicted_label": label_encoder.inverse_transform([pred_idx])[0],
                "confidence": float(probs.max()),
            }
        )

    results = pd.DataFrame(rows)
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        print(results.to_string(index=False))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict behavior labels for trajectories")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument(
        "--model",
        default="artifacts/cnn_lstm.keras",
        help="Path to trained model",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()
    run_prediction(args.input, args.model, args.config, args.output)


if __name__ == "__main__":
    main()
