from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from vehicle_behavior.features import extract_speed_position_features


DEFAULT_COLUMN_MAP = {
    "vehicle_id": "VehNr",
    "timestep": "Timestep",
    "speed": "Speeds",
    "acceleration": "Accelerations",
    "front_coords": "VehFrontCoords",
    "rear_coords": "VehRearCoords",
    "vehicle_type": "VehTypeName",
    "length": "Length",
}


def extract_vehicle_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map Aimsun vehicle type names to behavior labels."""

    def map_vehicle_type(veh_type_name: Any) -> str:
        veh_type = str(veh_type_name).strip()
        if "CAV" in veh_type:
            return "autonomous"
        if "Aggressive" in veh_type:
            return "aggressive"
        if "Cooperative" in veh_type:
            return "cooperative"
        if "Conventional" in veh_type or "Gipps" in veh_type:
            return "normal"
        return "normal"

    labeled = df.copy()
    labeled["behavior_label"] = labeled["VehTypeName"].apply(map_vehicle_type)
    return labeled


def read_aimsun_data(
    file_path: str | Path,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame | None:
    """Read CSV with flexible column mapping."""
    col_map = column_map or DEFAULT_COLUMN_MAP
    try:
        df = pd.read_csv(file_path)
        return df.rename(columns={v: k for k, v in col_map.items()})
    except (OSError, ValueError) as exc:
        print(f"Error reading {file_path}: {exc}")
        return None


def load_features_and_labels_from_csv(
    file_path: str | Path,
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load one trajectory CSV and return feature dicts with labels."""
    df = pd.read_csv(file_path)
    df = extract_vehicle_labels(df)
    if max_rows is not None:
        df = df.head(max_rows)

    features: list[dict[str, Any]] = []
    labels: list[str] = []

    for _, row in df.iterrows():
        if row["behavior_label"] == "autonomous":
            continue
        extracted = extract_speed_position_features(row)
        if extracted is not None:
            features.append(extracted)
            labels.append(row["behavior_label"])

    return features, labels


def process_labeled_data_speed_position(
    csv_files: list[str],
    folder_path: str | Path,
    max_files: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Process multiple CSV files from a folder."""
    all_features: list[dict[str, Any]] = []
    all_labels: list[str] = []
    processed_count = 0

    files_to_process = csv_files[:max_files] if max_files else csv_files

    for filename in files_to_process:
        try:
            file_path = os.path.join(folder_path, filename)
            features, labels = load_features_and_labels_from_csv(file_path)
            all_features.extend(features)
            all_labels.extend(labels)
            processed_count += len(features)
            print(f"Processed {filename}: {len(features)} vehicles ({processed_count} total)")
        except (OSError, ValueError, KeyError) as exc:
            print(f"Error processing {filename}: {exc}")

    return all_features, all_labels
