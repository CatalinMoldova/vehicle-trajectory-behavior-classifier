from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd


def _safe_eval_list(value: str) -> list[Any]:
    cleaned = (
        str(value)
        .replace("inf", "0")
        .replace("-inf", "0")
        .replace("nan", "None")
    )
    try:
        return ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        return []


def parse_array_string(array_str: Any) -> list[Any]:
    """Parse array strings with error handling."""
    try:
        if pd.isna(array_str):
            return []
        return _safe_eval_list(array_str)
    except (TypeError, ValueError):
        return []


def interpolate_missing_values(sequence: list[Any]) -> list[float]:
    """Interpolate only None or np.nan values; leave infinities unchanged."""
    if not sequence:
        return []

    arr = np.array(sequence, dtype=float)
    missing_mask = np.isnan(arr)
    valid_mask = ~missing_mask & np.isfinite(arr)
    valid_indices = np.where(valid_mask)[0]

    if np.all(missing_mask | ~np.isfinite(arr)):
        return [float(v) if np.isinf(v) else 0.0 for v in arr]

    if len(valid_indices) == 1:
        fill_value = arr[valid_indices[0]]
        return [float(v) if not np.isnan(v) else float(fill_value) for v in arr]

    interp_values = np.copy(arr)
    interp_indices = np.where(missing_mask & ~np.isinf(arr))[0]
    if len(valid_indices) >= 2 and len(interp_indices) > 0:
        interp_result = np.interp(
            interp_indices, valid_indices, arr[valid_mask]
        )
        interp_values[interp_indices] = interp_result

    return interp_values.tolist()


def parse_coordinate_string(coord_str: Any) -> list[tuple[float, float]]:
    """Parse coordinate data with interpolation."""
    try:
        if pd.isna(coord_str):
            return []
        coords = _safe_eval_list(coord_str)
        if not coords:
            return []
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        return list(
            zip(
                interpolate_missing_values(x_coords),
                interpolate_missing_values(y_coords),
            )
        )
    except (TypeError, ValueError, IndexError):
        return []


def calculate_lateral_speeds(
    front_coords: list[tuple[float, float]],
    rear_coords: list[tuple[float, float]] | None = None,
) -> list[float]:
    """Calculate lateral movement speeds from consecutive coordinates."""
    del rear_coords  # kept for notebook API compatibility
    if len(front_coords) < 2:
        return []

    lateral_speeds: list[float] = []
    for i in range(1, len(front_coords)):
        try:
            dx = front_coords[i][0] - front_coords[i - 1][0]
            dy = front_coords[i][1] - front_coords[i - 1][1]
            lateral_speeds.append(float(np.sqrt(dx**2 + dy**2)))
        except (TypeError, ValueError):
            lateral_speeds.append(0.0)
    return interpolate_missing_values(lateral_speeds)


def extract_speed_position_features(
    vehicle_row: pd.Series,
    min_sequence_length: int = 5,
) -> dict[str, Any] | None:
    """Extract speed and position sequences from one vehicle row."""
    try:
        speeds = parse_array_string(vehicle_row["Speeds"])
        front_coords = parse_coordinate_string(vehicle_row["VehFrontCoords"])

        def count_invalid(seq: list[Any]) -> int:
            return sum(
                (v is None)
                or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
                for v in seq
            )

        if count_invalid(speeds) >= 30 or count_invalid(front_coords) >= 30:
            return None

        speeds_clean = interpolate_missing_values(speeds)
        positions_clean = front_coords

        min_length = min(len(speeds_clean), len(positions_clean))
        if min_length < min_sequence_length:
            return None

        speeds_clean = speeds_clean[:min_length]
        positions_clean = positions_clean[:min_length]

        speeds_arr = np.array(speeds_clean, dtype=np.float32).reshape(-1, 1)
        positions_arr = np.array(positions_clean, dtype=np.float32).reshape(-1, 2)

        return {
            "speeds": speeds_arr,
            "positions": positions_arr,
            "sequence_length": min_length,
        }
    except (KeyError, TypeError, ValueError):
        return None


def prepare_speed_position_data(
    features_list: list[dict[str, Any]],
    max_sequence: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad speed and position arrays for model input."""
    x_speed: list[np.ndarray] = []
    x_pos: list[np.ndarray] = []

    for features in features_list:
        speeds = features["speeds"][:max_sequence]
        positions = features["positions"][:max_sequence]
        seq_len = min(len(speeds), max_sequence)

        speed_padded = np.zeros((max_sequence, 1), dtype=np.float32)
        pos_padded = np.zeros((max_sequence, 2), dtype=np.float32)
        speed_padded[:seq_len] = speeds[:seq_len]
        pos_padded[:seq_len] = positions[:seq_len]

        x_speed.append(speed_padded)
        x_pos.append(pos_padded)

    return np.array(x_speed, dtype=np.float32), np.array(x_pos, dtype=np.float32)


def extract_tabular_features(features: dict[str, Any]) -> np.ndarray:
    """Engineered summary features for classical baselines."""
    speeds = features["speeds"].flatten()
    positions = features["positions"]

    if len(speeds) > 1:
        accelerations = np.diff(speeds)
    else:
        accelerations = np.array([0.0], dtype=np.float32)

    if len(positions) > 1:
        displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        path_length = float(np.sum(displacements))
    else:
        path_length = 0.0

    return np.array(
        [
            float(np.mean(speeds)),
            float(np.std(speeds)),
            float(np.max(speeds)),
            float(np.min(speeds)),
            float(np.mean(accelerations)),
            float(np.std(accelerations)),
            float(np.max(np.abs(accelerations))),
            path_length,
            float(features["sequence_length"]),
        ],
        dtype=np.float32,
    )


def prepare_tabular_data(
    features_list: list[dict[str, Any]],
) -> np.ndarray:
    """Stack tabular features for sklearn baselines."""
    return np.vstack([extract_tabular_features(item) for item in features_list])
