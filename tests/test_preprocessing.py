import numpy as np
import pandas as pd

from vehicle_behavior.data import extract_vehicle_labels, load_features_and_labels_from_csv
from vehicle_behavior.features import (
    extract_speed_position_features,
    prepare_speed_position_data,
    prepare_tabular_data,
)


def test_prepare_speed_position_shape():
    features = [
        {
            "speeds": np.arange(10, dtype=np.float32).reshape(-1, 1),
            "positions": np.column_stack(
                [np.arange(10, dtype=np.float32), np.zeros(10, dtype=np.float32)]
            ),
            "sequence_length": 10,
        }
    ]
    x_speed, x_pos = prepare_speed_position_data(features, max_sequence=60)
    assert x_speed.shape == (1, 60, 1)
    assert x_pos.shape == (1, 60, 2)


def test_tabular_features_shape():
    features = [
        {
            "speeds": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "positions": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            "sequence_length": 3,
        }
    ]
    tabular = prepare_tabular_data(features)
    assert tabular.shape == (1, 9)


def test_extract_labels_from_types():
    df = pd.DataFrame(
        {
            "VehTypeName": [
                "HDV Aggressive",
                "HDV Cooperative",
                "HDV Conventional Gipps Model",
                "CAV",
            ]
        }
    )
    labeled = extract_vehicle_labels(df)
    assert labeled["behavior_label"].tolist() == [
        "aggressive",
        "cooperative",
        "normal",
        "autonomous",
    ]


def test_load_sample_csv_if_present():
    try:
        features, labels = load_features_and_labels_from_csv(
            "data/sample/sample_trajectories.csv"
        )
    except FileNotFoundError:
        return
    assert len(features) > 0
    assert len(labels) == len(features)
    assert set(labels).issubset({"aggressive", "cooperative", "normal"})
