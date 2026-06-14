from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    sequence_length: int = 60
    num_classes: int = 3
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    model: str = "cnn_lstm"
    sample_path: str = "data/sample/sample_trajectories.csv"
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data_section = data.get("data", {})
        return cls(
            sequence_length=data.get("sequence_length", 60),
            num_classes=data.get("num_classes", 3),
            test_size=data.get("test_size", 0.2),
            random_state=data.get("random_state", 42),
            batch_size=data.get("batch_size", 32),
            epochs=data.get("epochs", 20),
            learning_rate=data.get("learning_rate", 0.001),
            model=data.get("model", "cnn_lstm"),
            sample_path=data_section.get(
                "sample_path", "data/sample/sample_trajectories.csv"
            ),
            artifacts_dir=data.get("artifacts_dir", "artifacts"),
            results_dir=data.get("results_dir", "results"),
        )


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Config.from_dict(data)
