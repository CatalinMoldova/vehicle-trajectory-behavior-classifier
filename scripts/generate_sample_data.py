"""Generate synthetic Aimsun-style trajectory rows for local development."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _make_trajectory(
    veh_type: str,
    veh_nr: int,
    rng: np.random.Generator,
    n_steps: int,
) -> dict:
    if "Aggressive" in veh_type:
        speeds = np.cumsum(rng.normal(0.8, 0.5, n_steps)) + rng.uniform(12, 22)
        x = np.cumsum(rng.normal(1.8, 0.6, n_steps))
        y = np.cumsum(rng.normal(0.0, 0.4, n_steps))
    elif "Cooperative" in veh_type:
        speeds = np.cumsum(rng.normal(0.2, 0.2, n_steps)) + rng.uniform(8, 14)
        x = np.cumsum(rng.normal(1.0, 0.2, n_steps))
        y = np.cumsum(rng.normal(0.0, 0.1, n_steps))
    else:
        speeds = np.cumsum(rng.normal(0.4, 0.3, n_steps)) + rng.uniform(10, 18)
        x = np.cumsum(rng.normal(1.2, 0.3, n_steps))
        y = np.cumsum(rng.normal(0.0, 0.15, n_steps))

    speeds = np.clip(speeds, 0, 35)
    coord_pairs = [(round(float(xi), 2), round(float(yi), 2)) for xi, yi in zip(x, y)]
    return {
        "VehNr": veh_nr,
        "Timestep": n_steps,
        "Speeds": str([round(float(s), 2) for s in speeds]),
        "VehFrontCoords": str(coord_pairs),
        "VehTypeName": veh_type,
    }


def main() -> None:
    rng = np.random.default_rng(42)
    types = [
        "HDV Aggressive",
        "HDV Cooperative",
        "HDV Conventional Gipps Model",
    ]
    rows = []
    veh_nr = 1
    for veh_type in types:
        for _ in range(12):
            n_steps = int(rng.integers(25, 45))
            rows.append(_make_trajectory(veh_type, veh_nr, rng, n_steps))
            veh_nr += 1

    out = Path("data/sample/sample_trajectories.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
