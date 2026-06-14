# Data

## Synthetic sample (included)

`sample/sample_trajectories.csv` is a small **synthetic** dataset (36 trajectories)
so the pipeline can run end-to-end without credentials. It is intentionally trivial —
classes are separable by mean speed — so metrics on it are a smoke test, **not** a benchmark.

Regenerate with:

```bash
python scripts/generate_sample_data.py
```

## Real Aimsun dataset (private, not included)

The original Aimsun micro-simulation data is **private NYU research data** and is **not**
committed to this repository (it is large and confidential). It is gitignored.

To run on the real data locally, place the unzipped `Time_Series_data/` folder in the
project root:

```text
Time_Series_data/
├── scen7_8rep4/
│   ├── interval_0000_0060.csv
│   ├── interval_0060_0120.csv
│   └── ...
├── scen8_2rep4/
└── ...
```

Each CSV is a 60-second interval with columns including
`VehNr, VehTypeName, Speeds, VehFrontCoords, ...`. Vehicle types map to behavior labels:

| `VehTypeName` | Label |
|---|---|
| `HDV Aggressive` | aggressive |
| `HDV Cooperative` | cooperative |
| `HDV Conventional Gipps Model` | normal |
| `CAV` | autonomous (excluded from training) |

Then run the leakage-safe evaluation (splits by vehicle, scales on train only):

```bash
python scripts/run_on_real_data.py --files-per-scenario 4 --max-samples 4000 --epochs 15
```

### Results on real data (786 test trajectories, grouped split, no vehicle leakage)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.513 | 0.227 | 0.348 |
| Random Forest | 0.455 | 0.345 | 0.421 |
| LSTM | 0.513 | 0.226 | 0.348 |
| CNN-LSTM | 0.525 | 0.309 | 0.396 |

The majority class (`aggressive`) is ~51% of the test set, so accuracy near 0.51 means a
model is close to the majority baseline. Macro F1 is the more honest metric here. These
numbers reflect a genuinely hard task and leave clear headroom for class balancing,
longer training, and richer features.
