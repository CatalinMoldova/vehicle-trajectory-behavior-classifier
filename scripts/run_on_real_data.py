"""Run the pipeline on the real Aimsun Time_Series_data with a leakage-safe split.

- Groups by (scenario, VehNr) so the same vehicle never appears in both
  train and test.
- Scales features on TRAIN ONLY (fixes the missing-normalization issue).
- Trains all four models and prints a comparison table.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from vehicle_behavior.data import extract_vehicle_labels
from vehicle_behavior.features import (
    extract_speed_position_features,
    prepare_speed_position_data,
    prepare_tabular_data,
)
from vehicle_behavior.model import build_cnn_lstm_model, build_lstm_model

DATA_ROOT = "Time_Series_data"


def collect(files_per_scenario: int, max_samples: int):
    features, labels, groups = [], [], []
    for scen_dir in sorted(glob.glob(f"{DATA_ROOT}/*/")):
        scen = os.path.basename(scen_dir.rstrip("/"))
        csvs = sorted(glob.glob(scen_dir + "*.csv"))[:files_per_scenario]
        for f in csvs:
            df = extract_vehicle_labels(pd.read_csv(f))
            for _, row in df.iterrows():
                if row["behavior_label"] == "autonomous":
                    continue
                feat = extract_speed_position_features(row)
                if feat is None:
                    continue
                features.append(feat)
                labels.append(row["behavior_label"])
                groups.append(f"{scen}:{row['VehNr']}")
    if len(features) > max_samples:
        idx = np.random.RandomState(42).choice(len(features), max_samples, replace=False)
        features = [features[i] for i in idx]
        labels = [labels[i] for i in idx]
        groups = [groups[i] for i in idx]
    return features, labels, groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-per-scenario", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seq-len", type=int, default=60)
    args = ap.parse_args()

    print("Loading real data...")
    features, labels, groups = collect(args.files_per_scenario, args.max_samples)
    print(f"Samples: {len(features)} | classes: {dict(Counter(labels))}")
    print(f"Unique vehicles (groups): {len(set(groups))}")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    groups = np.array(groups)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(features, y, groups))
    tr_f = [features[i] for i in train_idx]
    te_f = [features[i] for i in test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"Train: {len(tr_f)} | Test: {len(te_f)} (split by vehicle, no leakage)")
    print(f"Test class balance: {dict(Counter(le.inverse_transform(y_te)))}\n")

    rows = []

    # --- classical baselines (train-only scaling baked into Pipeline) ---
    X_tr_tab = prepare_tabular_data(tr_f)
    X_te_tab = prepare_tabular_data(te_f)
    for name, clf in [
        ("logistic_regression", Pipeline([("s", StandardScaler()),
                                          ("c", LogisticRegression(max_iter=1000))])),
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]:
        clf.fit(X_tr_tab, y_tr)
        pred = clf.predict(X_te_tab)
        rows.append((name, accuracy_score(y_te, pred),
                     f1_score(y_te, pred, average="macro"),
                     f1_score(y_te, pred, average="weighted")))

    # --- deep models with TRAIN-ONLY standardization of sequences ---
    Xs_tr, Xp_tr = prepare_speed_position_data(tr_f, args.seq_len)
    Xs_te, Xp_te = prepare_speed_position_data(te_f, args.seq_len)

    def scale(train, test):
        m = train.reshape(-1, train.shape[-1]).mean(0)
        s = train.reshape(-1, train.shape[-1]).std(0) + 1e-6
        return (train - m) / s, (test - m) / s

    Xs_tr, Xs_te = scale(Xs_tr, Xs_te)
    Xp_tr, Xp_te = scale(Xp_tr, Xp_te)

    for name, builder in [("lstm", build_lstm_model), ("cnn_lstm", build_cnn_lstm_model)]:
        model = builder(sequence_length=args.seq_len, num_classes=len(le.classes_))
        model.fit([Xs_tr, Xp_tr], y_tr, epochs=args.epochs, batch_size=32, verbose=0)
        prob = model.predict([Xs_te, Xp_te], verbose=0)
        pred = prob.argmax(1)
        rows.append((name, accuracy_score(y_te, pred),
                     f1_score(y_te, pred, average="macro"),
                     f1_score(y_te, pred, average="weighted")))

    print(f"\n{'model':22s} {'acc':>6s} {'f1_macro':>9s} {'f1_weighted':>12s}")
    print("-" * 52)
    for name, acc, fm, fw in rows:
        print(f"{name:22s} {acc:6.3f} {fm:9.3f} {fw:12.3f}")


if __name__ == "__main__":
    main()
