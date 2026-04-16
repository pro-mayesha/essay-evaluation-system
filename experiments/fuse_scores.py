"""Merge prediction exports with availability features and normalize heuristics.

Normalization: z-score (standard scaling) fitted on the validation split only,
then applied to both validation and test rows. This avoids using test-set
statistics when scaling features (reduces leakage for held-out evaluation).
true_score and deberta_pred are left unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

VAL_PRED = Path("results_asap_debug/test_eval_results/val_predictions_with_ids.csv")
TEST_PRED = Path("results_asap_debug/test_eval_results/test_predictions_with_ids.csv")
FEATURES = Path("experiments/outputs/availability_features.csv")
OUT = Path("experiments/outputs/merged_experiment_data.csv")
RESULTS_OUT = Path("experiments/outputs/fusion_results.json")

HEURISTIC_COLS = [
    "concreteness",
    "specificity",
    "emotional_salience",
    "personal_experience_salience",
    "narrative_event_density",
]

OUTPUT_COLS = [
    "essay_id",
    "split",
    "text",
    "true_score",
    "deberta_pred",
    *HEURISTIC_COLS,
]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = float("nan")
    return {"rmse": rmse, "mae": mae, "pearson_r": pearson_r}


def qwk_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # ASAP targets are ordinal-like score levels, so QWK is computed on rounded scores.
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    lo = int(np.floor(np.min(y_true)))
    hi = int(np.ceil(np.max(y_true)))
    y_true_i = np.clip(np.round(y_true).astype(int), lo, hi)
    y_pred_i = np.clip(np.round(y_pred).astype(int), lo, hi)
    return float(cohen_kappa_score(y_true_i, y_pred_i, weights="quadratic"))


def main() -> None:
    val = pd.read_csv(VAL_PRED)
    test = pd.read_csv(TEST_PRED)
    feats = pd.read_csv(FEATURES)

    preds = pd.concat([val, test], axis=0, ignore_index=True)
    merged = preds.merge(
        feats,
        on=["essay_id", "split"],
        how="inner",
        validate="one_to_one",
    )

    missing = [c for c in HEURISTIC_COLS if c not in merged.columns]
    if missing:
        raise ValueError(f"Merged data missing heuristic columns: {missing}")

    val_mask = merged["split"].astype(str).str.lower() == "validation"
    scaler = StandardScaler()
    scaler.fit(merged.loc[val_mask, HEURISTIC_COLS].to_numpy(dtype=float))
    merged[HEURISTIC_COLS] = scaler.transform(merged[HEURISTIC_COLS].to_numpy(dtype=float))

    out_df = merged[OUTPUT_COLS].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)

    # Fit only on validation split, evaluate only on test split.
    test_mask = merged["split"].astype(str).str.lower() == "test"
    val_df = out_df.loc[val_mask].copy()
    test_df = out_df.loc[test_mask].copy()
    y_val = val_df["true_score"].to_numpy(dtype=float)
    y_test = test_df["true_score"].to_numpy(dtype=float)

    # Model A: DeBERTa only (direct baseline prediction, no fit).
    pred_a = test_df["deberta_pred"].to_numpy(dtype=float)
    model_a = regression_metrics(y_test, pred_a)
    model_a["qwk"] = qwk_metric(y_test, pred_a)

    # Model B: heuristic features only.
    x_val_b = val_df[HEURISTIC_COLS].to_numpy(dtype=float)
    x_test_b = test_df[HEURISTIC_COLS].to_numpy(dtype=float)
    reg_b = LinearRegression()
    reg_b.fit(x_val_b, y_val)
    pred_b = reg_b.predict(x_test_b)
    model_b = regression_metrics(y_test, pred_b)
    model_b["qwk"] = qwk_metric(y_test, pred_b)

    # Model C: DeBERTa + heuristic features.
    cols_c = ["deberta_pred", *HEURISTIC_COLS]
    x_val_c = val_df[cols_c].to_numpy(dtype=float)
    x_test_c = test_df[cols_c].to_numpy(dtype=float)
    reg_c = LinearRegression()
    reg_c.fit(x_val_c, y_val)
    pred_c = reg_c.predict(x_test_c)
    model_c = regression_metrics(y_test, pred_c)
    model_c["qwk"] = qwk_metric(y_test, pred_c)

    results = {
        "data": {
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "normalization": (
            "StandardScaler z-score on heuristic features only; fit on validation split, "
            "applied to validation and test."
        ),
        "models": {
            "A_deberta_only": model_a,
            "B_heuristics_only_linear_regression": model_b,
            "C_deberta_plus_heuristics_linear_regression": model_c,
        },
    }
    with open(RESULTS_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(out_df)} rows to {OUT}")
    print("Columns:", list(out_df.columns))
    print(
        "Normalization: StandardScaler (zero mean, unit variance per column), "
        "fit on validation split only; same transform applied to test."
    )
    print(f"Wrote experiment metrics to {RESULTS_OUT}")


if __name__ == "__main__":
    main()
