from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score


MERGED_PATH = Path("experiments/outputs/merged_experiment_data.csv")
GA_WEIGHTS_PATH = Path("experiments/outputs/reduced_feature_validation/reduced_feature_best_ga_weights.json")
OUT_PATH = Path("experiments/outputs/bootstrap_qwk_ci_summary.txt")

LOCKED_COLS = [
    "deberta_pred",
    "specificity",
    "emotional_salience",
    "personal_experience_salience",
]


def qwk_int(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lo = int(np.floor(np.min(y_true)))
    hi = int(np.ceil(np.max(y_true)))
    yt = np.clip(np.round(y_true).astype(int), lo, hi)
    yp = np.clip(np.round(y_pred).astype(int), lo, hi)
    return float(cohen_kappa_score(yt, yp, weights="quadratic"))


def to_int_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = int(np.floor(np.min(y_true)))
    hi = int(np.ceil(np.max(y_true)))
    yt = np.clip(np.round(y_true).astype(int), lo, hi)
    yp = np.clip(np.round(y_pred).astype(int), lo, hi)
    return yt, yp


def bootstrap_qwk_ci(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    n_iterations: int = 1000,
    ci: float = 95,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true_int)
    scores = np.empty(n_iterations, dtype=float)

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        scores[i] = cohen_kappa_score(y_true_int[idx], y_pred_int[idx], weights="quadratic")

    lower = float(np.percentile(scores, (100 - ci) / 2))
    upper = float(np.percentile(scores, 100 - (100 - ci) / 2))
    mean = float(np.mean(scores))
    return mean, lower, upper


def bootstrap_diff_ci(
    y_true_int: np.ndarray,
    y_pred_x_int: np.ndarray,
    y_pred_y_int: np.ndarray,
    n_iterations: int = 1000,
    ci: float = 95,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true_int)
    diffs = np.empty(n_iterations, dtype=float)

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        q_x = cohen_kappa_score(y_true_int[idx], y_pred_x_int[idx], weights="quadratic")
        q_y = cohen_kappa_score(y_true_int[idx], y_pred_y_int[idx], weights="quadratic")
        diffs[i] = q_x - q_y

    lower = float(np.percentile(diffs, (100 - ci) / 2))
    upper = float(np.percentile(diffs, 100 - (100 - ci) / 2))
    mean = float(np.mean(diffs))
    return mean, lower, upper


def main() -> None:
    df = pd.read_csv(MERGED_PATH)
    val_df = df[df["split"].astype(str).str.lower() == "validation"].copy()
    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
    y_true = test_df["true_score"].to_numpy(dtype=float)

    pred_a = test_df["deberta_pred"].to_numpy(dtype=float)

    reg = LinearRegression()
    reg.fit(val_df[LOCKED_COLS].to_numpy(dtype=float), val_df["true_score"].to_numpy(dtype=float))
    pred_c = reg.predict(test_df[LOCKED_COLS].to_numpy(dtype=float))

    with open(GA_WEIGHTS_PATH, encoding="utf-8") as f:
        best_weights = json.load(f)["core_plus_personal"]
    w = np.array([best_weights[c] for c in LOCKED_COLS], dtype=float)
    pred_d = test_df[LOCKED_COLS].to_numpy(dtype=float) @ w

    yt_a, yp_a = to_int_scores(y_true, pred_a)
    yt_c, yp_c = to_int_scores(y_true, pred_c)
    yt_d, yp_d = to_int_scores(y_true, pred_d)

    assert np.array_equal(yt_a, yt_c) and np.array_equal(yt_c, yt_d)
    yt = yt_a

    rows = []
    for name, pred_raw, pred_int in [
        ("A (Baseline)", pred_a, yp_a),
        ("C (+Avail)", pred_c, yp_c),
        ("D (+Avail+GA)", pred_d, yp_d),
    ]:
        point_qwk = qwk_int(y_true, pred_raw)
        mean, lo, hi = bootstrap_qwk_ci(yt, pred_int)
        rows.append((name, point_qwk, mean, lo, hi))

    diff_rows = []
    for left, right, pred_l, pred_r in [
        ("D", "A", yp_d, yp_a),
        ("D", "C", yp_d, yp_c),
        ("C", "A", yp_c, yp_a),
    ]:
        mean, lo, hi = bootstrap_diff_ci(yt, pred_l, pred_r)
        sig = "YES" if (lo > 0 or hi < 0) else "NO"
        diff_rows.append((left, right, mean, lo, hi, sig))

    lines = [
        "Bootstrap QWK Confidence Intervals (Locked Setup)",
        "=================================================",
        "",
        f"Rows: test={len(test_df)}",
        "Bootstrap: n_iterations=1000, ci=95, seed=42",
        "",
    ]
    for name, point, mean, lo, hi in rows:
        lines.append(
            f"{name}: point_qwk={point:.6f}, bootstrap_mean={mean:.6f}, 95% CI=[{lo:.6f}, {hi:.6f}]"
        )

    lines.extend(["", "Pairwise bootstrap differences (QWK_left - QWK_right):"])
    for left, right, mean, lo, hi, sig in diff_rows:
        lines.append(
            f"{left}-{right}: mean_diff={mean:.6f}, 95% CI=[{lo:.6f}, {hi:.6f}], significant={sig}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
