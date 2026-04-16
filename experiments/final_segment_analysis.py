from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


MERGED = Path("experiments/outputs/merged_experiment_data.csv")
REDUCED_WEIGHTS = Path("experiments/outputs/reduced_feature_validation/reduced_feature_best_ga_weights.json")
OUT_CSV = Path("experiments/outputs/final_segment_analysis.csv")
OUT_TXT = Path("experiments/outputs/final_segment_summary.txt")

FEATURES = [
    "deberta_pred",
    "specificity",
    "emotional_salience",
    "personal_experience_salience",
]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    pearson = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan")
    )
    return {"rmse": rmse, "mae": mae, "pearson": pearson}


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lo = int(np.floor(np.min(y_true)))
    hi = int(np.ceil(np.max(y_true)))
    yt = np.clip(np.round(y_true).astype(int), lo, hi)
    yp = np.clip(np.round(y_pred).astype(int), lo, hi)
    return float(cohen_kappa_score(yt, yp, weights="quadratic"))


def eval_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"qwk": qwk(y_true, y_pred), **regression_metrics(y_true, y_pred)}


def main() -> None:
    df = pd.read_csv(MERGED)
    test = df[df["split"].astype(str).str.lower() == "test"].copy()

    with open(REDUCED_WEIGHTS, encoding="utf-8") as f:
        payload = json.load(f)
    w = payload["core_plus_personal"]
    w_vec = np.array([w[c] for c in FEATURES], dtype=float)

    y = test["true_score"].to_numpy(dtype=float)
    pred_a = test["deberta_pred"].to_numpy(dtype=float)
    x = test[FEATURES].to_numpy(dtype=float)
    pred_d = x @ w_vec

    spec_med = float(test["specificity"].median())
    emo_med = float(test["emotional_salience"].median())

    groups = {
        "specificity_low": test["specificity"] <= spec_med,
        "specificity_high": test["specificity"] > spec_med,
        "emotional_salience_low": test["emotional_salience"] <= emo_med,
        "emotional_salience_high": test["emotional_salience"] > emo_med,
    }

    rows = []
    for name, mask in groups.items():
        idx = mask.to_numpy()
        y_g = y[idx]
        a_g = pred_a[idx]
        d_g = pred_d[idx]
        m_a = eval_all(y_g, a_g)
        m_d = eval_all(y_g, d_g)
        rows.append(
            {
                "group": name,
                "n": int(idx.sum()),
                "model_a_qwk": m_a["qwk"],
                "model_a_rmse": m_a["rmse"],
                "model_a_mae": m_a["mae"],
                "model_a_pearson": m_a["pearson"],
                "model_d_qwk": m_d["qwk"],
                "model_d_rmse": m_d["rmse"],
                "model_d_mae": m_d["mae"],
                "model_d_pearson": m_d["pearson"],
                "delta_qwk_d_minus_a": m_d["qwk"] - m_a["qwk"],
                "delta_rmse_d_minus_a": m_d["rmse"] - m_a["rmse"],
                "delta_mae_d_minus_a": m_d["mae"] - m_a["mae"],
                "delta_pearson_d_minus_a": m_d["pearson"] - m_a["pearson"],
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    high_emo = out[out["group"] == "emotional_salience_high"].iloc[0]
    low_emo = out[out["group"] == "emotional_salience_low"].iloc[0]
    high_spec = out[out["group"] == "specificity_high"].iloc[0]
    low_spec = out[out["group"] == "specificity_low"].iloc[0]

    lines = [
        "Final Segment Analysis (Test Split)",
        "===================================",
        "",
        "Grouping method:",
        "- Median split on normalized specificity and emotional_salience from merged_experiment_data.csv",
        f"- specificity median: {spec_med:.6f}",
        f"- emotional_salience median: {emo_med:.6f}",
        "",
        "Goal check: Does Model D improve more on high-salience essays?",
        f"- High emotional salience delta QWK (D-A): {high_emo['delta_qwk_d_minus_a']:.6f}",
        f"- Low emotional salience delta QWK (D-A): {low_emo['delta_qwk_d_minus_a']:.6f}",
        "",
        "Specificity split (delta QWK D-A):",
        f"- High specificity: {high_spec['delta_qwk_d_minus_a']:.6f}",
        f"- Low specificity: {low_spec['delta_qwk_d_minus_a']:.6f}",
        "",
        "Interpretation:",
        "- Positive delta QWK indicates Model D outperforms Model A on ordinal agreement in that segment.",
        "- Use CSV for full RMSE/MAE/Pearson deltas per segment.",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
