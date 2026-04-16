"""Error analysis: Model A (DeBERTa only) vs Model D (GA fusion) on test split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MERGED = Path("experiments/outputs/merged_experiment_data.csv")
WEIGHTS = Path("experiments/outputs/best_ga_weights.json")
RAW_FEATURES = Path("experiments/outputs/availability_features.csv")
OUT_IMPROVED = Path("experiments/outputs/error_analysis_improved.csv")
OUT_WORSENED = Path("experiments/outputs/error_analysis_worsened.csv")
OUT_SUMMARY = Path("experiments/outputs/error_analysis_summary.txt")

FEATURE_ORDER = [
    "deberta_pred",
    "concreteness",
    "specificity",
    "emotional_salience",
    "personal_experience_salience",
    "narrative_event_density",
]

HEURISTIC_RAW = [
    "concreteness",
    "specificity",
    "emotional_salience",
    "personal_experience_salience",
    "narrative_event_density",
]


def model_d_pred(df: pd.DataFrame, w: dict[str, float]) -> np.ndarray:
    x = np.column_stack([df[c].to_numpy(dtype=float) for c in FEATURE_ORDER])
    vec = np.array([w[c] for c in FEATURE_ORDER], dtype=float)
    return x @ vec


def main() -> None:
    merged = pd.read_csv(MERGED)
    test = merged[merged["split"].astype(str).str.lower() == "test"].copy()
    if test.empty:
        raise ValueError("No test rows in merged_experiment_data.csv")

    with open(WEIGHTS, encoding="utf-8") as f:
        payload = json.load(f)
    w = payload["weights"]

    y = test["true_score"].to_numpy(dtype=float)
    pred_a = test["deberta_pred"].to_numpy(dtype=float)
    pred_d = model_d_pred(test, w)

    abs_a = np.abs(pred_a - y)
    abs_d = np.abs(pred_d - y)
    # Positive => Model D reduced absolute error vs Model A
    abs_err_diff = abs_a - abs_d

    work = test[["essay_id"]].copy()
    work["true_score"] = y
    work["model_a_pred"] = pred_a
    work["model_d_pred"] = pred_d
    work["abs_error_model_a"] = abs_a
    work["abs_error_model_d"] = abs_d
    work["abs_error_diff"] = abs_err_diff

    raw = pd.read_csv(RAW_FEATURES)
    raw_test = raw[raw["split"].astype(str).str.lower() == "test"][
        ["essay_id", "split"] + HEURISTIC_RAW
    ].copy()
    for c in HEURISTIC_RAW:
        raw_test = raw_test.rename(columns={c: f"{c}_raw"})

    work = work.merge(raw_test, on=["essay_id"], how="left", validate="one_to_one")

    improved = work.nlargest(10, "abs_error_diff")
    worsened = work.nsmallest(10, "abs_error_diff")

    cols_out = [
        "essay_id",
        "true_score",
        "model_a_pred",
        "model_d_pred",
        "abs_error_diff",
        *[f"{h}_raw" for h in HEURISTIC_RAW],
    ]
    OUT_IMPROVED.parent.mkdir(parents=True, exist_ok=True)
    improved[cols_out].to_csv(OUT_IMPROVED, index=False)
    worsened[cols_out].to_csv(OUT_WORSENED, index=False)

    # Aggregate pattern hints: compare means on full test vs improved vs worsened samples
    def means(sub: pd.DataFrame) -> dict[str, float]:
        return {f"{h}_raw": float(sub[f"{h}_raw"].mean()) for h in HEURISTIC_RAW}

    m_all = means(work)
    m_imp = means(improved)
    m_worse = means(worsened)

    lines = [
        "Error analysis: Model A (deberta_pred) vs Model D (GA fusion weights on merged features)",
        "=====================================================================================",
        "",
        "Definition:",
        "- abs_error_diff = |true - model_a| - |true - model_d|  (positive => D improved)",
        "",
        f"Test rows: {len(work)}",
        "",
        "Mean raw heuristic values (full test vs top-10 improved vs top-10 worsened):",
    ]
    for label, m in [("full_test", m_all), ("top10_improved", m_imp), ("top10_worsened", m_worse)]:
        lines.append(f"  [{label}]")
        for k, v in m.items():
            lines.append(f"    {k}: {v:.6f}")

    lines += [
        "",
        "Short interpretation (heuristic proxies; not causal):",
    ]

    # Simple comparisons
    spec_delta_imp = m_imp["specificity_raw"] - m_all["specificity_raw"]
    spec_delta_w = m_worse["specificity_raw"] - m_all["specificity_raw"]
    pers_delta_imp = m_imp["personal_experience_salience_raw"] - m_all["personal_experience_salience_raw"]
    pers_delta_w = m_worse["personal_experience_salience_raw"] - m_all["personal_experience_salience_raw"]

    if spec_delta_imp > spec_delta_w and pers_delta_imp > pers_delta_w:
        lines.append(
            "- Improved essays (this run) tend to show slightly higher mean specificity and "
            "personal-experience salience than the worsened top-10, relative to the test average."
        )
    elif spec_delta_imp < spec_delta_w:
        lines.append(
            "- Specificity signal is mixed: worsened examples are not clearly less specific than improved in this small sample."
        )
    else:
        lines.append(
            "- Specificity/personal patterns are subtle in this n=10 slice; use with caution."
        )

    if m_worse["emotional_salience_raw"] > m_imp["emotional_salience_raw"]:
        lines.append(
            "- Worsened essays show higher mean emotional_salience_raw than improved essays here, "
            "suggesting GA corrections may occasionally misfire on more affect-heavy drafts."
        )
    else:
        lines.append(
            "- Emotional salience differences between improved and worsened groups are not large in this slice."
        )

    lines += [
        "",
        "Caveat: Top-10 slices are tiny; patterns are exploratory.",
        "",
        f"Saved: {OUT_IMPROVED}",
        f"Saved: {OUT_WORSENED}",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_IMPROVED}, {OUT_WORSENED}, {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
