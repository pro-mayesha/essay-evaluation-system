from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FUSION_RESULTS = Path("experiments/outputs/fusion_results.json")
REDUCED_METRICS = Path("experiments/outputs/reduced_feature_validation/reduced_feature_metrics.csv")

OUT_CSV = Path("experiments/outputs/final_results_locked.csv")
OUT_SUMMARY = Path("experiments/outputs/final_results_summary.txt")


def main() -> None:
    with open(FUSION_RESULTS, encoding="utf-8") as f:
        fusion = json.load(f)

    reduced = pd.read_csv(REDUCED_METRICS)
    r = reduced[reduced["config"] == "core_plus_personal"].copy()

    model_a = fusion["models"]["A_deberta_only"]
    model_c = r[r["model"] == "Model C"].iloc[0]
    model_d_mean = r[r["model"] == "Model D (mean_5_seeds)"].iloc[0]
    model_d_std = r[r["model"] == "Model D (std_5_seeds)"].iloc[0]

    rows = [
        {
            "model": "Model A: DeBERTa only",
            "feature_setup": "deberta_pred",
            "qwk": float(model_a["qwk"]),
            "rmse": float(model_a["rmse"]),
            "mae": float(model_a["mae"]),
            "pearson": float(model_a["pearson_r"]),
            "metric_type": "single_run",
        },
        {
            "model": "Model C: DeBERTa + reduced features",
            "feature_setup": "deberta_pred,specificity,emotional_salience,personal_experience_salience",
            "qwk": float(model_c["test_qwk"]),
            "rmse": float(model_c["test_rmse"]),
            "mae": float(model_c["test_mae"]),
            "pearson": float(model_c["test_pearson"]),
            "metric_type": "single_run",
        },
        {
            "model": "Model D: DeBERTa + reduced features + GA",
            "feature_setup": "deberta_pred,specificity,emotional_salience,personal_experience_salience",
            "qwk": float(model_d_mean["test_qwk"]),
            "rmse": float(model_d_mean["test_rmse"]),
            "mae": float(model_d_mean["test_mae"]),
            "pearson": float(model_d_mean["test_pearson"]),
            "metric_type": "mean_5_seeds",
        },
        {
            "model": "Model D: DeBERTa + reduced features + GA",
            "feature_setup": "deberta_pred,specificity,emotional_salience,personal_experience_salience",
            "qwk": float(model_d_std["test_qwk"]),
            "rmse": float(model_d_std["test_rmse"]),
            "mae": float(model_d_std["test_mae"]),
            "pearson": float(model_d_std["test_pearson"]),
            "metric_type": "std_5_seeds",
        },
    ]

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    summary_lines = [
        "Final Locked Results",
        "====================",
        "",
        "Locked feature setup for Model C/D:",
        "- deberta_pred + specificity + emotional_salience + personal_experience_salience",
        "",
        "Model A (single run):",
        f"- QWK: {rows[0]['qwk']:.6f}",
        f"- RMSE: {rows[0]['rmse']:.6f}",
        f"- MAE: {rows[0]['mae']:.6f}",
        f"- Pearson: {rows[0]['pearson']:.6f}",
        "",
        "Model C (single run):",
        f"- QWK: {rows[1]['qwk']:.6f}",
        f"- RMSE: {rows[1]['rmse']:.6f}",
        f"- MAE: {rows[1]['mae']:.6f}",
        f"- Pearson: {rows[1]['pearson']:.6f}",
        "",
        "Model D (5-seed aggregate):",
        f"- Mean QWK: {rows[2]['qwk']:.6f} | Std QWK: {rows[3]['qwk']:.6f}",
        f"- Mean RMSE: {rows[2]['rmse']:.6f} | Std RMSE: {rows[3]['rmse']:.6f}",
        f"- Mean MAE: {rows[2]['mae']:.6f} | Std MAE: {rows[3]['mae']:.6f}",
        f"- Mean Pearson: {rows[2]['pearson']:.6f} | Std Pearson: {rows[3]['pearson']:.6f}",
        "",
        "Notes:",
        "- Model A uses existing baseline output.",
        "- Model C and Model D use the locked reduced feature set.",
        "- Model D values are reported as mean/std across the 5 fixed seeds.",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
