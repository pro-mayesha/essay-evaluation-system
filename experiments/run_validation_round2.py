"""Validation round 2 runner.

Step 1: Stability check for Model C and Model D (5 seeds)
Step 2: Feature ablation for Model C and Model D
Step 3: Error analysis comparing Model A vs C vs D
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score


MERGED_PATH = Path("experiments/outputs/merged_experiment_data.csv")
RAW_FEATURES_PATH = Path("experiments/outputs/availability_features.csv")
BASE_WEIGHTS_PATH = Path("experiments/outputs/best_ga_weights.json")

OUT_C_STABILITY = Path("experiments/outputs/model_c_stability.csv")
OUT_D_STABILITY = Path("experiments/outputs/model_d_stability.csv")
OUT_STABILITY_SUMMARY = Path("experiments/outputs/stability_summary.txt")

OUT_ABLATION = Path("experiments/outputs/ablation_results.csv")
OUT_ABLATION_SUMMARY = Path("experiments/outputs/ablation_summary.txt")

OUT_ERR_IMPROVED = Path("experiments/outputs/error_analysis_improved.csv")
OUT_ERR_WORSENED = Path("experiments/outputs/error_analysis_worsened.csv")
OUT_ERR_SUMMARY = Path("experiments/outputs/error_analysis_summary.txt")

SEEDS = [11, 22, 33, 44, 55]

GA_POP_SIZE = 80
GA_GENERATIONS = 120
GA_ELITE_COUNT = 8
GA_TOURNAMENT_K = 4
GA_MUTATION_RATE = 0.25
GA_MUTATION_STD = 0.15
GA_INIT_MIN = -2.0
GA_INIT_MAX = 2.0

HEURISTICS = [
    "concreteness",
    "specificity",
    "emotional_salience",
    "personal_experience_salience",
    "narrative_event_density",
]
MODEL_C_COLS_FULL = ["deberta_pred", *HEURISTICS]


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
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    lo = int(np.floor(np.min(y_true)))
    hi = int(np.ceil(np.max(y_true)))
    y_true_i = np.clip(np.round(y_true).astype(int), lo, hi)
    y_pred_i = np.clip(np.round(y_pred).astype(int), lo, hi)
    return float(cohen_kappa_score(y_true_i, y_pred_i, weights="quadratic"))


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"qwk": qwk_metric(y_true, y_pred), **regression_metrics(y_true, y_pred)}


def predict_scores(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w


def tournament_select(rng: np.random.Generator, pop: np.ndarray, fitness: np.ndarray, k: int) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmax(fitness[idx])]].copy()


def run_ga_weights(x_val: np.ndarray, y_val: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_features = x_val.shape[1]
    pop = rng.uniform(GA_INIT_MIN, GA_INIT_MAX, size=(GA_POP_SIZE, n_features))

    best_w = None
    best_fit = -np.inf
    for _ in range(GA_GENERATIONS):
        fitness = np.zeros(GA_POP_SIZE, dtype=float)
        for i in range(GA_POP_SIZE):
            fitness[i] = qwk_metric(y_val, predict_scores(x_val, pop[i]))

        order = np.argsort(fitness)[::-1]
        if float(fitness[order[0]]) > best_fit:
            best_fit = float(fitness[order[0]])
            best_w = pop[order[0]].copy()

        elites = pop[order[:GA_ELITE_COUNT]].copy()
        next_pop = [e for e in elites]
        while len(next_pop) < GA_POP_SIZE:
            p1 = tournament_select(rng, pop, fitness, GA_TOURNAMENT_K)
            p2 = tournament_select(rng, pop, fitness, GA_TOURNAMENT_K)
            alpha = rng.uniform(0.0, 1.0, size=n_features)
            child = alpha * p1 + (1.0 - alpha) * p2
            for j in range(n_features):
                if rng.random() < GA_MUTATION_RATE:
                    child[j] += rng.normal(0.0, GA_MUTATION_STD)
            next_pop.append(child)
        pop = np.array(next_pop)

    assert best_w is not None
    return best_w


def main() -> None:
    df = pd.read_csv(MERGED_PATH)
    raw = pd.read_csv(RAW_FEATURES_PATH)

    val_df = df[df["split"].astype(str).str.lower() == "validation"].copy()
    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
    y_val = val_df["true_score"].to_numpy(dtype=float)
    y_test = test_df["true_score"].to_numpy(dtype=float)

    # ---------------- Step 1: Stability ----------------
    c_rows: list[dict[str, float | int]] = []
    d_rows: list[dict[str, float | int]] = []

    for seed in SEEDS:
        # Model C: linear regression on deberta_pred + heuristics (deterministic model).
        x_val_c = val_df[MODEL_C_COLS_FULL].to_numpy(dtype=float)
        x_test_c = test_df[MODEL_C_COLS_FULL].to_numpy(dtype=float)
        reg_c = LinearRegression()
        reg_c.fit(x_val_c, y_val)
        pred_val_c = reg_c.predict(x_val_c)
        pred_test_c = reg_c.predict(x_test_c)
        m_val_c = eval_metrics(y_val, pred_val_c)
        m_test_c = eval_metrics(y_test, pred_test_c)
        c_rows.append(
            {
                "seed": seed,
                "validation_qwk": m_val_c["qwk"],
                "validation_rmse": m_val_c["rmse"],
                "validation_mae": m_val_c["mae"],
                "validation_pearson": m_val_c["pearson_r"],
                "test_qwk": m_test_c["qwk"],
                "test_rmse": m_test_c["rmse"],
                "test_mae": m_test_c["mae"],
                "test_pearson": m_test_c["pearson_r"],
            }
        )

        # Model D: GA fusion on same columns; seed controls optimization randomness.
        w = run_ga_weights(x_val_c, y_val, seed)
        pred_val_d = predict_scores(x_val_c, w)
        pred_test_d = predict_scores(x_test_c, w)
        m_val_d = eval_metrics(y_val, pred_val_d)
        m_test_d = eval_metrics(y_test, pred_test_d)
        row_d: dict[str, float | int] = {
            "seed": seed,
            "validation_qwk": m_val_d["qwk"],
            "validation_rmse": m_val_d["rmse"],
            "validation_mae": m_val_d["mae"],
            "validation_pearson": m_val_d["pearson_r"],
            "test_qwk": m_test_d["qwk"],
            "test_rmse": m_test_d["rmse"],
            "test_mae": m_test_d["mae"],
            "test_pearson": m_test_d["pearson_r"],
        }
        for col, weight in zip(MODEL_C_COLS_FULL, w):
            row_d[f"w_{col}"] = float(weight)
        d_rows.append(row_d)

    c_stability = pd.DataFrame(c_rows)
    d_stability = pd.DataFrame(d_rows)
    OUT_C_STABILITY.parent.mkdir(parents=True, exist_ok=True)
    c_stability.to_csv(OUT_C_STABILITY, index=False)
    d_stability.to_csv(OUT_D_STABILITY, index=False)

    metric_cols = [
        "validation_qwk",
        "validation_rmse",
        "validation_mae",
        "validation_pearson",
        "test_qwk",
        "test_rmse",
        "test_mae",
        "test_pearson",
    ]
    c_mean = c_stability[metric_cols].mean()
    c_std = c_stability[metric_cols].std(ddof=1)
    d_mean = d_stability[metric_cols].mean()
    d_std = d_stability[metric_cols].std(ddof=1)

    stability_lines = [
        "Stability Summary (5 seeds)",
        "===========================",
        "",
        f"Seeds: {SEEDS}",
        "",
        "Model C mean ± std:",
    ]
    for c in metric_cols:
        stability_lines.append(f"  {c}: {c_mean[c]:.6f} ± {c_std[c]:.6f}")
    stability_lines += ["", "Model D mean ± std:"]
    for c in metric_cols:
        stability_lines.append(f"  {c}: {d_mean[c]:.6f} ± {d_std[c]:.6f}")
    stability_lines += [
        "",
        "Stability note:",
        "- Model C is effectively deterministic in this setup (linear regression), so seed variability is near zero.",
        "- Model D varies with GA seed; observed std indicates low-to-moderate variability if small (<~0.01 QWK std).",
    ]
    OUT_STABILITY_SUMMARY.write_text("\n".join(stability_lines) + "\n", encoding="utf-8")

    # ---------------- Step 2: Ablation ----------------
    ablation_rows: list[dict[str, float | str]] = []
    configs: list[tuple[str, list[str]]] = [("full", MODEL_C_COLS_FULL)]
    for drop in HEURISTICS:
        cols = ["deberta_pred"] + [h for h in HEURISTICS if h != drop]
        configs.append((f"drop_{drop}", cols))

    for cfg_name, cols in configs:
        x_val = val_df[cols].to_numpy(dtype=float)
        x_test = test_df[cols].to_numpy(dtype=float)

        # Model C
        reg = LinearRegression()
        reg.fit(x_val, y_val)
        pred_val_c = reg.predict(x_val)
        pred_test_c = reg.predict(x_test)
        m_val_c = eval_metrics(y_val, pred_val_c)
        m_test_c = eval_metrics(y_test, pred_test_c)
        ablation_rows.append(
            {
                "model": "Model C",
                "config": cfg_name,
                "features_used": ",".join(cols),
                "validation_qwk": m_val_c["qwk"],
                "test_qwk": m_test_c["qwk"],
                "rmse": m_test_c["rmse"],
                "mae": m_test_c["mae"],
                "pearson": m_test_c["pearson_r"],
            }
        )

        # Model D (GA with fixed seed for controlled ablation)
        w = run_ga_weights(x_val, y_val, seed=42)
        pred_val_d = predict_scores(x_val, w)
        pred_test_d = predict_scores(x_test, w)
        m_val_d = eval_metrics(y_val, pred_val_d)
        m_test_d = eval_metrics(y_test, pred_test_d)
        ablation_rows.append(
            {
                "model": "Model D",
                "config": cfg_name,
                "features_used": ",".join(cols),
                "validation_qwk": m_val_d["qwk"],
                "test_qwk": m_test_d["qwk"],
                "rmse": m_test_d["rmse"],
                "mae": m_test_d["mae"],
                "pearson": m_test_d["pearson_r"],
            }
        )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUT_ABLATION, index=False)

    ablation_lines = [
        "Ablation Summary",
        "================",
        "",
        "Delta test QWK vs full config by model:",
    ]
    for model_name in ["Model C", "Model D"]:
        sub = ablation_df[ablation_df["model"] == model_name].copy()
        full_qwk = float(sub[sub["config"] == "full"]["test_qwk"].iloc[0])
        sub = sub[sub["config"] != "full"].copy()
        sub["delta_test_qwk_vs_full"] = sub["test_qwk"] - full_qwk
        sub = sub.sort_values("delta_test_qwk_vs_full")
        ablation_lines.append(f"  [{model_name}] full_test_qwk={full_qwk:.6f}")
        for _, row in sub.iterrows():
            ablation_lines.append(
                f"    {row['config']}: test_qwk={float(row['test_qwk']):.6f}, "
                f"delta_vs_full={float(row['delta_test_qwk_vs_full']):.6f}"
            )
    OUT_ABLATION_SUMMARY.write_text("\n".join(ablation_lines) + "\n", encoding="utf-8")

    # ---------------- Step 3: Error analysis ----------------
    with open(BASE_WEIGHTS_PATH, encoding="utf-8") as f:
        w_payload = json.load(f)
    w_base = w_payload["weights"]
    w_vec = np.array([w_base[c] for c in MODEL_C_COLS_FULL], dtype=float)

    x_test_full = test_df[MODEL_C_COLS_FULL].to_numpy(dtype=float)
    pred_a = test_df["deberta_pred"].to_numpy(dtype=float)

    # Model C (full features, fixed fit)
    reg_c_full = LinearRegression()
    reg_c_full.fit(val_df[MODEL_C_COLS_FULL].to_numpy(dtype=float), y_val)
    pred_c = reg_c_full.predict(x_test_full)

    # Model D from current GA baseline weights
    pred_d = x_test_full @ w_vec

    abs_a = np.abs(pred_a - y_test)
    abs_d = np.abs(pred_d - y_test)
    diff = abs_a - abs_d

    err = pd.DataFrame(
        {
            "essay_id": test_df["essay_id"].values,
            "true_score": y_test,
            "model_a_pred": pred_a,
            "model_c_pred": pred_c,
            "model_d_pred": pred_d,
            "abs_error_diff_d_vs_a": diff,
        }
    )

    raw_test = raw[raw["split"].astype(str).str.lower() == "test"][
        ["essay_id", *HEURISTICS]
    ].copy()
    for h in HEURISTICS:
        raw_test = raw_test.rename(columns={h: f"{h}_raw"})

    err = err.merge(raw_test, on="essay_id", how="left", validate="one_to_one")
    out_cols = [
        "essay_id",
        "true_score",
        "model_a_pred",
        "model_c_pred",
        "model_d_pred",
        "abs_error_diff_d_vs_a",
        *[f"{h}_raw" for h in HEURISTICS],
    ]
    improved = err.nlargest(20, "abs_error_diff_d_vs_a")[out_cols]
    worsened = err.nsmallest(20, "abs_error_diff_d_vs_a")[out_cols]
    improved.to_csv(OUT_ERR_IMPROVED, index=False)
    worsened.to_csv(OUT_ERR_WORSENED, index=False)

    def group_means(df_in: pd.DataFrame) -> dict[str, float]:
        return {f"{h}_raw": float(df_in[f"{h}_raw"].mean()) for h in HEURISTICS}

    m_all = group_means(err)
    m_imp = group_means(improved)
    m_worse = group_means(worsened)

    err_lines = [
        "Error Analysis Summary (A vs C vs D on test split)",
        "===============================================",
        "",
        "Top-20 improved/worsened by Model D over Model A are saved to CSV.",
        "",
        "Mean raw heuristics:",
        "  [full_test]",
    ]
    for k, v in m_all.items():
        err_lines.append(f"    {k}: {v:.6f}")
    err_lines.append("  [top20_improved_d_vs_a]")
    for k, v in m_imp.items():
        err_lines.append(f"    {k}: {v:.6f}")
    err_lines.append("  [top20_worsened_d_vs_a]")
    for k, v in m_worse.items():
        err_lines.append(f"    {k}: {v:.6f}")
    err_lines += [
        "",
        "Pattern notes (exploratory):",
        "- Check whether specificity_raw and personal_experience_salience_raw are higher in improved vs worsened groups.",
        "- Check whether emotional_salience_raw is elevated in worsened essays.",
        "",
        "Caveat: top-20 slices are small; treat as directional, not causal.",
    ]
    OUT_ERR_SUMMARY.write_text("\n".join(err_lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_C_STABILITY}, {OUT_D_STABILITY}, {OUT_STABILITY_SUMMARY}")
    print(f"Wrote {OUT_ABLATION}, {OUT_ABLATION_SUMMARY}")
    print(f"Wrote {OUT_ERR_IMPROVED}, {OUT_ERR_WORSENED}, {OUT_ERR_SUMMARY}")


if __name__ == "__main__":
    main()
