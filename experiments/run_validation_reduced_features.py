"""Reduced-feature validation run (isolated outputs, no overwrite).

Keeps existing outputs untouched. Evaluates:
1) core set: specificity + emotional_salience
2) core+personal set: specificity + emotional_salience + personal_experience_salience
for Model C (linear regression) and Model D (GA weighted fusion).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score


MERGED_PATH = Path("experiments/outputs/merged_experiment_data.csv")
OUT_DIR = Path("experiments/outputs/reduced_feature_validation")
OUT_METRICS = OUT_DIR / "reduced_feature_metrics.csv"
OUT_STABILITY = OUT_DIR / "reduced_feature_stability.csv"
OUT_SUMMARY = OUT_DIR / "reduced_feature_summary.txt"
OUT_BEST_WEIGHTS = OUT_DIR / "reduced_feature_best_ga_weights.json"

SEEDS = [11, 22, 33, 44, 55]

GA_POP_SIZE = 80
GA_GENERATIONS = 120
GA_ELITE_COUNT = 8
GA_TOURNAMENT_K = 4
GA_MUTATION_RATE = 0.25
GA_MUTATION_STD = 0.15
GA_INIT_MIN = -2.0
GA_INIT_MAX = 2.0

BASE = "deberta_pred"
CORE = ["specificity", "emotional_salience"]
OPTIONAL = "personal_experience_salience"

CONFIGS = {
    "core": [BASE, *CORE],
    "core_plus_personal": [BASE, *CORE, OPTIONAL],
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    pearson_r = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan")
    )
    return {"rmse": rmse, "mae": mae, "pearson_r": pearson_r}


def qwk_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    lo = int(np.floor(np.min(y_true)))
    hi = int(np.ceil(np.max(y_true)))
    yt = np.clip(np.round(y_true).astype(int), lo, hi)
    yp = np.clip(np.round(y_pred).astype(int), lo, hi)
    return float(cohen_kappa_score(yt, yp, weights="quadratic"))


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"qwk": qwk_metric(y_true, y_pred), **regression_metrics(y_true, y_pred)}


def tournament_select(rng: np.random.Generator, pop: np.ndarray, fit: np.ndarray, k: int) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmax(fit[idx])]].copy()


def run_ga_weights(x_val: np.ndarray, y_val: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_features = x_val.shape[1]
    pop = rng.uniform(GA_INIT_MIN, GA_INIT_MAX, size=(GA_POP_SIZE, n_features))
    best_w = None
    best_fit = -np.inf

    for _ in range(GA_GENERATIONS):
        fit = np.zeros(GA_POP_SIZE, dtype=float)
        for i in range(GA_POP_SIZE):
            fit[i] = qwk_metric(y_val, x_val @ pop[i])

        order = np.argsort(fit)[::-1]
        if float(fit[order[0]]) > best_fit:
            best_fit = float(fit[order[0]])
            best_w = pop[order[0]].copy()

        elites = pop[order[:GA_ELITE_COUNT]].copy()
        next_pop = [e for e in elites]
        while len(next_pop) < GA_POP_SIZE:
            p1 = tournament_select(rng, pop, fit, GA_TOURNAMENT_K)
            p2 = tournament_select(rng, pop, fit, GA_TOURNAMENT_K)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MERGED_PATH)
    val_df = df[df["split"].astype(str).str.lower() == "validation"].copy()
    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()
    y_val = val_df["true_score"].to_numpy(dtype=float)
    y_test = test_df["true_score"].to_numpy(dtype=float)

    metric_rows = []
    stability_rows = []
    best_weights_payload: dict[str, dict[str, float]] = {}

    for config_name, cols in CONFIGS.items():
        x_val = val_df[cols].to_numpy(dtype=float)
        x_test = test_df[cols].to_numpy(dtype=float)

        # Model C (deterministic)
        reg = LinearRegression()
        reg.fit(x_val, y_val)
        pred_test_c = reg.predict(x_test)
        m_c = eval_metrics(y_test, pred_test_c)
        metric_rows.append(
            {
                "config": config_name,
                "model": "Model C",
                "features_used": ",".join(cols),
                "test_qwk": m_c["qwk"],
                "test_rmse": m_c["rmse"],
                "test_mae": m_c["mae"],
                "test_pearson": m_c["pearson_r"],
            }
        )

        # Model D (5-seed stability)
        d_runs = []
        for seed in SEEDS:
            w = run_ga_weights(x_val, y_val, seed)
            pred_test_d = x_test @ w
            m_d = eval_metrics(y_test, pred_test_d)
            row = {
                "config": config_name,
                "seed": seed,
                "test_qwk": m_d["qwk"],
                "test_rmse": m_d["rmse"],
                "test_mae": m_d["mae"],
                "test_pearson": m_d["pearson_r"],
            }
            for c, wi in zip(cols, w):
                row[f"w_{c}"] = float(wi)
            d_runs.append(row)
            stability_rows.append(row)

        d_df = pd.DataFrame(d_runs)
        means = d_df[["test_qwk", "test_rmse", "test_mae", "test_pearson"]].mean()
        stds = d_df[["test_qwk", "test_rmse", "test_mae", "test_pearson"]].std(ddof=1)
        metric_rows.append(
            {
                "config": config_name,
                "model": "Model D (mean_5_seeds)",
                "features_used": ",".join(cols),
                "test_qwk": float(means["test_qwk"]),
                "test_rmse": float(means["test_rmse"]),
                "test_mae": float(means["test_mae"]),
                "test_pearson": float(means["test_pearson"]),
            }
        )
        metric_rows.append(
            {
                "config": config_name,
                "model": "Model D (std_5_seeds)",
                "features_used": ",".join(cols),
                "test_qwk": float(stds["test_qwk"]),
                "test_rmse": float(stds["test_rmse"]),
                "test_mae": float(stds["test_mae"]),
                "test_pearson": float(stds["test_pearson"]),
            }
        )

        best_idx = int(d_df["test_qwk"].idxmax())
        best = d_df.loc[best_idx]
        best_weights_payload[config_name] = {
            "best_seed_by_test_qwk": int(best["seed"]),
            **{c: float(best[f"w_{c}"]) for c in cols},
        }

    pd.DataFrame(metric_rows).to_csv(OUT_METRICS, index=False)
    pd.DataFrame(stability_rows).to_csv(OUT_STABILITY, index=False)
    OUT_BEST_WEIGHTS.write_text(json.dumps(best_weights_payload, indent=2), encoding="utf-8")

    m = pd.read_csv(OUT_METRICS)
    c_core = m[(m["config"] == "core") & (m["model"] == "Model C")].iloc[0]
    c_cpp = m[(m["config"] == "core_plus_personal") & (m["model"] == "Model C")].iloc[0]
    d_core = m[(m["config"] == "core") & (m["model"] == "Model D (mean_5_seeds)")].iloc[0]
    d_cpp = m[(m["config"] == "core_plus_personal") & (m["model"] == "Model D (mean_5_seeds)")].iloc[0]
    d_core_std = m[(m["config"] == "core") & (m["model"] == "Model D (std_5_seeds)")].iloc[0]
    d_cpp_std = m[(m["config"] == "core_plus_personal") & (m["model"] == "Model D (std_5_seeds)")].iloc[0]

    lines = [
        "Reduced-Feature Validation Summary",
        "=================================",
        "",
        "Dropped by design: concreteness, narrative_event_density",
        "Core kept: specificity, emotional_salience",
        "Optional variant: + personal_experience_salience",
        "",
        "Model C (test):",
        f"  core -> qwk={c_core['test_qwk']:.6f}, rmse={c_core['test_rmse']:.6f}, mae={c_core['test_mae']:.6f}, pearson={c_core['test_pearson']:.6f}",
        f"  core_plus_personal -> qwk={c_cpp['test_qwk']:.6f}, rmse={c_cpp['test_rmse']:.6f}, mae={c_cpp['test_mae']:.6f}, pearson={c_cpp['test_pearson']:.6f}",
        "",
        "Model D mean across 5 seeds (test):",
        f"  core -> qwk={d_core['test_qwk']:.6f}, rmse={d_core['test_rmse']:.6f}, mae={d_core['test_mae']:.6f}, pearson={d_core['test_pearson']:.6f}",
        f"  core_plus_personal -> qwk={d_cpp['test_qwk']:.6f}, rmse={d_cpp['test_rmse']:.6f}, mae={d_cpp['test_mae']:.6f}, pearson={d_cpp['test_pearson']:.6f}",
        "",
        "Model D stability (std across 5 seeds, test):",
        f"  core -> qwk_std={d_core_std['test_qwk']:.6f}, rmse_std={d_core_std['test_rmse']:.6f}, mae_std={d_core_std['test_mae']:.6f}, pearson_std={d_core_std['test_pearson']:.6f}",
        f"  core_plus_personal -> qwk_std={d_cpp_std['test_qwk']:.6f}, rmse_std={d_cpp_std['test_rmse']:.6f}, mae_std={d_cpp_std['test_mae']:.6f}, pearson_std={d_cpp_std['test_pearson']:.6f}",
        "",
        "Interpretation:",
        "- Prefer the configuration with higher QWK and lower seed std if the goal is ordinal agreement with stable behavior.",
        "- Keeping the optional personal feature is worthwhile only if it improves QWK without hurting stability.",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_METRICS}")
    print(f"Wrote {OUT_STABILITY}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_BEST_WEIGHTS}")


if __name__ == "__main__":
    main()
