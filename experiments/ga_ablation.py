"""Ablation study for GA fusion setup.

Runs GA fusion with:
- full feature set (deberta_pred + all heuristics)
- one-at-a-time heuristic removals

Reports validation/test metrics per configuration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


DATA_PATH = Path("experiments/outputs/merged_experiment_data.csv")
OUT_CSV = Path("experiments/outputs/ablation_results.csv")
OUT_TXT = Path("experiments/outputs/ablation_summary.txt")

SEED = 42
POP_SIZE = 80
GENERATIONS = 120
ELITE_COUNT = 8
TOURNAMENT_K = 4
MUTATION_RATE = 0.25
MUTATION_STD = 0.15
INIT_MIN = -2.0
INIT_MAX = 2.0

BASE_COL = "deberta_pred"
HEURISTIC_COLS = [
    "concreteness",
    "specificity",
    "emotional_salience",
    "personal_experience_salience",
    "narrative_event_density",
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
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    lo = int(np.floor(np.min(y_true)))
    hi = int(np.ceil(np.max(y_true)))
    y_true_i = np.clip(np.round(y_true).astype(int), lo, hi)
    y_pred_i = np.clip(np.round(y_pred).astype(int), lo, hi)
    return float(cohen_kappa_score(y_true_i, y_pred_i, weights="quadratic"))


def predict_scores(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w


def tournament_select(rng: np.random.Generator, pop: np.ndarray, fitness: np.ndarray, k: int) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmax(fitness[idx])]].copy()


def run_ga(x_val: np.ndarray, y_val: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_features = x_val.shape[1]
    pop = rng.uniform(INIT_MIN, INIT_MAX, size=(POP_SIZE, n_features))

    best_w = None
    best_fit = -np.inf

    for _ in range(GENERATIONS):
        fitness = np.zeros(POP_SIZE, dtype=float)
        for i in range(POP_SIZE):
            pred = predict_scores(x_val, pop[i])
            fitness[i] = qwk_metric(y_val, pred)

        order = np.argsort(fitness)[::-1]
        if float(fitness[order[0]]) > best_fit:
            best_fit = float(fitness[order[0]])
            best_w = pop[order[0]].copy()

        elites = pop[order[:ELITE_COUNT]].copy()
        next_pop = [e for e in elites]
        while len(next_pop) < POP_SIZE:
            p1 = tournament_select(rng, pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(rng, pop, fitness, TOURNAMENT_K)
            alpha = rng.uniform(0.0, 1.0, size=n_features)
            child = alpha * p1 + (1.0 - alpha) * p2
            for j in range(n_features):
                if rng.random() < MUTATION_RATE:
                    child[j] += rng.normal(0.0, MUTATION_STD)
            next_pop.append(child)
        pop = np.array(next_pop)

    assert best_w is not None
    return best_w


def eval_config(df: pd.DataFrame, feature_cols: list[str], config_name: str) -> dict[str, float | str]:
    val_df = df[df["split"].astype(str).str.lower() == "validation"].copy()
    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()

    x_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df["true_score"].to_numpy(dtype=float)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["true_score"].to_numpy(dtype=float)

    w = run_ga(x_val, y_val, SEED)
    val_pred = predict_scores(x_val, w)
    test_pred = predict_scores(x_test, w)

    val_m = {"qwk": qwk_metric(y_val, val_pred), **regression_metrics(y_val, val_pred)}
    test_m = {"qwk": qwk_metric(y_test, test_pred), **regression_metrics(y_test, test_pred)}

    out: dict[str, float | str] = {
        "config": config_name,
        "features_used": ",".join(feature_cols),
        "validation_qwk": val_m["qwk"],
        "test_qwk": test_m["qwk"],
        "validation_rmse": val_m["rmse"],
        "test_rmse": test_m["rmse"],
        "validation_mae": val_m["mae"],
        "test_mae": test_m["mae"],
        "validation_pearson": val_m["pearson_r"],
        "test_pearson": test_m["pearson_r"],
    }
    for k, v in zip(feature_cols, w):
        out[f"w_{k}"] = float(v)
    return out


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    for c in ["split", "true_score", BASE_COL, *HEURISTIC_COLS]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    results: list[dict[str, float | str]] = []

    full_features = [BASE_COL, *HEURISTIC_COLS]
    results.append(eval_config(df, full_features, "full"))

    for dropped in HEURISTIC_COLS:
        kept = [BASE_COL] + [h for h in HEURISTIC_COLS if h != dropped]
        results.append(eval_config(df, kept, f"drop_{dropped}"))

    out_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    full_row = out_df[out_df["config"] == "full"].iloc[0]
    ablations = out_df[out_df["config"] != "full"].copy()
    ablations["delta_test_qwk_vs_full"] = ablations["test_qwk"] - float(full_row["test_qwk"])
    ablations = ablations.sort_values("delta_test_qwk_vs_full")

    lines = [
        "GA Ablation Summary",
        "===================",
        "",
        f"Full config test QWK: {float(full_row['test_qwk']):.6f}",
        f"Full config test RMSE: {float(full_row['test_rmse']):.6f}",
        "",
        "Ablation impact on test QWK (negative means performance dropped):",
    ]
    for _, r in ablations.iterrows():
        lines.append(
            f"- {r['config']}: test_qwk={float(r['test_qwk']):.6f}, "
            f"delta_vs_full={float(r['delta_test_qwk_vs_full']):.6f}"
        )

    worst = ablations.iloc[0]
    best = ablations.iloc[-1]
    lines += [
        "",
        f"Largest QWK drop: {worst['config']} ({float(worst['delta_test_qwk_vs_full']):.6f})",
        f"Smallest QWK drop / possible gain: {best['config']} ({float(best['delta_test_qwk_vs_full']):.6f})",
        "",
        "Interpretation: heuristic features whose removal causes larger QWK drops contribute more in this GA fusion setup.",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
