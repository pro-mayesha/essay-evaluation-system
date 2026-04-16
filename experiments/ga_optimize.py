"""Genetic algorithm optimization for fusion weights.

Optimizes a weighted fusion of:
- deberta_pred
- concreteness
- specificity
- emotional_salience
- personal_experience_salience
- narrative_event_density

Optimization uses validation split only.
Final metrics are reported on test split only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


DATA_PATH = Path("experiments/outputs/merged_experiment_data.csv")
OUT_WEIGHTS = Path("experiments/outputs/best_ga_weights.json")
OUT_RESULTS = Path("experiments/outputs/ga_results.json")
OUT_SUMMARY = Path("experiments/outputs/ga_summary.txt")
OUT_STABILITY_RUNS = Path("experiments/outputs/ga_stability_runs.csv")
OUT_STABILITY_SUMMARY = Path("experiments/outputs/ga_stability_summary.txt")

SEED = 42
POP_SIZE = 80
GENERATIONS = 120
ELITE_COUNT = 8
TOURNAMENT_K = 4
MUTATION_RATE = 0.25
MUTATION_STD = 0.15
INIT_MIN = -2.0
INIT_MAX = 2.0
STABILITY_SEEDS = [11, 22, 33, 44, 55]

FEATURE_COLS = [
    "deberta_pred",
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


def fitness_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    qwk = qwk_metric(y_true, y_pred)
    reg = regression_metrics(y_true, y_pred)
    # Primary objective: maximize QWK. RMSE included for reporting.
    return qwk, {"qwk": qwk, **reg}


def tournament_select(rng: np.random.Generator, pop: np.ndarray, fitness: np.ndarray, k: int) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    best_local = idx[np.argmax(fitness[idx])]
    return pop[best_local].copy()


def crossover(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    alpha = rng.uniform(0.0, 1.0, size=p1.shape[0])
    return alpha * p1 + (1.0 - alpha) * p2


def mutate(rng: np.random.Generator, child: np.ndarray) -> np.ndarray:
    out = child.copy()
    for i in range(out.shape[0]):
        if rng.random() < MUTATION_RATE:
            out[i] += rng.normal(0.0, MUTATION_STD)
    return out


def run_ga(x_val: np.ndarray, y_val: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n_features = x_val.shape[1]

    pop = rng.uniform(INIT_MIN, INIT_MAX, size=(POP_SIZE, n_features))
    history: list[dict] = []

    best_w = None
    best_fit = -np.inf
    best_metrics = None

    for gen in range(GENERATIONS):
        fitness = np.zeros(POP_SIZE, dtype=float)
        metrics_cache: list[dict[str, float]] = []
        for i in range(POP_SIZE):
            pred = predict_scores(x_val, pop[i])
            fit, m = fitness_from_predictions(y_val, pred)
            fitness[i] = fit
            metrics_cache.append(m)

        order = np.argsort(fitness)[::-1]
        elites = pop[order[:ELITE_COUNT]].copy()
        gen_best_idx = order[0]
        gen_best_fit = float(fitness[gen_best_idx])
        gen_best_metrics = metrics_cache[gen_best_idx]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_w = pop[gen_best_idx].copy()
            best_metrics = gen_best_metrics

        history.append(
            {
                "generation": gen,
                "best_qwk": gen_best_metrics["qwk"],
                "best_rmse": gen_best_metrics["rmse"],
                "mean_qwk": float(np.mean(fitness)),
            }
        )

        next_pop = [e for e in elites]
        while len(next_pop) < POP_SIZE:
            p1 = tournament_select(rng, pop, fitness, TOURNAMENT_K)
            p2 = tournament_select(rng, pop, fitness, TOURNAMENT_K)
            child = crossover(rng, p1, p2)
            child = mutate(rng, child)
            next_pop.append(child)
        pop = np.array(next_pop)

    assert best_w is not None
    assert best_metrics is not None
    return best_w, {"best_validation_metrics": best_metrics, "history": history}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    for c in ["essay_id", "split", "true_score", *FEATURE_COLS]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    val_df = df[df["split"].astype(str).str.lower() == "validation"].copy()
    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()

    x_val = val_df[FEATURE_COLS].to_numpy(dtype=float)
    y_val = val_df["true_score"].to_numpy(dtype=float)
    x_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
    y_test = test_df["true_score"].to_numpy(dtype=float)

    best_w, ga_info = run_ga(x_val, y_val, SEED)

    val_pred = predict_scores(x_val, best_w)
    test_pred = predict_scores(x_test, best_w)
    val_metrics = {"qwk": qwk_metric(y_val, val_pred), **regression_metrics(y_val, val_pred)}
    test_metrics = {"qwk": qwk_metric(y_test, test_pred), **regression_metrics(y_test, test_pred)}

    weights_dict = {k: float(v) for k, v in zip(FEATURE_COLS, best_w)}
    weights_payload = {
        "objective": "maximize_validation_qwk",
        "feature_order": FEATURE_COLS,
        "weights": weights_dict,
    }

    results_payload = {
        "setup": {
            "seed": SEED,
            "population_size": POP_SIZE,
            "generations": GENERATIONS,
            "elite_count": ELITE_COUNT,
            "tournament_k": TOURNAMENT_K,
            "mutation_rate": MUTATION_RATE,
            "mutation_std": MUTATION_STD,
            "init_range": [INIT_MIN, INIT_MAX],
            "objective": "validation_qwk",
        },
        "data": {
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        },
        "best_weights": weights_dict,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ga_history": ga_info["history"],
    }

    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_WEIGHTS, "w", encoding="utf-8") as f:
        json.dump(weights_payload, f, indent=2)
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    summary = [
        "GA Fusion Optimization Summary",
        "=============================",
        "",
        f"Objective: maximize validation QWK",
        f"Rows: validation={len(val_df)}, test={len(test_df)}",
        f"GA config: pop={POP_SIZE}, generations={GENERATIONS}, seed={SEED}",
        "",
        "Best weights:",
    ]
    for k in FEATURE_COLS:
        summary.append(f"  {k}: {weights_dict[k]:.6f}")
    summary += [
        "",
        "Validation metrics:",
        f"  QWK={val_metrics['qwk']:.6f}, RMSE={val_metrics['rmse']:.6f}, MAE={val_metrics['mae']:.6f}, r={val_metrics['pearson_r']:.6f}",
        "Test metrics:",
        f"  QWK={test_metrics['qwk']:.6f}, RMSE={test_metrics['rmse']:.6f}, MAE={test_metrics['mae']:.6f}, r={test_metrics['pearson_r']:.6f}",
        "",
        f"Saved: {OUT_WEIGHTS}",
        f"Saved: {OUT_RESULTS}",
    ]
    OUT_SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_WEIGHTS}")
    print(f"Wrote {OUT_RESULTS}")
    print(f"Wrote {OUT_SUMMARY}")
    print("Best validation QWK:", f"{val_metrics['qwk']:.6f}")
    print("Test QWK:", f"{test_metrics['qwk']:.6f}")

    # Stability check: same GA setup, only seed changes.
    stability_rows: list[dict[str, float | int]] = []
    for seed in STABILITY_SEEDS:
        w_seed, _ = run_ga(x_val, y_val, seed)
        val_pred_seed = predict_scores(x_val, w_seed)
        test_pred_seed = predict_scores(x_test, w_seed)
        val_m_seed = {"qwk": qwk_metric(y_val, val_pred_seed), **regression_metrics(y_val, val_pred_seed)}
        test_m_seed = {"qwk": qwk_metric(y_test, test_pred_seed), **regression_metrics(y_test, test_pred_seed)}

        row: dict[str, float | int] = {
            "seed": int(seed),
            "validation_qwk": float(val_m_seed["qwk"]),
            "test_qwk": float(test_m_seed["qwk"]),
            "validation_rmse": float(val_m_seed["rmse"]),
            "test_rmse": float(test_m_seed["rmse"]),
        }
        for k, v in zip(FEATURE_COLS, w_seed):
            row[f"w_{k}"] = float(v)
        stability_rows.append(row)

    stability_df = pd.DataFrame(stability_rows)
    OUT_STABILITY_RUNS.parent.mkdir(parents=True, exist_ok=True)
    stability_df.to_csv(OUT_STABILITY_RUNS, index=False)

    metric_cols = ["validation_qwk", "test_qwk", "validation_rmse", "test_rmse"]
    means = stability_df[metric_cols].mean()
    stds = stability_df[metric_cols].std(ddof=1)
    # Heuristic threshold: very small spread => stable, moderate => somewhat stable, large => variable.
    qwk_std = float(stds["test_qwk"])
    rmse_std = float(stds["test_rmse"])
    if qwk_std < 0.005 and rmse_std < 0.01:
        stability_note = "GA appears stable across seeds (low variability in test QWK/RMSE)."
    elif qwk_std < 0.015 and rmse_std < 0.03:
        stability_note = "GA shows moderate variability across seeds but remains reasonably consistent."
    else:
        stability_note = "GA appears highly variable across seeds; consider larger population/generations."

    lines = [
        "GA Stability Summary (5 seeds)",
        "=============================",
        "",
        f"Seeds: {STABILITY_SEEDS}",
        f"Fixed config: pop={POP_SIZE}, generations={GENERATIONS}, elite={ELITE_COUNT}, mutation_rate={MUTATION_RATE}, mutation_std={MUTATION_STD}",
        "",
        "Per-run metrics are saved in ga_stability_runs.csv.",
        "",
        "Metric mean ± std:",
        f"  validation_qwk: {means['validation_qwk']:.6f} ± {stds['validation_qwk']:.6f}",
        f"  test_qwk:       {means['test_qwk']:.6f} ± {stds['test_qwk']:.6f}",
        f"  validation_rmse:{means['validation_rmse']:.6f} ± {stds['validation_rmse']:.6f}",
        f"  test_rmse:      {means['test_rmse']:.6f} ± {stds['test_rmse']:.6f}",
        "",
        f"Stability note: {stability_note}",
    ]
    OUT_STABILITY_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_STABILITY_RUNS}")
    print(f"Wrote {OUT_STABILITY_SUMMARY}")


if __name__ == "__main__":
    main()
