"""
Evaluate saved ordinal ASAP model on the held-out test split (same split as train_asap_ordinal.py).

Reports the same global and stratified metrics as eval_asap.py, plus truncation diagnostics.
"""
import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Match train_asap_ordinal.py
CSV_PATH = "asap.csv"
OUTPUT_DIR = "./results_asap_ordinal"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
MAX_LENGTH = 512
SEED = 42
NUM_ORDINAL = 5
ORDINAL_THRESHOLD = 0.5
HOTSPOT_MIN_N = 15


def score_to_ordinal_vec(score) -> list:
    s = int(round(float(score)))
    s = max(1, min(6, s))
    k = s - 1
    return [1.0] * k + [0.0] * (NUM_ORDINAL - k)


def ordinal_logits_to_scores(logits: np.ndarray, threshold: float = ORDINAL_THRESHOLD) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 3:
        logits = logits.reshape(logits.shape[0], -1)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    z = np.clip(logits, -50, 50)
    probs = 1.0 / (1.0 + np.exp(-z))
    return 1.0 + np.sum(probs > threshold, axis=-1)


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if logits.ndim == 3:
        logits = logits.reshape(logits.shape[0], -1)
    pred = ordinal_logits_to_scores(logits, ORDINAL_THRESHOLD)
    # Labels in dataset are 5-dim ordinal targets (matches BCE training)
    if labels.ndim == 1:
        true_s = labels.reshape(-1)
    else:
        true_s = 1.0 + np.sum(labels, axis=-1)
    rmse = float(np.sqrt(np.mean((pred - true_s) ** 2)))
    return {"rmse": rmse}


def subset_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    n = int(len(y_true))
    if n == 0:
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "mse": float("nan"),
            "pearson_r": float("nan"),
            "exact_match_pct": float("nan"),
            "within_0_5_pct": float("nan"),
            "within_1_0_pct": float("nan"),
        }
    out = {
        "n": n,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err**2)),
        "exact_match_pct": float(np.mean(np.round(y_pred) == np.round(y_true)) * 100),
        "within_0_5_pct": float(np.mean(np.abs(err) <= 0.5) * 100),
        "within_1_0_pct": float(np.mean(np.abs(err) <= 1.0) * 100),
    }
    if n > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        out["pearson_r"] = float("nan")
    return out


def stratified_table(eval_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, sub in eval_df.groupby(group_col, dropna=False):
        m = subset_metrics(sub["true_score"].values, sub["predicted_score"].values)
        m[group_col] = key
        rows.append(m)
    out = pd.DataFrame(rows)
    cols = [
        group_col,
        "n",
        "rmse",
        "mae",
        "mse",
        "pearson_r",
        "exact_match_pct",
        "within_0_5_pct",
        "within_1_0_pct",
    ]
    out = out[[c for c in cols if c in out.columns]]
    return out


def truncation_diagnostics(
    texts: pd.Series,
    tokenizer,
    max_length: int,
    length_bin: pd.Series,
) -> dict:
    """
    Tokenize without truncation; count essays whose full token length exceeds max_length.
    """
    model_max = getattr(tokenizer, "model_max_length", None)
    if model_max is None or model_max > 100_000:
        model_max_report = "very_large_or_unset"
    else:
        model_max_report = int(model_max)

    token_lens = []
    truncated_flags = []
    for t in texts.astype(str):
        enc = tokenizer(t, add_special_tokens=True, truncation=False)
        L = len(enc["input_ids"])
        token_lens.append(L)
        truncated_flags.append(L > max_length)

    token_lens = np.array(token_lens, dtype=np.int32)
    truncated_flags = np.array(truncated_flags, dtype=bool)
    n = len(truncated_flags)
    n_trunc = int(truncated_flags.sum())

    by_quartile = []
    df_tmp = pd.DataFrame({"truncated": truncated_flags, "length_bin": length_bin.astype(str).values})
    for bin_name in sorted(df_tmp["length_bin"].unique()):
        sub = df_tmp[df_tmp["length_bin"] == bin_name]
        bn = len(sub)
        bt = int(sub["truncated"].sum())
        by_quartile.append(
            {
                "length_bin": bin_name,
                "n": bn,
                "n_truncated": bt,
                "pct_truncated": float(100.0 * bt / bn) if bn else float("nan"),
            }
        )

    return {
        "tokenizer_model_max_length": model_max_report,
        "eval_max_length": int(max_length),
        "n_essays": n,
        "n_token_length_gt_max_length": n_trunc,
        "pct_truncated": float(100.0 * n_trunc / n) if n else float("nan"),
        "token_length_summary": {
            "min": int(token_lens.min()) if n else None,
            "max": int(token_lens.max()) if n else None,
            "mean": float(token_lens.mean()) if n else float("nan"),
            "median": float(np.median(token_lens)) if n else float("nan"),
        },
        "by_length_quartile": by_quartile,
    }


def main():
    if not os.path.isdir(FINAL_MODEL_DIR):
        raise FileNotFoundError(
            f"Missing {FINAL_MODEL_DIR}. Run train_asap_ordinal.py first."
        )

    df = pd.read_csv(CSV_PATH)
    if "text" not in df.columns or "score" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'score' columns.")

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["text", "score"]).copy()
    df = df[np.isfinite(df["score"])].copy()
    df["text"] = df["text"].astype(str)
    df["score"] = df["score"].astype("float32")
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""].copy()

    if "prompt_id" in df.columns:
        df["prompt_id"] = df["prompt_id"].astype(str)
    elif "prompt" in df.columns:
        df["prompt_id"] = df["prompt"].astype(str)
    else:
        df["prompt_id"] = "unknown"

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    val_end = int(0.9 * n)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    print(f"Test size: {len(test_df)}")

    word_count = test_df["text"].str.split().str.len().astype(int)
    try:
        length_cat = pd.qcut(
            word_count,
            q=4,
            labels=["Q1_shortest", "Q2", "Q3", "Q4_longest"],
            duplicates="drop",
        )
    except ValueError:
        length_cat = pd.Series(["all"] * len(test_df), index=test_df.index)

    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)

    trunc_report = truncation_diagnostics(test_df["text"], tokenizer, MAX_LENGTH, length_cat)

    test_dataset = Dataset.from_pandas(test_df[["text", "score"]], preserve_index=False)

    def preprocess(example):
        encoded = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = score_to_ordinal_vec(example["score"])
        return encoded

    test_dataset = test_dataset.map(preprocess)
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in test_dataset.column_names:
        columns_to_keep.append("token_type_ids")
    test_dataset.set_format(type="torch", columns=columns_to_keep)
    test_dataset = test_dataset.with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        FINAL_MODEL_DIR,
        num_labels=NUM_ORDINAL,
        problem_type="multi_label_classification",
    )
    model = model.float()

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "eval_only"),
        per_device_eval_batch_size=2,
        fp16=False,
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(model=model, args=args, compute_metrics=compute_metrics)
    results = trainer.evaluate(test_dataset)

    print("\n--- Hugging Face eval (ordinal -> RMSE on scalar scores) ---")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    pred_out = trainer.predict(test_dataset)
    # Same as baseline eval: regress against raw CSV scores (float), not discretized ordinal sum
    y_true = test_df["score"].values.astype(np.float64).reshape(-1)
    raw = pred_out.predictions
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 3:
        raw = raw.reshape(raw.shape[0], -1)
    y_pred = ordinal_logits_to_scores(raw, ORDINAL_THRESHOLD).reshape(-1)

    err = y_pred - y_true

    tlen = []
    for t in test_df["text"].astype(str):
        enc = tokenizer(t, add_special_tokens=True, truncation=False)
        tlen.append(len(enc["input_ids"]))

    eval_df = pd.DataFrame(
        {
            "true_score": y_true,
            "predicted_score": y_pred,
            "error": err,
            "abs_error": np.abs(err),
            "prompt_id": test_df["prompt_id"].values,
            "word_count": word_count.values,
            "length_bin": length_cat.astype(str).values,
            "true_score_level": np.clip(np.round(y_true).astype(int), 1, 6),
            "token_count_full": tlen,
        }
    )
    eval_df["truncated_for_eval_max_length"] = eval_df["token_count_full"] > MAX_LENGTH

    global_m = subset_metrics(y_true, y_pred)
    global_m.update(
        {
            "score_range_observed": [float(np.min(y_true)), float(np.max(y_true))],
            "eval_loss_hf": float(results.get("eval_loss", float("nan"))),
            "eval_rmse_hf": float(results.get("eval_rmse", float("nan"))),
            "ordinal_num_thresholds": NUM_ORDINAL,
            "ordinal_threshold": ORDINAL_THRESHOLD,
        }
    )

    by_prompt = stratified_table(eval_df, "prompt_id")
    by_score = stratified_table(eval_df, "true_score_level")
    by_length = stratified_table(eval_df, "length_bin")

    wc_stats = (
        eval_df.groupby("length_bin", dropna=False)["word_count"]
        .agg(word_count_min="min", word_count_max="max", word_count_mean="mean")
        .reset_index()
    )
    by_length = by_length.merge(wc_stats, on="length_bin", how="left")

    out_dir = os.path.join(OUTPUT_DIR, "test_eval_results")
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "test_metrics.json")
    csv_path = os.path.join(out_dir, "test_predictions.csv")
    prompt_csv = os.path.join(out_dir, "stratified_by_prompt.csv")
    score_csv = os.path.join(out_dir, "stratified_by_score_level.csv")
    length_csv = os.path.join(out_dir, "stratified_by_length.csv")
    report_path = os.path.join(out_dir, "evaluation_report.json")
    trunc_path = os.path.join(out_dir, "truncation_diagnostics.json")
    trunc_csv_path = os.path.join(out_dir, "truncation_by_quartile.csv")

    def _json_default(o):
        if isinstance(o, (np.integer, np.floating)):
            return float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(global_m, f, indent=2, default=_json_default)

    with open(trunc_path, "w", encoding="utf-8") as f:
        json.dump(trunc_report, f, indent=2, default=_json_default)

    pd.DataFrame(trunc_report["by_length_quartile"]).to_csv(trunc_csv_path, index=False)

    eval_df.to_csv(csv_path, index=False)
    by_prompt.sort_values("prompt_id").to_csv(prompt_csv, index=False)
    by_score.sort_values("true_score_level").to_csv(score_csv, index=False)
    by_length.sort_values("length_bin").to_csv(length_csv, index=False)

    def hotspots(tbl: pd.DataFrame, key: str):
        h = tbl[tbl["n"] >= HOTSPOT_MIN_N].copy()
        if h.empty:
            return {"highest_rmse": [], "lowest_exact_match": []}
        worst_rmse = h.nlargest(5, "rmse")[[key, "n", "rmse", "mae", "exact_match_pct"]]
        worst_exact = h.nsmallest(5, "exact_match_pct")[
            [key, "n", "rmse", "mae", "exact_match_pct"]
        ]
        return {
            "highest_rmse": worst_rmse.to_dict(orient="records"),
            "lowest_exact_match": worst_exact.to_dict(orient="records"),
        }

    full_report = {
        "global": global_m,
        "truncation_diagnostics": trunc_report,
        "stratified_hotspots": {
            "by_prompt_id": hotspots(by_prompt, "prompt_id"),
            "by_score_level": hotspots(by_score, "true_score_level"),
            "by_length_bin": hotspots(by_length, "length_bin"),
        },
        "files": {
            "predictions": csv_path,
            "by_prompt": prompt_csv,
            "by_score_level": score_csv,
            "by_length": length_csv,
            "global_metrics": metrics_path,
            "truncation_diagnostics": trunc_path,
            "truncation_by_quartile_csv": trunc_csv_path,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=_json_default)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n========== GLOBAL (test set, ordinal model) ==========")
    print(
        f"  N = {global_m['n']}, RMSE = {global_m['rmse']:.4f}, MAE = {global_m['mae']:.4f}, "
        f"r = {global_m['pearson_r']:.4f}"
    )
    print(f"  Within ±0.5: {global_m['within_0_5_pct']:.1f}%  |  Within ±1.0: {global_m['within_1_0_pct']:.1f}%")
    print(f"  Exact match (rounded) = {global_m['exact_match_pct']:.1f}%")

    print("\n----- Truncation (full tokenize, no truncation) -----")
    print(f"  eval_max_length = {trunc_report['eval_max_length']}")
    print(f"  tokenizer.model_max_length = {trunc_report['tokenizer_model_max_length']}")
    print(
        f"  essays with len(input_ids) > max_length: {trunc_report['n_token_length_gt_max_length']} "
        f"({trunc_report['pct_truncated']:.2f}%)"
    )
    print("  By word-count quartile:")
    for row in trunc_report["by_length_quartile"]:
        print(
            f"    {row['length_bin']}: n={row['n']}, truncated={row['n_truncated']} "
            f"({row['pct_truncated']:.2f}%)"
        )

    print("\n----- By prompt_id -----")
    print(by_prompt.sort_values("prompt_id").to_string(index=False))

    print("\n----- By true score level (1–6) -----")
    print(by_score.sort_values("true_score_level").to_string(index=False))

    print("\n----- By essay length (word-count quartile) -----")
    print(by_length.sort_values("length_bin").to_string(index=False))

    print("\n========== HOTSPOTS (n >= %d) ==========" % HOTSPOT_MIN_N)
    for name, tbl, kcol in [
        ("prompt_id", by_prompt, "prompt_id"),
        ("score level", by_score, "true_score_level"),
        ("length bin", by_length, "length_bin"),
    ]:
        hp = hotspots(tbl, kcol)
        print(f"\n  [{name}] Highest RMSE:")
        for row in hp["highest_rmse"]:
            print(f"    {row}")
        print(f"  [{name}] Lowest exact-match %:")
        for row in hp["lowest_exact_match"]:
            print(f"    {row}")

    print(f"\n  Written: {report_path}")
    print(f"  Written: {trunc_path}, {trunc_csv_path}")
    print("========================================\n")


if __name__ == "__main__":
    main()
