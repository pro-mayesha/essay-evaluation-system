"""Evaluate saved ASAP regression model on the held-out test split (same split as train_asap.py).

Global metrics plus stratified breakdowns for paper-style analysis:
  - by prompt_id
  - by true score level (1–6)
  - by essay length (word-count quartiles)
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

# Must match train_asap.py
CSV_PATH = "asap.csv"
OUTPUT_DIR = "./results_asap_debug"
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_model")
MAX_LENGTH = 512
SEED = 42
# Minimum n to include a stratum in “hotspot” warnings (avoid noisy small cells)
HOTSPOT_MIN_N = 15


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.squeeze(predictions)
    labels = np.squeeze(labels)
    rmse = float(np.sqrt(np.mean((labels - predictions) ** 2)))
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
    # stable column order
    cols = [group_col, "n", "rmse", "mae", "mse", "pearson_r", "exact_match_pct", "within_0_5_pct", "within_1_0_pct"]
    out = out[[c for c in cols if c in out.columns]]
    return out


def main():
    if not os.path.isdir(FINAL_MODEL_DIR):
        raise FileNotFoundError(
            f"Missing {FINAL_MODEL_DIR}. Train first or set OUTPUT_DIR to match training."
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

    # Stable ID before shuffle (original row index in the cleaned frame)
    df["essay_id"] = df.index.to_numpy()

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    print(f"Test size: {len(test_df)}")

    test_dataset = Dataset.from_pandas(test_df[["text", "score"]], preserve_index=False)
    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)

    def preprocess(example):
        encoded = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = np.float32(example["score"])
        return encoded

    test_dataset = test_dataset.map(preprocess)
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in test_dataset.column_names:
        columns_to_keep.append("token_type_ids")
    test_dataset.set_format(type="torch", columns=columns_to_keep)
    test_dataset = test_dataset.with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_DIR)
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

    print("\n--- Hugging Face eval (global) ---")
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    pred_out = trainer.predict(test_dataset)
    y_true = np.asarray(pred_out.label_ids, dtype=np.float64).reshape(-1)
    raw = pred_out.predictions
    if raw.ndim > 1:
        y_pred = np.asarray(raw.squeeze(-1), dtype=np.float64).reshape(-1)
    else:
        y_pred = np.asarray(raw, dtype=np.float64).reshape(-1)

    def _length_bins_for_split(split_df: pd.DataFrame) -> pd.Series:
        wc = split_df["text"].str.split().str.len().astype(int)
        try:
            return pd.qcut(
                wc,
                q=4,
                labels=["Q1_shortest", "Q2", "Q3", "Q4_longest"],
                duplicates="drop",
            ).astype(str)
        except ValueError:
            return pd.Series(["all"] * len(split_df), index=split_df.index)

    def _enriched_export(split_df: pd.DataFrame, y_true_arr: np.ndarray, y_pred_arr: np.ndarray, split_name: str) -> pd.DataFrame:
        y_true_arr = np.asarray(y_true_arr, dtype=np.float64).reshape(-1)
        y_pred_arr = np.asarray(y_pred_arr, dtype=np.float64).reshape(-1)
        err = y_pred_arr - y_true_arr
        word_count = split_df["text"].str.split().str.len().astype(int)
        length_cat = _length_bins_for_split(split_df)
        return pd.DataFrame(
            {
                "essay_id": split_df["essay_id"].values,
                "split": split_name,
                "text": split_df["text"].values,
                "true_score": y_true_arr,
                "deberta_pred": y_pred_arr,
                "error": err,
                "abs_error": np.abs(err),
                "prompt_id": split_df["prompt_id"].values,
                "word_count": word_count.values,
                "length_bin": length_cat.values,
                "true_score_level": np.clip(np.round(y_true_arr).astype(int), 1, 6),
            }
        )

    err = y_pred - y_true
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
        }
    )

    global_m = subset_metrics(y_true, y_pred)
    global_m.update(
        {
            "score_range_observed": [float(np.min(y_true)), float(np.max(y_true))],
            "eval_loss_hf": float(results.get("eval_loss", float("nan"))),
            "eval_rmse_hf": float(results.get("eval_rmse", float("nan"))),
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
    test_enriched_path = os.path.join(out_dir, "test_predictions_with_ids.csv")
    val_enriched_path = os.path.join(out_dir, "val_predictions_with_ids.csv")
    prompt_csv = os.path.join(out_dir, "stratified_by_prompt.csv")
    score_csv = os.path.join(out_dir, "stratified_by_score_level.csv")
    length_csv = os.path.join(out_dir, "stratified_by_length.csv")
    report_path = os.path.join(out_dir, "evaluation_report.json")

    def _json_default(o):
        if isinstance(o, (np.integer, np.floating)):
            return float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(global_m, f, indent=2, default=_json_default)

    eval_df.to_csv(csv_path, index=False)

    val_dataset = Dataset.from_pandas(val_df[["text", "score"]], preserve_index=False)
    val_dataset = val_dataset.map(preprocess)
    val_dataset.set_format(type="torch", columns=columns_to_keep)
    val_dataset = val_dataset.with_format("torch")
    val_pred_out = trainer.predict(val_dataset)
    y_true_val = np.asarray(val_pred_out.label_ids, dtype=np.float64).reshape(-1)
    raw_val = val_pred_out.predictions
    if raw_val.ndim > 1:
        y_pred_val = np.asarray(raw_val.squeeze(-1), dtype=np.float64).reshape(-1)
    else:
        y_pred_val = np.asarray(raw_val, dtype=np.float64).reshape(-1)

    _enriched_export(val_df, y_true_val, y_pred_val, "validation").to_csv(val_enriched_path, index=False)
    _enriched_export(test_df, y_true, y_pred, "test").to_csv(test_enriched_path, index=False)

    by_prompt.sort_values("prompt_id").to_csv(prompt_csv, index=False)
    by_score.sort_values("true_score_level").to_csv(score_csv, index=False)
    by_length.sort_values("length_bin").to_csv(length_csv, index=False)

    # Hotspots: where RMSE is worst and exact match worst (among n >= HOTSPOT_MIN_N)
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
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, default=_json_default)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n========== GLOBAL (test set) ==========")
    print(f"  N = {global_m['n']}, RMSE = {global_m['rmse']:.4f}, MAE = {global_m['mae']:.4f}, r = {global_m['pearson_r']:.4f}")
    print(f"  Exact match (rounded) = {global_m['exact_match_pct']:.1f}%")

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
    print(f"  Written: {prompt_csv}, {score_csv}, {length_csv}")
    print(f"  Written: {val_enriched_path}, {test_enriched_path}")
    print("========================================\n")


if __name__ == "__main__":
    main()
