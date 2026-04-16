"""
Ordinal (cumulative threshold) training for ASAP scores 1–6 — experimental path.

Baseline `train_asap.py` is unchanged. This script uses 5 binary targets per essay
(BCEWithLogitsLoss) and saves to ./results_asap_ordinal/.

Gold score s in {1..6} -> labels [1]*(s-1) + [0]*(6-s) (length 5).
"""
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random

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

MODEL_NAME = "microsoft/deberta-v3-base"
CSV_PATH = "asap.csv"
OUTPUT_DIR = "./results_asap_ordinal"
MAX_LENGTH = 512
SEED = 42
NUM_ORDINAL = 5
# Sigmoid threshold for decoding (match eval_asap_ordinal.py)
ORDINAL_THRESHOLD = 0.5


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_to_ordinal_vec(score) -> list:
    """ASAP 1–6 -> 5 cumulative threshold labels (float 0/1)."""
    s = int(round(float(score)))
    s = max(1, min(6, s))
    k = s - 1
    return [1.0] * k + [0.0] * (NUM_ORDINAL - k)


def ordinal_logits_to_scores(logits: np.ndarray, threshold: float = ORDINAL_THRESHOLD) -> np.ndarray:
    """(N, 5) logits -> predicted scores in [1, 6]."""
    logits = np.asarray(logits, dtype=np.float64)
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
    true_s = 1.0 + np.sum(labels, axis=-1)
    rmse = float(np.sqrt(np.mean((pred - true_s) ** 2)))
    return {"rmse": rmse}


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading CSV...")
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

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nTrain size: {len(train_df)}")
    print(f"Valid size: {len(val_df)}")
    print(f"Test size:  {len(test_df)}")

    train_dataset = Dataset.from_pandas(train_df[["text", "score"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[["text", "score"]], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[["text", "score"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(example):
        encoded = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = score_to_ordinal_vec(example["score"])
        return encoded

    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(preprocess)
    val_dataset = val_dataset.map(preprocess)
    test_dataset = test_dataset.map(preprocess)

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_dataset.column_names:
        columns_to_keep.append("token_type_ids")

    train_dataset.set_format(type="torch", columns=columns_to_keep)
    val_dataset.set_format(type="torch", columns=columns_to_keep)
    test_dataset.set_format(type="torch", columns=columns_to_keep)
    train_dataset = train_dataset.with_format("torch")
    val_dataset = val_dataset.with_format("torch")
    test_dataset = test_dataset.with_format("torch")

    # 5 logits + BCEWithLogitsLoss (multi-label classification path in HF)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_ORDINAL,
        problem_type="multi_label_classification",
    )
    model = model.float()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting ordinal (multi-label BCE) training...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("Test results:", test_results)

    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    print(f"\nSaving final model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
