import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import mean_squared_error
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


MODEL_NAME = "microsoft/deberta-v3-base"
CSV_PATH = "asap.csv"
OUTPUT_DIR = "./results_asap_debug"
MAX_LENGTH = 512
SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.squeeze(predictions)
    labels = np.squeeze(labels)

    # Avoid `mean_squared_error(..., squared=False)` since some sklearn
    # versions don’t support the `squared` kwarg.
    rmse = float(np.sqrt(np.mean((labels - predictions) ** 2)))
    return {"rmse": rmse}


def main():
    set_seed(SEED)

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("Original columns:", df.columns.tolist())

    if "text" not in df.columns or "score" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'score' columns.")

    # Clean data
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["text", "score"]).copy()
    df = df[np.isfinite(df["score"])].copy()
    df["text"] = df["text"].astype(str)
    df["score"] = df["score"].astype("float32")

    # Optional: strip empty texts
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""].copy()

    print("\nAfter cleaning:")
    print(df.dtypes)
    print(df[["text", "score"]].head())
    print("\nScore summary:")
    print(df["score"].describe())
    print("Any NaN in score:", df["score"].isna().any())
    print("All finite in score:", np.isfinite(df["score"]).all())

    assert df["text"].isna().sum() == 0, "Found missing text values."
    assert df["score"].isna().sum() == 0, "Found missing score values."
    assert np.isfinite(df["score"]).all(), "Found non-finite score values."

    # Shuffle once before split
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Split: 80 / 10 / 10
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
            max_length=512,
        )
        encoded["labels"] = np.float32(example["score"])
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

    # Ensure Hugging Face Dataset returns torch tensors.
    train_dataset = train_dataset.with_format("torch")
    val_dataset = val_dataset.with_format("torch")
    test_dataset = test_dataset.with_format("torch")

    print("\nSample tokenized item:")
    print(train_dataset[0])
    print("Sample label:", train_dataset[0]["labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
    )
    # Leave `problem_type` unset for `num_labels==1` so HF's regression path
    # aligns logits dtype with labels (important on MPS).
    model = model.float()

    # MPS-safe / stable settings
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

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("Test results:", test_results)

    print("\nSaving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

    print("\nDone.")


if __name__ == "__main__":
    main()