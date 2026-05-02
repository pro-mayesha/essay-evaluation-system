from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


IMPROVED_PATH = Path("experiments/outputs/error_analysis_improved.csv")
WORSENED_PATH = Path("experiments/outputs/error_analysis_worsened.csv")
OUT_DIR = Path("experiments/outputs")
OUT_PNG = OUT_DIR / "essay_error_heatmap.png"
OUT_PDF = OUT_DIR / "essay_error_heatmap.pdf"


def main() -> None:
    # 1–4. Load CSVs, take top 20 rows from each, add group label, and combine.
    improved_df = pd.read_csv(IMPROVED_PATH).head(20).copy()
    worsened_df = pd.read_csv(WORSENED_PATH).head(20).copy()

    improved_df["group"] = "Improved"
    worsened_df["group"] = "Worsened"

    combined = pd.concat([improved_df, worsened_df], axis=0, ignore_index=True)

    # 5. Compute absolute errors for each model vs true_score.
    true = combined["true_score"].to_numpy(dtype=float)
    a_pred = combined["model_a_pred"].to_numpy(dtype=float)
    c_pred = combined["model_c_pred"].to_numpy(dtype=float)
    d_pred = combined["model_d_pred"].to_numpy(dtype=float)

    model_a_abs_error = np.abs(a_pred - true)
    model_c_abs_error = np.abs(c_pred - true)
    model_d_abs_error = np.abs(d_pred - true)

    combined["model_a_abs_error"] = model_a_abs_error
    combined["model_c_abs_error"] = model_c_abs_error
    combined["model_d_abs_error"] = model_d_abs_error

    # 6. Error change relative to Model A (negative => closer to true_score).
    model_c_delta = model_c_abs_error - model_a_abs_error
    model_d_delta = model_d_abs_error - model_a_abs_error

    combined["model_c_delta"] = model_c_delta
    combined["model_d_delta"] = model_d_delta

    # 7. Matrix for heatmap: rows = essay_id (improved first, then worsened).
    row_labels = combined["essay_id"].astype(str).tolist()
    heatmap_data = pd.DataFrame(
        {
            "Model C vs A": combined["model_c_delta"].to_numpy(dtype=float),
            "Model D vs A": combined["model_d_delta"].to_numpy(dtype=float),
        },
        index=row_labels,
    )

    # 8. Plot heatmap (seaborn).
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6.0, 8.0))

    vmin = float(np.nanmin(heatmap_data.to_numpy(dtype=float)))
    vmax = float(np.nanmax(heatmap_data.to_numpy(dtype=float)))
    max_abs = max(abs(vmin), abs(vmax))
    if max_abs == 0.0:
        max_abs = 1.0

    hm = sns.heatmap(
        heatmap_data,
        cmap="RdBu",
        center=0.0,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Absolute-error change vs Model A (negative = closer)"},
    )

    # Separator line between Improved and Worsened groups.
    n_improved = len(improved_df)
    ax.axhline(n_improved, color="black", linewidth=1.0)

    ax.set_title("Essay-level absolute-error change vs baseline")
    ax.set_xlabel("Model comparison")
    ax.set_ylabel("Essay ID")

    # Keep y-axis labels but downsize for readability.
    ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax.set_yticklabels(heatmap_data.index, fontsize=6)

    # Group markers on the left side (non-clutter, no essay ranking implied).
    ax.text(-0.5, n_improved / 2, "Improved", ha="right", va="center", fontsize=7, color="black")
    ax.text(-0.5, n_improved + (len(heatmap_data) - n_improved) / 2, "Worsened", ha="right", va="center", fontsize=7, color="black")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()

    # 11. Summary statistics.
    mean_c = float(model_c_delta.mean())
    mean_d = float(model_d_delta.mean())
    n_c_better = int((model_c_delta < 0).sum())
    n_d_better = int((model_d_delta < 0).sum())

    print("Essay-level absolute-error change relative to Model A")
    print(f"  Model C vs A: mean delta = {mean_c:.6f}")
    print(f"  Model D vs A: mean delta = {mean_d:.6f}")
    print(f"  Model C: # essays closer than Model A (delta < 0) = {n_c_better}")
    print(f"  Model D: # essays closer than Model A (delta < 0) = {n_d_better}")
    print(f"Saved heatmaps to {OUT_PNG} and {OUT_PDF}")


if __name__ == "__main__":
    main()

