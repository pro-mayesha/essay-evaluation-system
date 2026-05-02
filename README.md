# RubriQ Framework

**Automated Essay Scoring with Availability Signals and Genetic Algorithms: The RubriQ Framework**

Official reference implementation and experiment artifacts for the paper above. **Canonical repository:** [github.com/pro-mayesha/rubriq-framework](https://github.com/pro-mayesha/rubriq-framework)

## Overview

Instead of treating essays as plain text sequences only, RubriQ models essays as structured documents. It pairs DeBERTa-based prediction with human-interpretable **availability** (document-level) signals and uses a **genetic algorithm** to learn fusion weights over those signals.

## Problem

Many automated essay scorers:

- Rely mainly on surface-level or sequence patterns  
- Underuse signals such as concreteness, narrative structure, and other cues human raters use when such information is *available* at the document level  

That gap can produce scores that look numerically fine but are misaligned with how humans judge quality.

## Approach (RubriQ)

### 1. Transformer-based scoring

- Model: DeBERTa-v3  
- Trained to predict holistic (or ordinal) essay scores on benchmark data  

### 2. Availability signals

Document-level features that reflect how writing *presents* to a rater (e.g. concreteness, specificity, tone, narrative density). These are fused with the neural model after extraction from essay text.

### 3. Genetic algorithm fusion

- Joins model predictions and engineered features  
- Searches for weights that improve agreement metrics (e.g. QWK)  
- Supports ablation and stability analysis in `experiments/`  

### Training defaults (DeBERTa path)

- Optimizer: AdamW (`adamw_torch`)  
- Learning rate: 5e-6  
- Batch size: 2 (per device train/eval)  
- Epochs: 3  
- Weight decay: 0.01  

## Datasets

- **ASAP** (`asap.csv`): primary training and evaluation pipeline in this repository  
- **Persuade** (`persuade.csv`): included for reference / extensions; the scripted pipeline here is centered on ASAP + exported predictions (see Reproducibility)  

## Results (summary)

Reported metrics and tables are stored under `experiments/outputs/` (CSV, JSON, summary text, figures). Large checkpoints and local training runs are excluded from git (see below).

## Why it matters

RubriQ targets more interpretable scoring, better potential for feedback, and alignment with criteria human raters use—suitable for extensions in feedback systems, admissions support, and educational platforms.

## Repository layout

| Path | Role |
|------|------|
| `train_asap.py`, `train_asap_ordinal.py` | DeBERTa training (regression / ordinal) |
| `eval_asap.py`, `eval_asap_ordinal.py` | Evaluation on the held-out split |
| `experiments/` | Feature extraction, GA (`ga_optimize.py`), fusion (`fuse_scores.py`), validation scripts |
| `experiments/outputs/` | Locked tables, summaries, and figures aligned with the paper’s experiments |
| `VERSION` | Release-style version string; keep in sync with git tags when you mint releases |
| `CITATION.cff` | Machine-readable citation metadata (GitHub **Cite this repository**) |

## Citing this work

Use the **same wording** in your manuscript as in `CITATION.cff` so the paper title and author match exactly.

**Paper (recommended primary citation):**

```bibtex
@article{proma2026rubriq,
  title   = {Automated Essay Scoring with Availability Signals and Genetic Algorithms: The RubriQ Framework},
  author  = {Proma, Mayesha Maliha},
  year    = {2026},
  note    = {Add venue, volume, issue, and pages (or arXiv id) after publication.},
}
```

**Artifact (this repository)—exact line for your manuscript** (same title and author as above; matches [`CITATION.cff`](CITATION.cff)):

> Reference implementation and experiment artifacts: [https://github.com/pro-mayesha/rubriq-framework](https://github.com/pro-mayesha/rubriq-framework), release **v1.0.0**, commit **f421e58** (full SHA `f421e58ba67b7c3b27830298a3e393e92bb1ea65`).

Ready-to-paste variants (LaTeX / plain text) live in [`ARTIFACT_CITATION.md`](ARTIFACT_CITATION.md).

GitHub reads [`CITATION.cff`](CITATION.cff) for the **Cite this repository** button. After you have a DOI (publisher or Zenodo archive), add it under `preferred-citation` / `identifiers` in that file so the repo and PDF stay in sync.

## Reproducibility (what is and is not on GitHub)

**Included in git:** source code, `asap.csv`, `persuade.csv`, and committed outputs under `experiments/outputs/` (summaries, metrics, figures).

**Not included (see `.gitignore`):** local training directories (`results_asap/`, `results_asap_debug/`, etc.), saved model folders (`asap_model/`), and weight file globs (`*.pt`, `*.bin`, …). Re-running the full pipeline requires training/evaluation locally first so scripts that read `results_asap_debug/...` exports can run.

Suggested order: train → eval (writes prediction CSVs) → `experiments/` scripts → compare with `experiments/outputs/`.

## Next steps

- Richer feature extractors (e.g. sentence encoders)  
- Broader prompt sets and cross-prompt evaluation  
- Joint score + feedback generation  
