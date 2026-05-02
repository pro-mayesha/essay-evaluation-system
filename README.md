# RubriQ Framework

**Automated Essay Scoring with Availability Signals and Genetic Algorithms**

This repository holds the code and experiments for **RubriQ**: a hybrid automated essay scoring approach that combines transformer-based scoring, **availability** (document-level) signals, and a **genetic algorithm** to learn fusion weights over those signals.

## Overview

Instead of treating essays as plain text sequences only, RubriQ models essays as structured documents. It pairs DeBERTa-based prediction with human-interpretable availability features and uses a genetic algorithm to combine those sources for stronger agreement with human scores.

## Problem

Many automated essay scorers:

- Rely mainly on surface-level or sequence patterns  
- Underuse signals such as concreteness, narrative structure, and other cues that human raters use when scores are *available* at the document level  

That gap can produce scores that look numerically fine but are misaligned with how humans judge quality.

## Approach (RubriQ)

### 1. Transformer-based scoring

- Model: DeBERTa-v3  
- Trained to predict holistic (or ordinal) essay scores on benchmark data  

### 2. Availability signals

Document-level features that reflect how writing *presents* to a rater (e.g. concreteness, specificity, tone, narrative density). These are the “availability” side of the framework: signals that are observable in the text and available for fusion with the neural model.

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

- **ASAP**: primary training and evaluation for essay scores  
- **Persuade** (where used): additional discourse and writing patterns  

## Results (summary)

- Improved consistency over a DeBERTa-only baseline in reported runs  
- Trade-off: added components (features + GA) for interpretability and controlled fusion  

## Why it matters

RubriQ targets more interpretable scoring, better potential for feedback, and alignment with criteria human raters use—suitable for extensions in feedback systems, admissions support, and educational platforms.

## Repository layout

- `train_asap.py` / `train_asap_ordinal.py` — DeBERTa training (regression / ordinal)  
- `eval_asap.py` / `eval_asap_ordinal.py` — evaluation on the held-out split  
- `experiments/` — feature extraction, GA optimization (`ga_optimize.py`), fusion (`fuse_scores.py`), ablations, validation, and result exports under `experiments/outputs/`  

## Citation

If you use this software or the RubriQ framework, please cite the associated work (see your paper: *Automated Essay Scoring with Availability Signals and Genetic Algorithms: The RubriQ Framework*).

## Next steps

- Richer feature extractors (e.g. sentence encoders)  
- Broader prompt sets and cross-prompt evaluation  
- Joint score + feedback generation  
