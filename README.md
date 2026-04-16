# Essay Evaluation System

## Overview
This project builds an advanced essay evaluation system that goes beyond treating essays as plain text.

Instead of relying only on sequence-based scoring, the system analyzes essays as structured documents. It combines transformer-based modeling with human-inspired features to better capture quality, clarity, and narrative strength.

## Problem
Most automated essay scoring systems:
- Treat essays as simple text sequences  
- Focus mainly on surface-level patterns  
- Miss deeper signals like storytelling, specificity, and structure  

This leads to scores that may be numerically accurate but not truly aligned with human judgment.

## Approach
We propose a hybrid system that combines:

### 1. Transformer-based scoring
- Model: DeBERTa-v3  
- Trained on essay datasets to predict holistic scores  

### 2. Document-level features
Inspired by how humans evaluate writing:
- Concreteness  
- Specificity  
- Emotional tone  
- Personal experience  
- Narrative event density  

These features capture deeper qualities of writing beyond grammar or length.

### 3. Genetic Algorithm (GA) fusion
- Combines model predictions and engineered features  
- Learns optimal weights for each signal  
- Improves overall scoring performance and robustness  

## Datasets
- ASAP: used for training and evaluation of essay scores  
- Persuade: used to capture discourse structure and writing patterns  

## Results
- Improved consistency over baseline transformer model  
- Better alignment with human-like evaluation signals  
- Trade-off: slight increase in complexity for improved interpretability  

## Why this matters
This system moves toward:
- More interpretable AI scoring  
- Better feedback for students  
- Stronger alignment with real evaluation criteria  

It can be extended to:
- Essay feedback systems  
- Admission support tools  
- Educational platforms  

## Project Structure
- `models/` → transformer model and training  
- `features/` → document-level feature extraction  
- `ga/` → genetic algorithm fusion logic  
- `data/` → dataset processing (ASAP, Persuade)  

## Next Steps
- Improve feature extraction using NLP models (e.g., SBERT)  
- Expand evaluation across multiple prompts  
- Integrate feedback generation alongside scoring  
