# Ensemble Models Results

## Overview
This document summarizes the results from ensemble model development and evaluation.

## Ensemble Methods

1. **Averaging Ensemble**: Simple average of all base model predictions
2. **Weighted Averaging**: Optimized weights learned from validation set
3. **Stacking Ensemble**: Meta-learner (Ridge Regression) trained on base model predictions
4. **Blending Ensemble**: Linear regression to learn optimal blend weights

## Methodology

### Averaging Ensemble
- Simple mean of all base model predictions
- No training required
- Fast and interpretable

### Weighted Averaging
- Weights optimized using scipy.optimize
- Minimizes MSE on validation set
- Weights sum to 1

### Stacking Ensemble
- Base models: Random Forest, XGBoost, LightGBM, CatBoost
- Meta-learner: Ridge Regression
- Meta-features: Base model predictions on validation set
- Meta-learner trained on meta-features

### Blending Ensemble
- Similar to stacking but simpler
- Linear regression to learn blend weights
- Trained on validation set predictions

## Evaluation

All ensemble methods evaluated on test set (unseen data):
- MAE, RMSE, R², MAPE

## Expected Performance

Ensemble methods typically:
- Outperform individual models
- Reduce variance
- Improve generalization
- Weighted averaging often performs best

## Usage

To train and evaluate ensemble models:
```bash
python scripts/modeling/ensemble.py
```

## Output Files

- Models: `models/saved/ensemble_models.pkl`
- Results: `models/saved/ensemble_results.pkl`

## Best Model Selection

The best ensemble method is selected based on:
- Lowest RMSE on test set
- Good balance of MAE and RMSE
- Generalization to unseen data

## Notes

- Ensemble methods combine strengths of individual models
- Weighted averaging is often the best balance of performance and simplicity
- Stacking can capture non-linear interactions between models
- Results depend on base model quality


