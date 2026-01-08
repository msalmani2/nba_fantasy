# Baseline Models Results

## Overview
This document summarizes the results from training baseline models for NBA Fantasy Score Prediction.

## Models Trained

1. **Linear Regression**: Simple linear baseline
2. **Ridge Regression**: L2 regularization (alpha=1.0)
3. **Lasso Regression**: L1 regularization (alpha=1.0)
4. **Decision Tree**: Non-linear baseline (max_depth=10)

## Evaluation Metrics

Models are evaluated on the validation set using:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

## Expected Results

Baseline models typically show:
- **Linear Regression**: Basic performance, may overfit
- **Ridge Regression**: Better generalization than Linear Regression
- **Lasso Regression**: Feature selection through L1 regularization
- **Decision Tree**: Non-linear patterns, may overfit

## Usage

To train baseline models:
```bash
python scripts/modeling/baseline_models.py
```

Results are saved to:
- Models: `models/saved/baseline_*.pkl`
- Results: `models/saved/baseline_results.pkl`

## Notes

- Baseline models establish a performance floor
- Advanced models should outperform baselines
- Results may vary based on data and hyperparameters


