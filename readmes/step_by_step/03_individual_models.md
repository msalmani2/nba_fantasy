# Individual Advanced Models Results

## Overview
This document summarizes the results from training individual advanced models (Random Forest, XGBoost, LightGBM, CatBoost).

## Models Trained

1. **Random Forest**: Ensemble of decision trees
2. **XGBoost**: Gradient boosting with early stopping
3. **LightGBM**: Fast gradient boosting
4. **CatBoost**: Gradient boosting with categorical handling

## Model Configurations

### Random Forest
- n_estimators: 100
- max_depth: 20
- min_samples_split: 5

### XGBoost
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- Early stopping: 10 rounds

### LightGBM
- n_estimators: 100
- num_leaves: 31
- learning_rate: 0.1
- Early stopping: 10 rounds

### CatBoost
- iterations: 100
- learning_rate: 0.1
- depth: 6
- Early stopping: 10 rounds

## Evaluation

Models are evaluated on validation set using standard regression metrics.

## Feature Importance

Each model provides feature importance:
- Random Forest: Gini importance
- XGBoost: Gain importance
- LightGBM: Split importance
- CatBoost: Prediction value change

## Usage

To train all advanced models:
```bash
python scripts/modeling/train_models.py
```

To train individual models:
```bash
python scripts/modeling/random_forest.py
python scripts/modeling/xgboost_model.py
python scripts/modeling/lightgbm_model.py
python scripts/modeling/catboost_model.py
```

## Model Files

Models are saved to `models/saved/`:
- `random_forest.pkl`
- `xgboost.pkl`
- `lightgbm.pkl`
- `catboost.pkl`
- `advanced_models_results.pkl`

## Notes

- XGBoost and LightGBM typically perform best
- Feature importance can guide feature engineering
- Early stopping prevents overfitting
- Models can be tuned further with hyperparameter optimization


