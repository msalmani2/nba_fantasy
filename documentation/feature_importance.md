# Feature Importance Analysis

## Overview
This document describes feature importance analysis for NBA Fantasy Score Prediction models.

## Methods

### 1. Tree-Based Model Importance
- **Random Forest**: Gini importance
- **XGBoost**: Gain importance
- **LightGBM**: Split importance
- **CatBoost**: Prediction value changez

### 2. Permutation Importance
- Shuffle feature values
- Measure impact on model performance
- Model-agnostic method

### 3. Correlation Analysis
- Correlation with target variable
- Simple but informative
- Linear relationships only

## Top Features (Expected)

Based on domain knowledge and typical model behavior:

### Temporal Features
- Recent fantasy score averages (MA5, MA10)
- Career averages
- Recent form indicators

### Player Statistics
- Points (direct component of fantasy score)
- Assists (high weight in scoring)
- Rebounds (moderate weight)
- Minutes played

### Game Context
- Home/Away
- Days of rest
- Opponent strength

## Analysis Process

1. **Extract Importances**: From trained models
2. **Normalize**: Scale to 0-1 or percentage
3. **Rank**: Sort by importance
4. **Visualize**: Bar charts, heatmaps
5. **Interpret**: Understand why features matter

## Visualization

- **Bar Charts**: Top N features
- **Heatmaps**: Feature importance across models
- **Comparison**: Model-specific vs consensus

## Feature Selection

Feature importance can guide:
- **Feature Selection**: Remove low-importance features
- **Feature Engineering**: Focus on important feature types
- **Model Interpretation**: Understand model behavior

## Implementation

Feature importance extracted from:
- `scripts/modeling/random_forest.py`
- `scripts/modeling/xgboost_model.py`
- `scripts/modeling/lightgbm_model.py`
- `scripts/modeling/catboost_model.py`

## Notes

- Importance varies by model
- Consensus across models indicates robust features
- High importance doesn't guarantee causality
- Low importance doesn't mean feature is useless
- Context matters for interpretation


