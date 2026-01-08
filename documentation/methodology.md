# Methodology - NBA Fantasy Score Prediction

## Overview
This document describes the methodology used to predict NBA players' FanDuel fantasy scores using historical player statistics and ensemble machine learning models.

## Data Pipeline

### 1. Data Loading
- Source: Kaggle dataset "eoinamoore/historical-nba-data-and-player-box-scores"
- Main dataset: `PlayerStatistics.csv`
- Total records: ~1.6 million player-game observations
- Updated daily

### 2. Data Preprocessing

#### 2.1 Data Cleaning
- Removed DNP (Did Not Play) games where all statistics are missing
- Handled missing values in feature columns:
  - Rolling averages: Forward-filled within each player
  - Other features: Filled with 0
- Removed outliers in fantasy scores (1st-99th percentile kept)

#### 2.2 Feature Engineering

**Temporal Features:**
- Rolling averages (3, 5, 10 game windows) for key statistics
- Recent form indicators (last N games performance)
- Season-to-date averages
- Momentum features (trending up/down)

**Player-Specific Features:**
- Career averages
- Minutes played trends
- Usage rate approximation
- Efficiency metrics (field goal efficiency)
- Games played (experience)

**Game Context Features:**
- Home/Away indicator
- Days of rest
- Back-to-back games
- Day of week
- Month/Season indicators
- Opponent and team encoding

**Advanced Features:**
- Player vs opponent historical performance
- Team average fantasy score

#### 2.3 Data Splitting
- **Temporal Split** (not random to prevent data leakage):
  - Training: 70% (earliest data)
  - Validation: 15% (middle data)
  - Test: 15% (most recent data)

#### 2.4 Feature Selection
- Removed highly correlated features (threshold: 0.95)
- Final feature count: ~100-200 features (varies based on data)

## Model Development

### 1. Baseline Models
- **Linear Regression**: Simple baseline
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization
- **Decision Tree**: Non-linear baseline

### 2. Advanced Models

#### 2.1 Random Forest
- Hyperparameters:
  - n_estimators: 100
  - max_depth: 20
  - min_samples_split: 5
- Provides feature importance analysis

#### 2.2 XGBoost
- Hyperparameters:
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
  - subsample: 0.8
- Early stopping on validation set

#### 2.3 LightGBM
- Hyperparameters:
  - n_estimators: 100
  - num_leaves: 31
  - learning_rate: 0.1
  - feature_fraction: 0.8
- Early stopping on validation set

#### 2.4 CatBoost
- Hyperparameters:
  - iterations: 100
  - learning_rate: 0.1
  - depth: 6
- Automatic categorical feature handling
- Early stopping on validation set

### 3. Ensemble Methods

#### 3.1 Averaging Ensemble
- Simple average of all base model predictions

#### 3.2 Weighted Averaging Ensemble
- Optimized weights learned from validation set
- Weights sum to 1
- Optimized using scipy.optimize

#### 3.3 Stacking Ensemble
- Meta-learner: Ridge Regression
- Base models trained on training set
- Meta-features generated on validation set
- Meta-learner trained on meta-features

#### 3.4 Blending Ensemble
- Similar to stacking but simpler
- Linear regression to learn optimal blend weights

## Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more
- **R² (Coefficient of Determination)**: Proportion of variance explained
- **MAPE (Mean Absolute Percentage Error)**: Percentage error

## Model Selection

- Models evaluated on validation set
- Best ensemble method selected based on RMSE
- Final evaluation on test set (unseen data)

## Prediction Pipeline

1. Load new player statistics
2. Apply fantasy scoring
3. Create features using feature engineering pipeline
4. Clean and preprocess data
5. Load trained ensemble model
6. Generate predictions
7. Save predictions to file

## Model Persistence

- All models saved as pickle files in `models/saved/`
- Feature list saved for preprocessing new data
- Ensemble models and weights saved separately

## Continuous Improvement

- Models can be retrained with new data
- Hyperparameters can be tuned using validation set
- Feature engineering can be enhanced based on analysis
- New models can be added to the ensemble

## Limitations

1. **Data Quality**: Depends on accuracy of source data
2. **Temporal Changes**: NBA rules and player performance change over time
3. **Injuries**: Model doesn't account for injuries (if not in data)
4. **Matchups**: Limited opponent-specific features
5. **Context**: Game context (playoffs, back-to-back) may need more features

## Future Improvements

1. Add injury data
2. Incorporate advanced opponent statistics
3. Add player position encoding
4. Include team pace and style metrics
5. Add weather/venue factors if relevant
6. Implement online learning for model updates
7. Add uncertainty quantification


