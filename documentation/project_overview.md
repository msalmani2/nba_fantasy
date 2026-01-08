# NBA Fantasy Score Prediction - Project Overview

## Project Description

This project aims to predict NBA players' FanDuel daily fantasy scores using historical player statistics and ensemble machine learning techniques. The system uses a comprehensive dataset of NBA player statistics and implements multiple advanced models combined through ensemble methods to achieve accurate predictions.

## Objectives

1. **Data Exploration**: Understand the structure and quality of NBA player statistics data
2. **Fantasy Scoring**: Research and implement FanDuel's NBA scoring system
3. **Feature Engineering**: Create meaningful features from raw statistics
4. **Model Development**: Build and tune multiple machine learning models
5. **Ensemble Methods**: Combine models to improve prediction accuracy
6. **Evaluation**: Comprehensive model evaluation and error analysis
7. **Deployment**: Create a prediction pipeline for new data

## Dataset

- **Source**: Kaggle dataset "eoinamoore/historical-nba-data-and-player-box-scores"
- **Main File**: PlayerStatistics.csv
- **Size**: ~1.6 million player-game observations
- **Update Frequency**: Daily
- **Columns**: 35 columns including player stats, game info, team info

## Methodology Summary

1. **Data Loading**: Load dataset using kagglehub
2. **Data Cleaning**: Handle missing values, outliers, DNP games
3. **Fantasy Scoring**: Calculate FanDuel fantasy scores from statistics
4. **Feature Engineering**: Create temporal, player-specific, and game context features
5. **Preprocessing**: Temporal train/validation/test split, feature selection
6. **Baseline Models**: Linear Regression, Ridge, Lasso, Decision Tree
7. **Advanced Models**: Random Forest, XGBoost, LightGBM, CatBoost
8. **Ensemble Methods**: Averaging, Weighted Averaging, Stacking, Blending
9. **Evaluation**: Comprehensive metrics and error analysis
10. **Prediction Pipeline**: Deployable system for new predictions

## Success Metrics

- **Target MAE**: < 5 fantasy points
- **Target RMSE**: < 7 fantasy points
- **Ensemble Performance**: Outperform individual models
- **Reproducibility**: Fully documented and reproducible code

## Project Structure

```
nba_fantasy/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned and feature-engineered data
│   └── external/         # Additional datasets
├── scripts/
│   ├── data_processing/  # Data loading, cleaning, feature engineering
│   ├── modeling/         # Model training and evaluation
│   ├── utils/            # Utility functions
│   └── temp/             # Temporary scripts
├── notebooks/
│   ├── exploration/      # EDA notebooks
│   └── modeling/         # Model development notebooks
├── models/
│   ├── saved/            # Trained model files
│   └── predictions/      # Model predictions
├── documentation/        # Project documentation
├── readmes/             # Step-by-step readmes
├── config/              # Configuration files
└── tests/               # Unit tests
```

## Key Features

### FanDuel Scoring System
- Points: +1.0 per point
- Rebounds: +1.2 per rebound
- Assists: +1.5 per assist
- Steals: +2.0 per steal
- Blocks: +2.0 per block
- Turnovers: -1.0 per turnover

### Feature Engineering
- Temporal features (rolling averages, recent form)
- Player-specific features (career averages, efficiency)
- Game context features (home/away, rest days, opponent)
- Advanced features (player vs opponent, team performance)

### Models
- Baseline: Linear, Ridge, Lasso, Decision Tree
- Advanced: Random Forest, XGBoost, LightGBM, CatBoost
- Ensemble: Averaging, Weighted Averaging, Stacking, Blending

## Usage

### Training Models
```bash
# Train baseline models
python scripts/modeling/baseline_models.py

# Train advanced models
python scripts/modeling/train_models.py

# Train ensemble
python scripts/modeling/ensemble.py
```

### Making Predictions
```bash
python scripts/modeling/predict.py
```

### Data Processing
```bash
# Load and explore data
python scripts/data_processing/load_data.py

# Create features
python scripts/data_processing/feature_engineering.py

# Preprocess data
python scripts/data_processing/train_test_split.py
```

## Dependencies

See `requirements.txt` for full list. Key packages:
- pandas, numpy
- scikit-learn
- xgboost, lightgbm, catboost
- matplotlib, seaborn
- kagglehub

## Documentation

- `data_dictionary.md`: Dataset schema and column descriptions
- `fanduel_scoring.md`: FanDuel scoring system documentation
- `methodology.md`: Detailed methodology and approach
- `features.md`: Feature engineering documentation
- `project_overview.md`: This file

## Results

Model performance is evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

Results are documented in:
- `readmes/step_by_step/`: Step-by-step results
- `documentation/results/`: Final results reports

## Future Improvements

1. Add injury data integration
2. Incorporate advanced opponent statistics
3. Add player position encoding
4. Include team pace and style metrics
5. Implement online learning for continuous updates
6. Add uncertainty quantification
7. Create web interface for predictions

## License

[Add your license here]

## Author

[Add your name here]

## Acknowledgments

- Kaggle dataset: "eoinamoore/historical-nba-data-and-player-box-scores"
- FanDuel for scoring system reference


