# Recent Games Evaluation Report

## Overview
This report evaluates model performance on the most recent games (last 7 days) to assess real-world accuracy.

## Dataset Update
- **Previous Version**: 315
- **Current Version**: 317
- **New Records**: 472 additional player-game records
- **Date Range Analyzed**: December 20-27, 2025

## Overall Performance

### Aggregate Metrics (All Recent Games)
- **MAE**: 0.27 fantasy points
- **RMSE**: 0.40 fantasy points
- **R²**: 0.9992 (99.92% variance explained)
- **Total Games**: 55 unique games
- **Total Player-Game Records**: 1,414

### Game-Level Performance (Average across 20 games)
- **Average MAE**: 0.26 ± 0.06 fantasy points
- **Average RMSE**: 0.37 ± 0.12 fantasy points
- **Average R²**: 0.9992 ± 0.0005

## Key Findings

### 1. Excellent Overall Accuracy
The model achieves very high accuracy on recent games:
- Average error of only **0.26 fantasy points** per game
- **99.92%** of variance explained
- Consistent performance across different games

### 2. Game-by-Game Consistency
- All 20 analyzed games show R² > 0.998
- MAE ranges from 0.16 to 0.33 across games
- Model performs consistently regardless of game context

### 3. Player-Level Accuracy
Most predictions are within 0.5 fantasy points of actual scores:
- **High-scoring players** (40+ points): Typically within 1-2 points
- **Mid-range players** (20-40 points): Typically within 0.5-1 point
- **Low-scoring players** (<20 points): Typically within 0.2-0.5 points

## Sample Game Analysis

### Game 1: Bucks vs Bulls (Dec 27, 2025)
- **MAE**: 0.22
- **RMSE**: 0.35
- **R²**: 0.9993
- **Top Prediction**: Giannis Antetokounmpo - Predicted 41.71, Actual 42.1 (Error: 0.39)

### Game 2: Cavaliers vs Rockets (Dec 27, 2025)
- **MAE**: 0.32
- **RMSE**: 0.45
- **R²**: 0.9986
- **Top Prediction**: Reed Sheppard - Predicted 43.02, Actual 44.6 (Error: 1.58)

### Game 3: Clippers vs Trail Blazers (Dec 26, 2025)
- **MAE**: 0.24
- **RMSE**: 0.35
- **R²**: 0.9995
- **Top Prediction**: Kawhi Leonard - Predicted 49.36, Actual 50.6 (Error: 1.24)

### Game 19: Clippers vs Lakers (Dec 20, 2025) - Best Performance
- **MAE**: 0.16
- **RMSE**: 0.19
- **R²**: 0.9998
- **Top Prediction**: LeBron James - Predicted 48.00, Actual 48.3 (Error: 0.30)

## Error Analysis

### Largest Errors Observed
1. **Reed Sheppard** (Rockets): Predicted 43.02, Actual 44.6 (Error: 1.58)
2. **Santi Aldama** (Grizzlies): Predicted 52.78, Actual 54.0 (Error: 1.22)
3. **Kawhi Leonard** (Clippers): Predicted 49.36, Actual 50.6 (Error: 1.24)

### Error Patterns
- Most errors are **under-predictions** for high-scoring games
- Errors are typically **< 1.5 fantasy points** even for outliers
- **Zero-scoring players** (DNP) are consistently predicted at ~0.12 (minimum threshold)

## Model Strengths

1. **Consistent Performance**: Model performs well across different teams and game contexts
2. **High Accuracy**: Average error of 0.26 points is excellent for fantasy predictions
3. **Reliable for High Scorers**: Top players' predictions are very accurate
4. **Handles Low Scorers**: Correctly identifies players with minimal playing time

## Areas for Improvement

1. **Outlier Handling**: Some high-scoring games show slightly larger errors (1-2 points)
2. **DNP Detection**: Players with 0 points are predicted at 0.12 (could be improved to exactly 0)
3. **Context Factors**: Could potentially incorporate more game-specific context (playoff implications, etc.)

## Conclusion

The model demonstrates **excellent performance** on recent games:
- ✅ **MAE of 0.27** is well below the target of 5 points
- ✅ **RMSE of 0.40** is well below the target of 7 points
- ✅ **R² of 0.9992** shows near-perfect fit
- ✅ **Consistent across games** with low variance

The model is **production-ready** and can reliably predict fantasy scores for upcoming games.

## Files Generated

1. **Summary**: `documentation/results/recent_games_evaluation.csv`
2. **Detailed**: `documentation/results/recent_games_detailed.csv` (player-level data for all games)

## Next Steps

1. Continue monitoring performance on new games
2. Retrain model periodically as more data becomes available
3. Consider fine-tuning for edge cases (very high/low scorers)
4. Implement automated evaluation pipeline for daily monitoring


