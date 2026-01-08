# Feature Engineering Documentation

## Overview
This document describes all features created during the feature engineering process for NBA Fantasy Score Prediction.

## Feature Categories

### 1. Temporal Features

#### Rolling Averages
- **Format**: `{stat}_MA{window}`
- **Windows**: 3, 5, 10 games
- **Statistics**: points, reboundsTotal, assists, steals, blocks, turnovers, fantasyScore, numMinutes
- **Example**: `points_MA5` = 5-game rolling average of points

#### Recent Form Indicators
- **Format**: `{stat}_last{n}games`
- **Windows**: 1, 3, 5 games
- **Statistics**: fantasyScore, points
- **Note**: Uses shifted values to prevent data leakage

#### Momentum Features
- **fantasyScore_momentum**: Trend in fantasy score over last 5 games
- Calculated as: (latest - earliest) / number of games

#### Season-to-Date Averages
- **Format**: `{stat}_season_avg`
- **Statistics**: All key statistics
- Calculated as expanding mean up to current game (shifted)

### 2. Player-Specific Features

#### Career Averages
- **Format**: `{stat}_career_avg`
- **Statistics**: points, reboundsTotal, assists, steals, blocks, turnovers, fantasyScore, numMinutes
- Calculated as expanding mean across all games (shifted)

#### Minutes Played Features
- **minutes_MA5**: 5-game rolling average of minutes
- **minutes_trend**: Trend in minutes played

#### Usage Rate
- **usage_rate_approx**: Field goal attempts per minute
- **usage_rate_MA5**: 5-game rolling average of usage rate

#### Efficiency Metrics
- **fg_efficiency**: Field goal percentage (made/attempted)
- **fg_efficiency_MA5**: 5-game rolling average of FG efficiency

#### Experience
- **games_played**: Cumulative number of games played by player

### 3. Game Context Features

#### Location
- **is_home**: Binary indicator (1 = home, 0 = away)

#### Rest Days
- **days_rest**: Number of days since last game (capped at 7)
- **is_back_to_back**: Binary indicator (1 = back-to-back game)

#### Time Features
- **day_of_week**: Day of week (0 = Monday, 6 = Sunday)
- **is_weekend**: Binary indicator (1 = weekend)
- **month**: Month of year (1-12)
- **year**: Year
- **is_playoff_season**: Binary indicator (1 = April-June)

#### Game Outcome
- **is_win**: Binary indicator (1 = win, 0 = loss)

#### Team Encoding
- **opponent_encoded**: Numeric encoding of opponent team
- **team_encoded**: Numeric encoding of player's team

### 4. Advanced Features

#### Player vs Opponent
- **vs_opponent_avg**: Average fantasy score against this opponent in last 5 games

#### Team Performance
- **team_avg_fantasy**: Average fantasy score of all players on team for this game
- **team_avg_fantasy_MA5**: 5-game rolling average of team fantasy score

## Feature Selection

### Highly Correlated Features Removed
- Features with correlation > 0.95 are removed
- Prevents multicollinearity issues
- Reduces model complexity

### Final Feature Count
- Typically 100-200 features after selection
- Exact count depends on data availability and correlation analysis

## Feature Importance

Feature importance is analyzed using:
- Random Forest feature importances
- XGBoost feature importances
- LightGBM feature importances
- CatBoost feature importances

Top features typically include:
- Recent fantasy score averages
- Career averages
- Minutes played
- Points and assists (key scoring components)

## Data Leakage Prevention

All features are designed to prevent data leakage:
- Rolling averages use only past data
- Career averages use expanding mean with shift
- Recent form indicators use shifted values
- No future information is included

## Missing Value Handling

- Rolling averages: Forward-filled within each player
- Career averages: Forward-filled within each player
- Other features: Filled with 0
- DNP games: Removed from dataset

## Feature Scaling

- Most tree-based models (Random Forest, XGBoost, LightGBM, CatBoost) don't require scaling
- Linear models (Ridge, Lasso) may benefit from scaling, but not implemented in baseline
- Can be added if needed for specific models

## Notes

1. Features are created in a specific order to ensure dependencies are met
2. All temporal features require data to be sorted by player and date
3. Some features may be missing for players with limited game history
4. Feature engineering is modular and can be extended easily


