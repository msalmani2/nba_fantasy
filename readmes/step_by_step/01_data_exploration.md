# Step 1: Data Exploration

## Overview
This document summarizes the initial exploration of the PlayerStatistics.csv dataset from the Kaggle dataset "eoinamoore/historical-nba-data-and-player-box-scores".

## Dataset Summary

### Basic Information
- **Total Records**: 1,641,703 rows
- **Total Columns**: 35 columns
- **File Size**: ~304 MB

### Key Findings

#### 1. Dataset Structure
The dataset contains player-level game statistics with the following main categories:
- Player identification (name, ID)
- Game identification (game ID, date, type)
- Team information (player team, opponent team, home/away)
- Statistical measures (points, rebounds, assists, steals, blocks, turnovers, etc.)

#### 2. Missing Values Analysis

**High Missing Rate (>90%)**:
- `gameSubLabel`: 99.61% missing
- `gameLabel`: 94.22% missing  
- `seriesGameNumber`: 91.77% missing

These columns are likely only relevant for specific game types (e.g., playoffs) and can be handled accordingly.

**Moderate Missing Rate**:
- `numMinutes`: 9.99% missing (likely DNP games)

**Low Missing Rate (<1%)**:
- All statistical columns: 0.07% missing (1,219 rows)
- `gameType`: 0.75% missing

**Note**: The same 1,219 rows appear to have missing values for all statistical columns, suggesting these are games where the player did not play (DNP).

#### 3. Data Types
- **Float64**: 21 columns (mostly statistical measures)
- **Object**: 10 columns (names, teams, dates, labels)
- **Integer**: 4 columns (IDs, flags)

#### 4. Key Columns for Fantasy Scoring
The following columns are directly relevant for calculating FanDuel fantasy scores:
- `points`: Points scored
- `reboundsTotal`: Total rebounds
- `assists`: Assists
- `steals`: Steals
- `blocks`: Blocks
- `turnovers`: Turnovers

## Data Quality Issues Identified

1. **Mixed Types Warning**: Columns 10 and 11 (likely `gameType` and `gameLabel`) have mixed types and may need type conversion
2. **Date Format**: `gameDateTimeEst` is stored as object and needs to be converted to datetime
3. **DNP Games**: 1,219 rows with missing statistical data need to be handled (either removed or marked)

## Next Steps

1. **Data Cleaning**:
   - Convert `gameDateTimeEst` to datetime
   - Handle missing values (especially DNP games)
   - Fix mixed type columns

2. **FanDuel Scoring Research**:
   - Research official FanDuel NBA scoring system
   - Implement fantasy score calculator
   - Validate scoring calculations

3. **Feature Engineering**:
   - Create temporal features (rolling averages, recent form)
   - Add player-specific features
   - Include game context features

## Files Created

- `scripts/data_processing/load_data.py`: Data loading script
- `documentation/data_dictionary.md`: Complete data dictionary
- `data/raw/PlayerStatistics.csv`: Raw dataset (saved locally)

## Usage

To load and explore the data:

```python
from scripts.data_processing.load_data import load_player_statistics, explore_dataset

# Load the dataset
df = load_player_statistics()

# Explore the dataset
exploration_results = explore_dataset(df)
```

## Notes

- The dataset is updated daily, ensuring access to the most recent games
- Each row represents one player's performance in one game
- The dataset spans multiple seasons (exact date range to be determined in further analysis)


