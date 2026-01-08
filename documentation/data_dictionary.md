# Data Dictionary - PlayerStatistics.csv

## Dataset Overview

- **Source**: Kaggle dataset "eoinamoore/historical-nba-data-and-player-box-scores"
- **File**: PlayerStatistics.csv
- **Total Rows**: 1,641,703
- **Total Columns**: 35
- **Last Updated**: Daily (as per dataset description)

## Column Descriptions

### Player Identification
- **firstName** (object): Player's first name
- **lastName** (object): Player's last name
- **personId** (int64): Unique identifier for the player

### Game Identification
- **gameId** (int64): Unique identifier for the game
- **gameDateTimeEst** (object): Date and time of the game in Eastern Time
- **gameType** (object): Type of game (e.g., Regular Season, Playoffs) - 0.75% missing
- **gameLabel** (object): Game label/description - 94.22% missing
- **gameSubLabel** (object): Game sub-label - 99.61% missing
- **seriesGameNumber** (float64): Game number in a series (for playoffs) - 91.77% missing

### Team Information
- **playerteamCity** (object): City of the player's team
- **playerteamName** (object): Name of the player's team
- **opponentteamCity** (object): City of the opponent team
- **opponentteamName** (object): Name of the opponent team
- **win** (int64): Whether the player's team won (1) or lost (0)
- **home** (int64): Whether the player's team was home (1) or away (0)

### Playing Time
- **numMinutes** (float64): Minutes played in the game - 9.99% missing

### Scoring Statistics
- **points** (float64): Total points scored - 0.07% missing
- **fieldGoalsMade** (float64): Field goals made
- **fieldGoalsAttempted** (float64): Field goals attempted
- **fieldGoalsPercentage** (float64): Field goal percentage
- **threePointersMade** (float64): Three-pointers made
- **threePointersAttempted** (float64): Three-pointers attempted
- **threePointersPercentage** (float64): Three-point percentage
- **freeThrowsMade** (float64): Free throws made
- **freeThrowsAttempted** (float64): Free throws attempted
- **freeThrowsPercentage** (float64): Free throw percentage

### Other Statistics
- **assists** (float64): Assists - 0.07% missing
- **reboundsTotal** (float64): Total rebounds
- **reboundsDefensive** (float64): Defensive rebounds
- **reboundsOffensive** (float64): Offensive rebounds
- **steals** (float64): Steals - 0.07% missing
- **blocks** (float64): Blocks - 0.07% missing
- **turnovers** (float64): Turnovers - 0.07% missing
- **foulsPersonal** (float64): Personal fouls
- **plusMinusPoints** (float64): Plus/minus rating

## Data Quality Notes

### Missing Values
1. **High Missing Rate (>90%)**:
   - `gameSubLabel`: 99.61% missing - likely only relevant for specific game types
   - `gameLabel`: 94.22% missing - likely only relevant for specific game types
   - `seriesGameNumber`: 91.77% missing - only relevant for playoff series

2. **Moderate Missing Rate (5-10%)**:
   - `numMinutes`: 9.99% missing - may indicate DNP (Did Not Play) games

3. **Low Missing Rate (<1%)**:
   - All statistical columns (points, assists, blocks, steals, etc.): 0.07% missing
   - `gameType`: 0.75% missing

### Data Types
- **Numeric (float64)**: 21 columns - mostly statistical measures
- **Object (string)**: 10 columns - names, teams, dates, labels
- **Integer (int64)**: 4 columns - IDs, win/loss, home/away flags

### Data Quality Considerations
1. The same 1,219 rows appear to have missing values for all statistical columns - these may be DNP games
2. Mixed types warning for columns 10 and 11 (likely `gameType` and `gameLabel`) - may need type conversion
3. Date column (`gameDateTimeEst`) is stored as object - will need to convert to datetime

## Key Statistics for Fantasy Scoring

The following columns are directly relevant for FanDuel fantasy scoring:
- `points`: Points scored
- `reboundsTotal`: Total rebounds (or `reboundsDefensive` + `reboundsOffensive`)
- `assists`: Assists
- `steals`: Steals
- `blocks`: Blocks
- `turnovers`: Turnovers

## Usage Notes

- The dataset is updated daily, so it includes the most recent games
- Each row represents one player's performance in one game
- Multiple rows can have the same `gameId` (one for each player)
- Multiple rows can have the same `personId` (one for each game the player played)
- The dataset spans multiple seasons (exact range to be determined during exploration)


