# Team-Based Predictions Guide

## Overview
Get fantasy score predictions for players on specific teams using the team-based prediction script.

## Quick Start

### Recent Games (Recommended)

**Get predictions for recent games (default: last 1 month, e.g., December 2025):**
```bash
# Quick script for recent games only
python scripts/modeling/predict_recent_teams.py --teams Lakers Warriors --top 20

# Or with main script (filters to recent by default)
python scripts/modeling/predict_by_teams.py --teams Lakers Warriors --top 20
```

### Command Line Usage

**List all available teams:**
```bash
python scripts/modeling/predict_by_teams.py --list-teams
```

**Get predictions for specific teams:**
```bash
# Single team (recent games only, default)
python scripts/modeling/predict_by_teams.py --teams Lakers

# Multiple teams
python scripts/modeling/predict_by_teams.py --teams Lakers Warriors Celtics

# Show only top 20 players
python scripts/modeling/predict_by_teams.py --teams Lakers Warriors --top 20

# Specify months back (e.g., last 2 months)
python scripts/modeling/predict_by_teams.py --teams Lakers --months 2

# Include all historical games (not just recent)
python scripts/modeling/predict_by_teams.py --teams Lakers --all-dates

# Use existing predictions (faster, but may be outdated)
python scripts/modeling/predict_by_teams.py --teams Lakers --use-existing

# Save results to CSV
python scripts/modeling/predict_by_teams.py --teams Lakers Warriors --save lakers_warriors_predictions.csv
```

### Interactive Usage

For an interactive experience:
```bash
python scripts/modeling/predict_by_teams_simple.py
```

This will prompt you to:
1. Enter team names (comma-separated)
2. Optionally specify top N players
3. Optionally save results to CSV

## Examples

### Example 1: Get Top Players from Lakers
```bash
python scripts/modeling/predict_by_teams.py --teams Lakers --top 10
```

### Example 2: Compare Multiple Teams
```bash
python scripts/modeling/predict_by_teams.py --teams Lakers Warriors Celtics --top 15
```

### Example 3: Save Results
```bash
python scripts/modeling/predict_by_teams.py --teams Lakers Warriors --save my_predictions.csv
```

## Team Name Matching

- **Case-insensitive**: "lakers", "Lakers", "LAKERS" all work
- **Partial matching**: "Laker" will match "Lakers"
- **Full names work**: "Los Angeles Lakers" works if that's the team name in the data

## Output Format

The output includes:
- **Player**: Player's full name
- **Team**: Team name
- **Game Date**: Date of the game (if available)
- **Actual Fantasy Score**: Actual score (if available in data)
- **Predicted Fantasy Score**: Model's prediction

Results are sorted by predicted fantasy score (highest first).

## Notes

1. **Recent Games by Default**: The script automatically filters to recent games (last 3 months by default) to show current players. This ensures you see players from recent games like December 2025, not historical players from decades ago.

2. **Date Filtering**: 
   - Default: Shows games from last 3 months
   - Use `--months N` to change the time window
   - Use `--all-dates` to see all historical games

3. **Using Existing Predictions**: The `--use-existing` flag uses previously generated predictions, which is much faster. However, if the predictions file is outdated, you'll get a warning.

4. **Generating New Predictions**: If you don't use `--use-existing`, the script will:
   - Load the latest data from Kaggle
   - Filter to recent games
   - Apply feature engineering
   - Generate new predictions
   - This takes a few minutes but ensures you get the latest players

5. **Top N Players**: When using `--top N`, only the top N players by predicted score are shown.

6. **Most Recent Predictions**: The script shows the most recent prediction for each player (if multiple games exist).

## Troubleshooting

**No players found:**
- Check team name spelling
- Use `--list-teams` to see all available teams
- Try partial team names (e.g., "Laker" instead of "Los Angeles Lakers")

**Slow performance:**
- Use `--use-existing` flag to use cached predictions
- Limit results with `--top N`

**Missing predictions:**
- Run the main prediction script first: `python scripts/modeling/predict.py`
- Or remove `--use-existing` to generate new predictions

## Integration with Other Scripts

This script uses the same prediction pipeline as the main prediction script, so predictions are consistent across both methods.

