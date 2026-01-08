# FanDuel NBA Daily Fantasy Scoring System

## Overview
This document describes the FanDuel NBA daily fantasy scoring system used in this project.

## Scoring Formula

FanDuel uses the following scoring system for NBA daily fantasy:

| Statistic | Points |
|-----------|--------|
| 3-Point Field Goal Made | +3.0 per made |
| 2-Point Field Goal Made | +2.0 per made |
| Free Throw Made | +1.0 per made |
| Rebounds | +1.2 per rebound |
| Assists | +1.5 per assist |
| Steals | +3.0 per steal |
| Blocks | +3.0 per block |
| Turnovers | -1.0 per turnover |

## Formula

```
Fantasy Score = (3-Pointers Made × 3.0) + 
                (2-Pointers Made × 2.0) + 
                (Free Throws Made × 1.0) + 
                (Rebounds × 1.2) + 
                (Assists × 1.5) + 
                (Steals × 3.0) + 
                (Blocks × 3.0) - 
                (Turnovers × 1.0)
```

Where:
- 2-Pointers Made = Total Field Goals Made - 3-Pointers Made

## Example Calculation

**Player Statistics:**
- 3-Pointers Made: 4
- Total Field Goals Made: 12 (so 8 two-pointers)
- Free Throws Made: 5
- Rebounds: 10
- Assists: 8
- Steals: 2
- Blocks: 1
- Turnovers: 3

**Fantasy Score Calculation:**
```
= (4 × 3.0) + (8 × 2.0) + (5 × 1.0) + (10 × 1.2) + (8 × 1.5) + (2 × 3.0) + (1 × 3.0) - (3 × 1.0)
= 12 + 16 + 5 + 12 + 12 + 6 + 3 - 3
= 63.0 fantasy points
```

## Implementation

The scoring system is implemented in `scripts/utils/fantasy_scoring.py` with the following functions:

- `calculate_fanduel_score(row)`: Calculate score for a single game/row
- `calculate_fanduel_scores(df)`: Calculate scores for an entire DataFrame
- `add_fantasy_score_column(df)`: Add fantasy score column to DataFrame

## Data Mapping

The following columns from PlayerStatistics.csv are used:
- `threePointersMade` → 3-point field goals made
- `fieldGoalsMade` → Total field goals made (used to calculate 2-pointers)
- `freeThrowsMade` → Free throws made
- `reboundsTotal` → Total rebounds
- `assists` → Assists
- `steals` → Steals
- `blocks` → Blocks
- `turnovers` → Turnovers

**Note**: 2-pointers made = `fieldGoalsMade` - `threePointersMade`

## Notes

1. **Missing Values**: If any statistic is missing (NaN), it is treated as 0 in the calculation.

2. **Verification**: The scoring formula has been verified against known examples and should match FanDuel's official scoring system. However, it's recommended to verify with current FanDuel rules as scoring systems may change.

3. **Historical Accuracy**: This scoring system is based on FanDuel's standard NBA daily fantasy scoring. Different contest types may have different scoring rules.

## References

- FanDuel NBA Daily Fantasy Rules (verify current rules on fanduel.com)
- The scoring system emphasizes assists and defensive stats (steals, blocks) more than basic points

