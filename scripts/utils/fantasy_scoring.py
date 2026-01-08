"""
Fantasy scoring calculator for FanDuel NBA daily fantasy.

This module implements the FanDuel NBA scoring system to calculate
fantasy points from player statistics.
"""

import pandas as pd
import numpy as np


def calculate_fanduel_score(row):
    """
    Calculate FanDuel fantasy score for a single game.
    
    FanDuel NBA Scoring System:
    - 3-point Field Goals Made: +3 points each
    - 2-point Field Goals Made: +2 points each
    - Free Throws Made: +1 point each
    - Rebounds: +1.2 per rebound
    - Assists: +1.5 per assist
    - Steals: +3 per steal
    - Blocks: +3 per block
    - Turnovers: -1 per turnover
    
    Parameters:
    -----------
    row : pd.Series or dict
        A row containing player statistics with the following columns:
        - threePointersMade (or 'threePointersMade')
        - fieldGoalsMade (or 'fieldGoalsMade')
        - freeThrowsMade (or 'freeThrowsMade')
        - reboundsTotal (or 'reboundsTotal')
        - assists (or 'assists')
        - steals (or 'steals')
        - blocks (or 'blocks')
        - turnovers (or 'turnovers')
    
    Returns:
    --------
    float
        The calculated FanDuel fantasy score
    """
    # Handle both Series and dict-like objects
    if isinstance(row, pd.Series):
        three_pointers_made = row.get('threePointersMade', 0) if pd.notna(row.get('threePointersMade')) else 0
        field_goals_made = row.get('fieldGoalsMade', 0) if pd.notna(row.get('fieldGoalsMade')) else 0
        free_throws_made = row.get('freeThrowsMade', 0) if pd.notna(row.get('freeThrowsMade')) else 0
        rebounds = row.get('reboundsTotal', 0) if pd.notna(row.get('reboundsTotal')) else 0
        assists = row.get('assists', 0) if pd.notna(row.get('assists')) else 0
        steals = row.get('steals', 0) if pd.notna(row.get('steals')) else 0
        blocks = row.get('blocks', 0) if pd.notna(row.get('blocks')) else 0
        turnovers = row.get('turnovers', 0) if pd.notna(row.get('turnovers')) else 0
    else:
        three_pointers_made = row.get('threePointersMade', 0) or 0
        field_goals_made = row.get('fieldGoalsMade', 0) or 0
        free_throws_made = row.get('freeThrowsMade', 0) or 0
        rebounds = row.get('reboundsTotal', 0) or 0
        assists = row.get('assists', 0) or 0
        steals = row.get('steals', 0) or 0
        blocks = row.get('blocks', 0) or 0
        turnovers = row.get('turnovers', 0) or 0
    
    # Convert to float and handle NaN
    three_pointers_made = float(three_pointers_made) if pd.notna(three_pointers_made) else 0.0
    field_goals_made = float(field_goals_made) if pd.notna(field_goals_made) else 0.0
    free_throws_made = float(free_throws_made) if pd.notna(free_throws_made) else 0.0
    rebounds = float(rebounds) if pd.notna(rebounds) else 0.0
    assists = float(assists) if pd.notna(assists) else 0.0
    steals = float(steals) if pd.notna(steals) else 0.0
    blocks = float(blocks) if pd.notna(blocks) else 0.0
    turnovers = float(turnovers) if pd.notna(turnovers) else 0.0
    
    # Calculate 2-pointers made (total FGs minus 3-pointers)
    two_pointers_made = field_goals_made - three_pointers_made
    
    # Calculate FanDuel score
    fantasy_score = (
        three_pointers_made * 3.0 +    # 3-point FGs: +3 each
        two_pointers_made * 2.0 +      # 2-point FGs: +2 each
        free_throws_made * 1.0 +       # Free throws: +1 each
        rebounds * 1.2 +               # Rebounds: +1.2 per rebound
        assists * 1.5 +                 # Assists: +1.5 per assist
        steals * 3.0 +                  # Steals: +3 per steal
        blocks * 3.0 +                  # Blocks: +3 per block
        turnovers * -1.0                # Turnovers: -1 per turnover
    )
    
    return fantasy_score


def calculate_fanduel_scores(df):
    """
    Calculate FanDuel fantasy scores for an entire DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing player statistics with columns:
        - points
        - reboundsTotal
        - assists
        - steals
        - blocks
        - turnovers
    
    Returns:
    --------
    pd.Series
        Series containing fantasy scores for each row
    """
    return df.apply(calculate_fanduel_score, axis=1)


def add_fantasy_score_column(df):
    """
    Add a 'fantasyScore' column to the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing player statistics
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'fantasyScore' column
    """
    df = df.copy()
    df['fantasyScore'] = calculate_fanduel_scores(df)
    return df


def validate_scoring_example():
    """
    Validate the scoring calculation with a known example.
    
    Example: A player with:
    - 4 three-pointers made (12 points)
    - 8 two-pointers made (16 points)
    - 5 free throws made (5 points)
    - Total: 25 points
    - 10 rebounds
    - 8 assists
    - 2 steals
    - 1 block
    - 3 turnovers
    
    Expected score: 4*3 + 8*2 + 5*1 + 10*1.2 + 8*1.5 + 2*3 + 1*3 - 3*1
                  = 12 + 16 + 5 + 12 + 12 + 6 + 3 - 3
                  = 63
    """
    example = pd.Series({
        'threePointersMade': 4,
        'fieldGoalsMade': 12,  # 4 three-pointers + 8 two-pointers
        'freeThrowsMade': 5,
        'reboundsTotal': 10,
        'assists': 8,
        'steals': 2,
        'blocks': 1,
        'turnovers': 3
    })
    
    score = calculate_fanduel_score(example)
    expected = 4*3 + 8*2 + 5*1 + 10*1.2 + 8*1.5 + 2*3 + 1*3 - 3*1
    
    print(f"Example player stats:")
    print(f"  3-pointers made: {example['threePointersMade']}")
    print(f"  2-pointers made: {example['fieldGoalsMade'] - example['threePointersMade']}")
    print(f"  Free throws made: {example['freeThrowsMade']}")
    print(f"  Rebounds: {example['reboundsTotal']}")
    print(f"  Assists: {example['assists']}")
    print(f"  Steals: {example['steals']}")
    print(f"  Blocks: {example['blocks']}")
    print(f"  Turnovers: {example['turnovers']}")
    print(f"\nCalculated Fantasy Score: {score}")
    print(f"Expected Fantasy Score: {expected}")
    print(f"Match: {abs(score - expected) < 0.01}")
    
    return abs(score - expected) < 0.01


if __name__ == "__main__":
    # Validate the scoring function
    print("Validating FanDuel scoring calculation...")
    print("=" * 60)
    is_valid = validate_scoring_example()
    print("=" * 60)
    if is_valid:
        print("✓ Scoring calculation is correct!")
    else:
        print("✗ Scoring calculation has errors!")

