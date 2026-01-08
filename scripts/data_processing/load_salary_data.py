"""
Load and integrate fantasy salary data.

This script can load salary data from various sources:
1. CSV files (e.g., from RickRunGood)
2. API endpoints (if available)
3. Manual data entry

Currently supports loading from CSV files.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_salary_csv(file_path, platform='fanduel'):
    """
    Load salary data from CSV file.
    
    Expected CSV format (example from RickRunGood):
    - Player Name
    - Position
    - Salary
    - Team
    - Opponent
    - Date
    
    Parameters:
    -----------
    file_path : str or Path
        Path to CSV file
    platform : str
        Platform name (fanduel, draftkings, yahoo)
    
    Returns:
    --------
    pd.DataFrame
        Salary data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names
        column_mapping = {
            'Player': 'player_name',
            'Player Name': 'player_name',
            'Name': 'player_name',
            'Salary': 'salary',
            'FD Salary': 'salary',
            'DK Salary': 'salary',
            'FanDuel Salary': 'salary',
            'DraftKings Salary': 'salary',
            'Team': 'team',
            'Opponent': 'opponent',
            'Opp': 'opponent',
            'Date': 'date',
            'Game Date': 'date',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add platform
        df['platform'] = platform
        
        return df
    
    except Exception as e:
        print(f"Error loading salary CSV: {e}")
        return None


def merge_salary_with_predictions(predictions_df, salary_df, date_col='gameDate'):
    """
    Merge salary data with predictions.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with predictions
    salary_df : pd.DataFrame
        DataFrame with salary data
    date_col : str
        Date column name
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with predictions and salaries
    """
    # Ensure date columns are datetime
    if date_col in predictions_df.columns:
        predictions_df[date_col] = pd.to_datetime(predictions_df[date_col], errors='coerce')
    
    if 'date' in salary_df.columns:
        salary_df['date'] = pd.to_datetime(salary_df['date'], errors='coerce')
    
    # Create player name in predictions if needed
    if 'Player' not in predictions_df.columns:
        if 'firstName' in predictions_df.columns and 'lastName' in predictions_df.columns:
            predictions_df['Player'] = predictions_df['firstName'] + ' ' + predictions_df['lastName']
        else:
            print("Warning: Cannot create player name for merging")
            return predictions_df
    
    # Merge on player name and date
    merged = predictions_df.merge(
        salary_df,
        left_on=['Player', date_col],
        right_on=['player_name', 'date'],
        how='left'
    )
    
    # Calculate value metrics if salary available
    if 'salary' in merged.columns and 'predicted_fantasy_score' in merged.columns:
        merged['points_per_dollar'] = merged['predicted_fantasy_score'] / (merged['salary'] / 1000)
        merged['value_rating'] = merged['points_per_dollar'].rank(ascending=False)
    
    return merged


def get_value_picks(predictions_df, salary_df, min_salary=3000, max_salary=12000, top_n=20):
    """
    Identify value picks (high points per dollar).
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        Predictions DataFrame
    salary_df : pd.DataFrame
        Salary DataFrame
    min_salary : int
        Minimum salary to consider
    max_salary : int
        Maximum salary to consider
    top_n : int
        Number of top value picks to return
    
    Returns:
    --------
    pd.DataFrame
        Top value picks
    """
    merged = merge_salary_with_predictions(predictions_df, salary_df)
    
    # Filter by salary range
    if 'salary' in merged.columns:
        filtered = merged[
            (merged['salary'] >= min_salary) & 
            (merged['salary'] <= max_salary)
        ].copy()
        
        # Sort by points per dollar
        if 'points_per_dollar' in filtered.columns:
            filtered = filtered.sort_values('points_per_dollar', ascending=False)
            return filtered.head(top_n)
    
    return merged


if __name__ == "__main__":
    print("="*80)
    print("FANTASY SALARY DATA LOADER")
    print("="*80)
    print("\nThis script helps load and integrate fantasy salary data.")
    print("\nTo use:")
    print("1. Download salary CSV from RickRunGood or similar source")
    print("2. Place it in data/external/ directory")
    print("3. Load using: load_salary_csv('data/external/salaries.csv')")
    print("\nFor API integration, you'll need API keys from:")
    print("- SportsDataIO: https://discoverylab.sportsdata.io")
    print("- Sportradar: https://developer.sportradar.com")
    print("\nSee documentation/fantasy_salary_sources.md for more info.")


