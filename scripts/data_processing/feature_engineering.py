"""
Feature engineering pipeline for NBA Fantasy Score Prediction.

This module creates various features including:
- Temporal features (rolling averages, recent form)
- Player-specific features (career averages, position)
- Game context features (home/away, days of rest, opponent stats)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_config():
    """Load configuration from config.yaml"""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_temporal_features(df, windows=[3, 5, 10]):
    """
    Create temporal features like rolling averages and recent form.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with player statistics, must be sorted by player and date
    windows : list
        List of window sizes for rolling averages
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Ensure we have a date column
    if 'gameDate' not in df.columns and 'gameDateTimeEst' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDateTimeEst'], format='mixed', errors='coerce', utc=True)
        # Remove timezone info to avoid mixing timezone-aware and naive datetimes
        if pd.api.types.is_datetime64_any_dtype(df['gameDate']):
            if df['gameDate'].dt.tz is not None:
                df['gameDate'] = df['gameDate'].dt.tz_localize(None)
    
    # Sort by player and date
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    # Key statistics for rolling averages
    stat_cols = ['points', 'reboundsTotal', 'assists', 'steals', 'blocks', 
                 'turnovers', 'fantasyScore', 'numMinutes']
    
    # Create rolling averages for each window
    for window in windows:
        for stat in stat_cols:
            if stat in df.columns:
                col_name = f'{stat}_MA{window}'
                df[col_name] = df.groupby('personId')[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
    
    # Create recent form indicators (last N games)
    for n in [1, 3, 5]:
        for stat in ['fantasyScore', 'points']:
            if stat in df.columns:
                col_name = f'{stat}_last{n}games'
                df[col_name] = df.groupby('personId')[stat].transform(
                    lambda x: x.shift(1).rolling(window=n, min_periods=1).mean()
                )
    
    # Create momentum features (trending up/down)
    if 'fantasyScore' in df.columns:
        df['fantasyScore_momentum'] = df.groupby('personId')['fantasyScore'].transform(
            lambda x: x.rolling(window=5, min_periods=2).apply(
                lambda y: (y.iloc[-1] - y.iloc[0]) / len(y) if len(y) > 1 else 0
            )
        )
    
    # Season-to-date averages
    if 'gameDate' in df.columns:
        df['year'] = df['gameDate'].dt.year
        for stat in stat_cols:
            if stat in df.columns:
                col_name = f'{stat}_season_avg'
                df[col_name] = df.groupby(['personId', 'year'])[stat].transform(
                    lambda x: x.expanding().mean().shift(1)
                )
    
    return df


def create_player_specific_features(df):
    """
    Create player-specific features like career averages and position encoding.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with player statistics
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added player-specific features
    """
    df = df.copy()
    
    # Career averages (up to current game)
    stat_cols = ['points', 'reboundsTotal', 'assists', 'steals', 'blocks', 
                 'turnovers', 'fantasyScore', 'numMinutes']
    
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    for stat in stat_cols:
        if stat in df.columns:
            col_name = f'{stat}_career_avg'
            df[col_name] = df.groupby('personId')[stat].transform(
                lambda x: x.expanding().mean().shift(1)
            )
    
    # Minutes played trends
    if 'numMinutes' in df.columns:
        df['minutes_MA5'] = df.groupby('personId')['numMinutes'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df['minutes_trend'] = df.groupby('personId')['numMinutes'].transform(
            lambda x: x.rolling(window=5, min_periods=2).apply(
                lambda y: (y.iloc[-1] - y.iloc[0]) / len(y) if len(y) > 1 else 0
            )
        )
    
    # Usage rate approximation (if we have field goal attempts)
    if 'fieldGoalsAttempted' in df.columns and 'numMinutes' in df.columns:
        df['usage_rate_approx'] = df['fieldGoalsAttempted'] / (df['numMinutes'] + 0.1)
        df['usage_rate_MA5'] = df.groupby('personId')['usage_rate_approx'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    
    # Efficiency metrics
    if 'fieldGoalsMade' in df.columns and 'fieldGoalsAttempted' in df.columns:
        df['fg_efficiency'] = df['fieldGoalsMade'] / (df['fieldGoalsAttempted'] + 0.1)
        df['fg_efficiency_MA5'] = df.groupby('personId')['fg_efficiency'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    
    # Player experience (number of games played)
    df['games_played'] = df.groupby('personId').cumcount() + 1
    
    return df


def create_game_context_features(df):
    """
    Create game context features like home/away, days of rest, etc.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with player statistics
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added game context features
    """
    df = df.copy()
    
    # Ensure date column exists
    if 'gameDate' not in df.columns and 'gameDateTimeEst' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDateTimeEst'], format='mixed', errors='coerce', utc=True)
        # Remove timezone info to avoid mixing timezone-aware and naive datetimes
        if pd.api.types.is_datetime64_any_dtype(df['gameDate']):
            if df['gameDate'].dt.tz is not None:
                df['gameDate'] = df['gameDate'].dt.tz_localize(None)
    
    # Sort by player and date
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    # Home/Away (already exists, but ensure it's binary)
    if 'home' in df.columns:
        df['is_home'] = df['home'].astype(int)
    else:
        df['is_home'] = 0  # Default if not available
    
    # Days of rest
    df['days_rest'] = df.groupby('personId')['gameDate'].diff().dt.days.fillna(0)
    df['days_rest'] = df['days_rest'].clip(lower=0, upper=7)  # Cap at 7 days
    
    # Back-to-back indicator
    df['is_back_to_back'] = (df['days_rest'] == 1).astype(int)
    
    # Day of week
    if 'gameDate' in df.columns:
        df['day_of_week'] = df['gameDate'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Month/Season
    if 'gameDate' in df.columns:
        df['month'] = df['gameDate'].dt.month
        df['year'] = df['gameDate'].dt.year
        # NBA season typically Oct-Apr, but we'll use month as feature
        df['is_playoff_season'] = ((df['month'] >= 4) & (df['month'] <= 6)).astype(int)
    
    # Win/Loss indicator
    if 'win' in df.columns:
        df['is_win'] = df['win'].astype(int)
    else:
        df['is_win'] = 0
    
    # Opponent team encoding (simple numeric encoding)
    if 'opponentteamName' in df.columns:
        opponent_encoding = {team: idx for idx, team in enumerate(df['opponentteamName'].unique())}
        df['opponent_encoded'] = df['opponentteamName'].map(opponent_encoding)
    
    # Player team encoding
    if 'playerteamName' in df.columns:
        team_encoding = {team: idx for idx, team in enumerate(df['playerteamName'].unique())}
        df['team_encoded'] = df['playerteamName'].map(team_encoding)
    
    return df


def create_advanced_features(df):
    """
    Create advanced features like player vs opponent historical performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with player statistics
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added advanced features
    """
    df = df.copy()
    
    # Player vs opponent historical average (if we have enough data)
    if 'opponentteamName' in df.columns and 'fantasyScore' in df.columns:
        # This is computationally expensive, so we'll do a simplified version
        # Average fantasy score against this opponent in last 5 games
        df = df.sort_values(['personId', 'opponentteamName', 'gameDate']).reset_index(drop=True)
        
        df['vs_opponent_avg'] = df.groupby(['personId', 'opponentteamName'])['fantasyScore'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
    
    # Team performance indicators (average team fantasy score)
    if 'playerteamName' in df.columns and 'fantasyScore' in df.columns:
        df = df.sort_values(['playerteamName', 'gameDate']).reset_index(drop=True)
        df['team_avg_fantasy'] = df.groupby(['playerteamName', 'gameDate'])['fantasyScore'].transform('mean')
        df['team_avg_fantasy_MA5'] = df.groupby('playerteamName')['team_avg_fantasy'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
    
    return df


def create_all_features(df):
    """
    Create all features by combining all feature engineering functions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame with player statistics
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all engineered features
    """
    import time
    start_time = time.time()
    print(f"Starting feature engineering on {len(df):,} rows...")
    
    print("Creating temporal features...")
    df = create_temporal_features(df)
    print(f"  Temporal features done. Shape: {df.shape}, Time: {time.time()-start_time:.1f}s")
    
    print("Creating player-specific features...")
    df = create_player_specific_features(df)
    print(f"  Player-specific features done. Shape: {df.shape}, Time: {time.time()-start_time:.1f}s")
    
    print("Creating game context features...")
    df = create_game_context_features(df)
    print(f"  Game context features done. Shape: {df.shape}, Time: {time.time()-start_time:.1f}s")
    
    print("Creating advanced features...")
    df = create_advanced_features(df)
    print(f"  Advanced features done. Shape: {df.shape}, Total time: {time.time()-start_time:.1f}s")
    
    return df


def get_feature_list(df):
    """
    Get list of all feature columns (excluding target and identifiers).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    
    Returns:
    --------
    list
        List of feature column names
    """
    # Columns to exclude
    exclude_cols = [
        'fantasyScore',  # Target variable
        'firstName', 'lastName',  # Identifiers
        'personId', 'gameId',  # IDs
        'gameDateTimeEst', 'gameDate',  # Dates (we use derived features)
        'playerteamCity', 'playerteamName',  # Team names (we use encoded versions)
        'opponentteamCity', 'opponentteamName',  # Opponent names (we use encoded versions)
        'gameType', 'gameLabel', 'gameSubLabel',  # Game metadata
        'seriesGameNumber',  # Mostly missing
    ]
    
    # Get all columns that are not excluded
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter to numeric columns only (for modeling)
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    return numeric_features


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Load data
    from scripts.data_processing.load_data import load_player_statistics
    from scripts.utils.fantasy_scoring import add_fantasy_score_column
    
    print("Loading dataset...")
    df = load_player_statistics(save_raw=False)
    
    print("Applying fantasy scoring...")
    df = add_fantasy_score_column(df)
    
    print("Creating features...")
    df = create_all_features(df)
    
    # Get feature list
    features = get_feature_list(df)
    print(f"\nTotal features created: {len(features)}")
    print(f"Feature columns: {features[:10]}...")  # Show first 10
    
    # Save processed data
    processed_path = project_root / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_path / "player_statistics_features.csv"
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data with features saved to: {output_file}")
    print(f"Final dataset shape: {df.shape}")

