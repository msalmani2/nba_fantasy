"""
Enhanced features for NBA fantasy prediction improvements.

This module adds:
1. Last 2-game features
2. Trending indicators
3. Consistency metrics
4. Double-double probability features
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


def add_last_n_games_features(df, n_games=2):
    """
    Add features for last N games (more recent than rolling averages).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with game data (must be sorted by player and date)
    n_games : int
        Number of recent games to consider
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added features
    """
    print(f"Adding last {n_games} games features...")
    
    # Ensure sorted by player and date
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    # Stats to track
    stats = ['points', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'minutesCalculated']
    
    for stat in stats:
        if stat in df.columns:
            # Last N games average
            df[f'{stat}_last{n_games}'] = (
                df.groupby('personId')[stat]
                .transform(lambda x: x.rolling(n_games, min_periods=1).mean().shift(1))
            )
            
            # Last N games max (for boom potential)
            df[f'{stat}_last{n_games}_max'] = (
                df.groupby('personId')[stat]
                .transform(lambda x: x.rolling(n_games, min_periods=1).max().shift(1))
            )
    
    return df


def add_trending_features(df):
    """
    Add trending indicators (is player trending up or down?).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with game data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with trending features
    """
    print("Adding trending indicators...")
    
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    # Fantasy score trend (last 3 vs previous 3)
    if 'fantasy_score' in df.columns:
        # Last 3 games average
        df['fantasy_score_last3'] = (
            df.groupby('personId')['fantasy_score']
            .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
        )
        
        # Previous 3 games average (games 4-6)
        df['fantasy_score_prev3'] = (
            df.groupby('personId')['fantasy_score']
            .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(4))
        )
        
        # Trend indicator: positive if improving, negative if declining
        df['fantasy_score_trend'] = df['fantasy_score_last3'] - df['fantasy_score_prev3']
        
        # Binary trend indicator
        df['trending_up'] = (df['fantasy_score_trend'] > 0).astype(int)
    
    # Minutes trend (playing time increasing/decreasing)
    if 'minutesCalculated' in df.columns:
        df['minutes_last3'] = (
            df.groupby('personId')['minutesCalculated']
            .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
        )
        
        df['minutes_prev3'] = (
            df.groupby('personId')['minutesCalculated']
            .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(4))
        )
        
        df['minutes_trend'] = df['minutes_last3'] - df['minutes_prev3']
        df['minutes_increasing'] = (df['minutes_trend'] > 0).astype(int)
    
    return df


def add_consistency_metrics(df):
    """
    Add consistency metrics (standard deviation, coefficient of variation).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with game data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with consistency features
    """
    print("Adding consistency metrics...")
    
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    # Fantasy score consistency (last 5 games)
    if 'fantasy_score' in df.columns:
        # Standard deviation (lower = more consistent)
        df['fantasy_score_std5'] = (
            df.groupby('personId')['fantasy_score']
            .transform(lambda x: x.rolling(5, min_periods=2).std().shift(1))
        )
        
        # Coefficient of variation (std / mean)
        df['fantasy_score_cv5'] = (
            df['fantasy_score_std5'] / (df['fantasy_score_MA5'] + 1e-6)
        )
        
        # Min/max range (last 5 games)
        df['fantasy_score_min5'] = (
            df.groupby('personId')['fantasy_score']
            .transform(lambda x: x.rolling(5, min_periods=1).min().shift(1))
        )
        
        df['fantasy_score_max5'] = (
            df.groupby('personId')['fantasy_score']
            .transform(lambda x: x.rolling(5, min_periods=1).max().shift(1))
        )
        
        df['fantasy_score_range5'] = df['fantasy_score_max5'] - df['fantasy_score_min5']
        
        # Consistency score (inverse of CV, higher = more consistent)
        df['fantasy_score_consistency'] = 1 / (1 + df['fantasy_score_cv5'])
    
    return df


def add_double_double_features(df):
    """
    Add double-double probability and related features.
    
    A double-double is 10+ in two of: points, rebounds, assists, steals, blocks
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with game data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with double-double features
    """
    print("Adding double-double features...")
    
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    # Calculate double-double for current game
    stat_cols = ['points', 'reboundsTotal', 'assists', 'steals', 'blocks']
    available_cols = [col for col in stat_cols if col in df.columns]
    
    if len(available_cols) >= 2:
        # Count how many stats are >= 10
        double_double_count = sum([(df[col] >= 10).astype(int) for col in available_cols])
        df['is_double_double'] = (double_double_count >= 2).astype(int)
        
        # Historical double-double rate (last 10 games)
        df['double_double_rate10'] = (
            df.groupby('personId')['is_double_double']
            .transform(lambda x: x.rolling(10, min_periods=3).mean().shift(1))
        )
        
        # Games with at least one stat >= 10 (last 5 games)
        for col in available_cols:
            df[f'{col}_ge10_rate5'] = (
                df.groupby('personId')[col]
                .transform(lambda x: (x >= 10).astype(int).rolling(5, min_periods=1).mean().shift(1))
            )
    
    return df


def add_game_context_enhanced(df):
    """
    Add enhanced game context features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with game data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with enhanced context features
    """
    print("Adding enhanced game context features...")
    
    df = df.sort_values(['personId', 'gameDate']).reset_index(drop=True)
    
    # Days since last game
    df['gameDate_dt'] = pd.to_datetime(df['gameDate'])
    df['days_since_last_game'] = (
        df.groupby('personId')['gameDate_dt']
        .transform(lambda x: x.diff().dt.days)
    )
    
    # Back-to-back games indicator
    df['is_back_to_back'] = (df['days_since_last_game'] == 1).astype(int)
    
    # Games in last 7 days (fatigue indicator)
    df['games_last_7days'] = (
        df.groupby('personId')['gameDate_dt']
        .transform(lambda x: x.rolling('7D', on='gameDate_dt').count() - 1)  # Exclude current game
    )
    
    # Rest advantage (more rest = potentially better performance)
    df['well_rested'] = (df['days_since_last_game'] >= 3).astype(int)
    
    return df


def add_all_enhanced_features(df):
    """
    Add all enhanced features at once.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with game data
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all enhanced features
    """
    print("\n" + "="*80)
    print("ADDING ENHANCED FEATURES")
    print("="*80)
    
    initial_cols = len(df.columns)
    
    # Add features
    df = add_last_n_games_features(df, n_games=2)
    df = add_trending_features(df)
    df = add_consistency_metrics(df)
    df = add_double_double_features(df)
    df = add_game_context_enhanced(df)
    
    new_cols = len(df.columns) - initial_cols
    print(f"\n✓ Added {new_cols} enhanced features")
    print(f"  Total features now: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    print("Enhanced features module loaded successfully!")
    print("\nAvailable functions:")
    print("  - add_last_n_games_features()")
    print("  - add_trending_features()")
    print("  - add_consistency_metrics()")
    print("  - add_double_double_features()")
    print("  - add_game_context_enhanced()")
    print("  - add_all_enhanced_features()")

