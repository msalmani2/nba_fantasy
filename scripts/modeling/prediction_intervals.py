"""
Prediction intervals for fantasy score predictions.

This module provides confidence intervals for predictions using:
1. Ensemble variance (disagreement between models)
2. Historical prediction errors
3. Player-specific uncertainty
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def calculate_ensemble_variance(predictions_dict):
    """
    Calculate variance across ensemble models.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary with model names as keys and predictions as values
    
    Returns:
    --------
    np.ndarray
        Variance of predictions across models
    """
    # Stack predictions from all models
    all_preds = np.array([preds for preds in predictions_dict.values()])
    
    # Calculate variance across models
    variance = np.var(all_preds, axis=0)
    
    return variance


def calculate_prediction_intervals_ensemble(predictions_dict, confidence=0.80):
    """
    Calculate prediction intervals using ensemble disagreement.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary with model names as keys and predictions as values
    confidence : float
        Confidence level (0.80 = 80%, 0.95 = 95%)
    
    Returns:
    --------
    tuple
        (mean_prediction, lower_bound, upper_bound, std_dev)
    """
    # Stack predictions
    all_preds = np.array([preds for preds in predictions_dict.values()])
    
    # Calculate mean and std
    mean_pred = np.mean(all_preds, axis=0)
    std_pred = np.std(all_preds, axis=0)
    
    # Calculate Z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate intervals
    lower_bound = mean_pred - z_score * std_pred
    upper_bound = mean_pred + z_score * std_pred
    
    return mean_pred, lower_bound, upper_bound, std_pred


def calculate_historical_errors(y_true, y_pred, percentiles=[10, 90]):
    """
    Calculate historical prediction errors for calibration.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    percentiles : list
        Percentiles to calculate (e.g., [10, 90] for 80% interval)
    
    Returns:
    --------
    dict
        Dictionary with error percentiles
    """
    errors = np.array(y_true) - np.array(y_pred)
    
    error_stats = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'mae': np.mean(np.abs(errors)),
    }
    
    for p in percentiles:
        error_stats[f'percentile_{p}'] = np.percentile(errors, p)
    
    return error_stats


def add_prediction_intervals_to_df(df, predictions_dict, confidence_levels=[0.80, 0.95]):
    """
    Add prediction intervals to a DataFrame with predictions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with predictions
    predictions_dict : dict
        Dictionary with model predictions
    confidence_levels : list
        List of confidence levels to calculate
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added interval columns
    """
    df = df.copy()
    
    for conf in confidence_levels:
        mean_pred, lower, upper, std = calculate_prediction_intervals_ensemble(
            predictions_dict, confidence=conf
        )
        
        conf_pct = int(conf * 100)
        df[f'predicted_fantasy_score'] = mean_pred
        df[f'lower_{conf_pct}'] = lower
        df[f'upper_{conf_pct}'] = upper
        df[f'std_dev'] = std
        df[f'interval_width_{conf_pct}'] = upper - lower
    
    # Add consistency score (inverse of uncertainty)
    df['consistency_score'] = 1 / (1 + df['std_dev'])
    
    # Add risk category
    df['risk_category'] = pd.cut(
        df['std_dev'],
        bins=[0, 2, 4, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    return df


def calculate_player_specific_uncertainty(df, player_col='personId', score_col='fantasy_score', window=10):
    """
    Calculate player-specific uncertainty based on historical volatility.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with historical scores
    player_col : str
        Column name for player ID
    score_col : str
        Column name for fantasy score
    window : int
        Rolling window for volatility calculation
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with player-specific uncertainty metrics
    """
    df = df.sort_values([player_col, 'gameDate']).copy()
    
    # Calculate rolling standard deviation
    df['historical_volatility'] = (
        df.groupby(player_col)[score_col]
        .transform(lambda x: x.rolling(window, min_periods=3).std())
    )
    
    # Calculate coefficient of variation
    df['historical_mean'] = (
        df.groupby(player_col)[score_col]
        .transform(lambda x: x.rolling(window, min_periods=3).mean())
    )
    
    df['historical_cv'] = df['historical_volatility'] / (df['historical_mean'] + 1e-6)
    
    return df


def adjust_intervals_for_player_history(df, base_std_col='std_dev', historical_vol_col='historical_volatility'):
    """
    Adjust prediction intervals based on player's historical volatility.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with base intervals and historical volatility
    base_std_col : str
        Column name for base standard deviation
    historical_vol_col : str
        Column name for historical volatility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with adjusted intervals
    """
    df = df.copy()
    
    # Combine ensemble uncertainty with historical volatility
    df['adjusted_std'] = np.sqrt(
        df[base_std_col]**2 + df[historical_vol_col]**2
    )
    
    # Recalculate intervals with adjusted std
    for conf in [0.80, 0.95]:
        z_score = stats.norm.ppf((1 + conf) / 2)
        conf_pct = int(conf * 100)
        
        df[f'adjusted_lower_{conf_pct}'] = df['predicted_fantasy_score'] - z_score * df['adjusted_std']
        df[f'adjusted_upper_{conf_pct}'] = df['predicted_fantasy_score'] + z_score * df['adjusted_std']
        df[f'adjusted_width_{conf_pct}'] = df[f'adjusted_upper_{conf_pct}'] - df[f'adjusted_lower_{conf_pct}']
    
    return df


def format_prediction_with_interval(pred, lower, upper, confidence=80):
    """
    Format a prediction with its interval for display.
    
    Parameters:
    -----------
    pred : float
        Predicted value
    lower : float
        Lower bound
    upper : float
        Upper bound
    confidence : int
        Confidence level percentage
    
    Returns:
    --------
    str
        Formatted string
    """
    interval_width = upper - lower
    return f"{pred:.1f} (±{interval_width/2:.1f}, {confidence}% CI: [{lower:.1f}, {upper:.1f}])"


def load_predictions_with_intervals(predictions_path, models_path=None):
    """
    Load predictions and add confidence intervals.
    
    Parameters:
    -----------
    predictions_path : str or Path
        Path to predictions CSV
    models_path : str or Path, optional
        Path to saved models (for loading individual predictions)
    
    Returns:
    --------
    pd.DataFrame
        Predictions with confidence intervals
    """
    df = pd.read_csv(predictions_path)
    
    if models_path and Path(models_path).exists():
        # Load individual model predictions
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        
        print(f"Loaded {len(models)} models for interval calculation")
        
        # Note: This would require re-predicting with each model
        # For now, use a simplified approach based on ensemble_prediction column
    
    # If ensemble predictions are available, calculate intervals
    if 'predicted_fantasy_score' in df.columns:
        # Use a default uncertainty estimate
        df['std_dev'] = df['predicted_fantasy_score'] * 0.10  # 10% uncertainty
        
        for conf in [0.80, 0.95]:
            z_score = stats.norm.ppf((1 + conf) / 2)
            conf_pct = int(conf * 100)
            
            df[f'lower_{conf_pct}'] = df['predicted_fantasy_score'] - z_score * df['std_dev']
            df[f'upper_{conf_pct}'] = df['predicted_fantasy_score'] + z_score * df['std_dev']
            df[f'interval_width_{conf_pct}'] = df[f'upper_{conf_pct}'] - df[f'lower_{conf_pct}']
        
        df['consistency_score'] = 1 / (1 + df['std_dev'])
        df['risk_category'] = pd.cut(
            df['std_dev'],
            bins=[0, 2, 4, 100],
            labels=['Low', 'Medium', 'High']
        )
    
    return df


if __name__ == "__main__":
    print("Prediction intervals module loaded successfully!")
    print("\nAvailable functions:")
    print("  - calculate_ensemble_variance()")
    print("  - calculate_prediction_intervals_ensemble()")
    print("  - add_prediction_intervals_to_df()")
    print("  - calculate_player_specific_uncertainty()")
    print("  - format_prediction_with_interval()")

