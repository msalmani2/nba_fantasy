"""
Data preprocessing and train/validation/test split script.

This script handles:
- Data cleaning (missing values, outliers)
- Feature encoding
- Temporal train/test split
- Feature selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.feature_engineering import get_feature_list


def load_config():
    """Load configuration from config.yaml"""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    df = df.copy()
    
    print("Cleaning data...")
    print(f"Initial shape: {df.shape}")
    
    # Remove rows where player didn't play (all stats are NaN)
    stat_cols = ['points', 'assists', 'reboundsTotal', 'steals', 'blocks', 'turnovers']
    missing_stats = df[stat_cols].isnull().all(axis=1)
    df = df[~missing_stats].copy()
    print(f"Removed {missing_stats.sum()} rows with missing stats (DNP games)")
    
    # Fill missing values in feature columns
    # For rolling averages and derived features, fill with 0 or forward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'fantasyScore':  # Don't fill target variable
            if col.startswith(('MA', '_MA', '_last', '_career', '_season')):
                # For rolling averages, forward fill within each player
                df[col] = df.groupby('personId')[col].fillna(method='ffill').fillna(0)
            else:
                df[col] = df[col].fillna(0)
    
    # Handle outliers in fantasy score (cap extreme values)
    if 'fantasyScore' in df.columns:
        q1 = df['fantasyScore'].quantile(0.01)
        q99 = df['fantasyScore'].quantile(0.99)
        df = df[(df['fantasyScore'] >= q1) & (df['fantasyScore'] <= q99)].copy()
        print(f"Removed outliers in fantasy score (kept 1st-99th percentile)")
    
    print(f"Final shape: {df.shape}")
    return df


def encode_categorical_features(df):
    """
    Encode categorical features if any remain.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded features
    """
    df = df.copy()
    
    # Most categorical features should already be encoded in feature engineering
    # But we'll handle any remaining ones
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Remove identifier columns
    categorical_cols = [col for col in categorical_cols 
                       if col not in ['firstName', 'lastName', 'gameDateTimeEst', 
                                     'playerteamCity', 'playerteamName',
                                     'opponentteamCity', 'opponentteamName',
                                     'gameType', 'gameLabel', 'gameSubLabel']]
    
    if len(categorical_cols) > 0:
        print(f"Encoding {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def temporal_train_test_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, 
                              date_col='gameDate', feature_cols=None, random_state=42):
    """
    Perform temporal train/validation/test split.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features
    train_ratio : float
        Proportion of data for training
    val_ratio : float
        Proportion of data for validation
    test_ratio : float
        Proportion of data for testing
    date_col : str
        Name of date column
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    """
    print("Performing temporal train/validation/test split...")
    
    # Ensure date column exists
    if date_col not in df.columns and 'gameDateTimeEst' in df.columns:
        df[date_col] = pd.to_datetime(df['gameDateTimeEst'], format='mixed', errors='coerce')
        # Remove timezone info to avoid mixing timezone-aware and naive datetimes
        if df[date_col].dt.tz is not None:
            df[date_col] = df[date_col].dt.tz_localize(None)
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Get split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split by date
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Train set: {len(train_df)} samples ({train_df[date_col].min()} to {train_df[date_col].max()})")
    print(f"Validation set: {len(val_df)} samples ({val_df[date_col].min()} to {val_df[date_col].max()})")
    print(f"Test set: {len(test_df)} samples ({test_df[date_col].min()} to {test_df[date_col].max()})")
    
    # Get feature columns
    if feature_cols is None:
        feature_cols = get_feature_list(df)
    
    # Separate features and target
    X_train = train_df[feature_cols].copy()
    y_train = train_df['fantasyScore'].copy()
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df['fantasyScore'].copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df['fantasyScore'].copy()
    
    # Handle any remaining NaN values
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def remove_highly_correlated_features(X_train, feature_cols, threshold=0.95):
    """
    Remove highly correlated features.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    feature_cols : list
        List of feature column names
    threshold : float
        Correlation threshold for removal
    
    Returns:
    --------
    list
        List of selected feature columns
    """
    print(f"Removing highly correlated features (threshold={threshold})...")
    
    # Calculate correlation matrix
    corr_matrix = X_train.corr().abs()
    
    # Find pairs of highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    # Keep features that are not highly correlated
    selected_features = [f for f in feature_cols if f not in to_drop]
    
    print(f"Removed {len(to_drop)} highly correlated features")
    print(f"Remaining features: {len(selected_features)}")
    
    return selected_features


def preprocess_data(df, remove_correlated=True, correlation_threshold=0.95):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame with features
    remove_correlated : bool
        Whether to remove highly correlated features
    correlation_threshold : float
        Correlation threshold for removal
    
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    """
    # Clean data
    df = clean_data(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Get initial feature list
    feature_cols = get_feature_list(df)
    
    # Remove highly correlated features if requested
    if remove_correlated:
        X_temp = df[feature_cols].fillna(0)
        feature_cols = remove_highly_correlated_features(X_temp, feature_cols, correlation_threshold)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, final_features = temporal_train_test_split(
        df, feature_cols=feature_cols
    )
    
    # Ensure we only use selected features
    X_train = X_train[final_features]
    X_val = X_val[final_features]
    X_test = X_test[final_features]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, final_features


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Load processed data with features
    processed_path = project_root / "data" / "processed"
    data_file = processed_path / "player_statistics_features.csv"
    
    if not data_file.exists():
        print("Feature-engineered data not found. Running feature engineering...")
        from scripts.data_processing.feature_engineering import create_all_features
        from scripts.data_processing.load_data import load_player_statistics
        from scripts.utils.fantasy_scoring import add_fantasy_score_column
        
        df = load_player_statistics(save_raw=False)
        df = add_fantasy_score_column(df)
        df = create_all_features(df)
    else:
        print(f"Loading feature-engineered data from {data_file}...")
        df = pd.read_csv(data_file)
        if 'gameDate' not in df.columns and 'gameDateTimeEst' in df.columns:
            df['gameDate'] = pd.to_datetime(df['gameDateTimeEst'])
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, features = preprocess_data(df)
    
    print(f"\nPreprocessing complete!")
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Number of features: {len(features)}")
    
    # Save preprocessed data
    preprocessed_path = processed_path / "preprocessed"
    preprocessed_path.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(preprocessed_path / "X_train.csv", index=False)
    X_val.to_csv(preprocessed_path / "X_val.csv", index=False)
    X_test.to_csv(preprocessed_path / "X_test.csv", index=False)
    y_train.to_csv(preprocessed_path / "y_train.csv", index=False)
    y_val.to_csv(preprocessed_path / "y_val.csv", index=False)
    y_test.to_csv(preprocessed_path / "y_test.csv", index=False)
    
    # Save feature list
    with open(preprocessed_path / "feature_list.pkl", 'wb') as f:
        pickle.dump(features, f)
    
    print(f"\nPreprocessed data saved to {preprocessed_path}")

