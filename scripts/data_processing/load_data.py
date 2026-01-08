"""
Data loading script for NBA Fantasy Score Prediction Project.

This script loads the PlayerStatistics.csv dataset from Kaggle using kagglehub.
"""

import os
import sys
import pandas as pd
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError:
    print("kagglehub not installed. Please install it using: pip install kagglehub[pandas-datasets]")
    sys.exit(1)


def load_config():
    """Load configuration from config.yaml"""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_player_statistics(file_path=None, save_raw=True):
    """
    Load PlayerStatistics.csv dataset from Kaggle.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the specific file to load. If None, uses config default.
    save_raw : bool, default True
        Whether to save the raw data to data/raw/ directory.
    
    Returns:
    --------
    pd.DataFrame
        The loaded dataset
    """
    config = load_config()
    
    if file_path is None:
        file_path = config['data']['main_file']
    
    dataset_name = config['data']['dataset_name']
    raw_path = project_root / config['data']['raw_path']
    
    print(f"Loading dataset: {dataset_name}")
    print(f"File: {file_path}")
    
    try:
        # Load the dataset
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset_name,
            file_path,
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Save raw data if requested
        if save_raw:
            raw_path.mkdir(parents=True, exist_ok=True)
            output_file = raw_path / file_path
            df.to_csv(output_file, index=False)
            print(f"Raw data saved to: {output_file}")
        
        return df
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def explore_dataset(df):
    """
    Perform initial exploration of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to explore
    """
    print("\n" + "="*80)
    print("DATASET EXPLORATION")
    print("="*80)
    
    # Basic info
    print("\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    
    # Column info
    print("\n2. Column Information:")
    print(f"   Column names: {list(df.columns)}")
    print(f"\n   Data types:")
    print(df.dtypes)
    
    # Missing values
    print("\n3. Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("   No missing values found!")
    
    # Basic statistics
    print("\n4. Basic Statistics:")
    print(df.describe())
    
    # First few rows
    print("\n5. First 5 Rows:")
    print(df.head())
    
    # Data types summary
    print("\n6. Data Types Summary:")
    print(df.dtypes.value_counts())
    
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': missing.to_dict() if len(missing_df) > 0 else {},
        'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }


if __name__ == "__main__":
    # Load the dataset
    df = load_player_statistics()
    
    # Explore the dataset
    exploration_results = explore_dataset(df)
    
    print("\n" + "="*80)
    print("Exploration complete!")
    print("="*80)


