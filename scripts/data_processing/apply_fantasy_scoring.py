"""
Script to apply fantasy scoring to the dataset and save processed data.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.load_data import load_player_statistics
from scripts.utils.fantasy_scoring import add_fantasy_score_column


def apply_fantasy_scoring_to_dataset():
    """
    Load the dataset, apply fantasy scoring, and save processed data.
    """
    print("Loading dataset...")
    df = load_player_statistics(save_raw=False)  # Don't save raw again
    
    print("\nApplying fantasy scoring...")
    df = add_fantasy_score_column(df)
    
    print(f"\nFantasy score statistics:")
    print(df['fantasyScore'].describe())
    
    # Save processed data
    processed_path = project_root / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_path / "player_statistics_with_fantasy.csv"
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    df = apply_fantasy_scoring_to_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


