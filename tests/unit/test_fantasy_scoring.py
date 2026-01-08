"""
Unit tests for fantasy scoring calculations.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.fantasy_scoring import (
    calculate_fanduel_score,
    calculate_fanduel_scores,
    add_fantasy_score_column
)


class TestFantasyScoring:
    """Test cases for fantasy scoring functions."""
    
    def test_basic_calculation(self):
        """Test basic fantasy score calculation."""
        row = pd.Series({
            'threePointersMade': 4,
            'fieldGoalsMade': 12,  # 4 three-pointers + 8 two-pointers
            'freeThrowsMade': 5,
            'reboundsTotal': 10,
            'assists': 8,
            'steals': 2,
            'blocks': 1,
            'turnovers': 3
        })
        
        score = calculate_fanduel_score(row)
        expected = 4*3 + 8*2 + 5*1 + 10*1.2 + 8*1.5 + 2*3 + 1*3 - 3*1
        assert abs(score - expected) < 0.01, f"Expected {expected}, got {score}"
    
    def test_zero_stats(self):
        """Test calculation with all zeros."""
        row = pd.Series({
            'points': 0,
            'reboundsTotal': 0,
            'assists': 0,
            'steals': 0,
            'blocks': 0,
            'turnovers': 0
        })
        
        score = calculate_fanduel_score(row)
        assert score == 0.0
    
    def test_high_scoring_game(self):
        """Test calculation for a high-scoring game."""
        row = pd.Series({
            'threePointersMade': 6,
            'fieldGoalsMade': 20,  # 6 three-pointers + 14 two-pointers
            'freeThrowsMade': 8,
            'reboundsTotal': 15,
            'assists': 12,
            'steals': 4,
            'blocks': 3,
            'turnovers': 5
        })
        
        score = calculate_fanduel_score(row)
        expected = 6*3 + 14*2 + 8*1 + 15*1.2 + 12*1.5 + 4*3 + 3*3 - 5*1
        assert abs(score - expected) < 0.01
    
    def test_missing_values(self):
        """Test handling of missing values (NaN)."""
        row = pd.Series({
            'points': 20,
            'reboundsTotal': np.nan,
            'assists': 5,
            'steals': np.nan,
            'blocks': 1,
            'turnovers': 2
        })
        
        score = calculate_fanduel_score(row)
        # NaN values should be treated as 0
        expected = 20*1 + 0*1.2 + 5*1.5 + 0*2 + 1*2 - 2*1
        assert abs(score - expected) < 0.01
    
    def test_dataframe_calculation(self):
        """Test calculation for entire DataFrame."""
        df = pd.DataFrame({
            'points': [10, 20, 30],
            'reboundsTotal': [5, 10, 15],
            'assists': [3, 6, 9],
            'steals': [1, 2, 3],
            'blocks': [0, 1, 2],
            'turnovers': [2, 3, 4]
        })
        
        scores = calculate_fanduel_scores(df)
        assert len(scores) == 3
        assert all(scores >= 0)  # All scores should be non-negative for these examples
    
    def test_add_fantasy_score_column(self):
        """Test adding fantasy score column to DataFrame."""
        df = pd.DataFrame({
            'points': [25],
            'reboundsTotal': [10],
            'assists': [8],
            'steals': [2],
            'blocks': [1],
            'turnovers': [3]
        })
        
        df_with_score = add_fantasy_score_column(df)
        assert 'fantasyScore' in df_with_score.columns
        assert len(df_with_score) == 1
        expected = 25*1 + 10*1.2 + 8*1.5 + 2*2 + 1*2 - 3*1
        assert abs(df_with_score['fantasyScore'].iloc[0] - expected) < 0.01
    
    def test_dict_input(self):
        """Test calculation with dictionary input."""
        row_dict = {
            'threePointersMade': 2,
            'fieldGoalsMade': 7,  # 2 three-pointers + 5 two-pointers
            'freeThrowsMade': 3,
            'reboundsTotal': 8,
            'assists': 5,
            'steals': 1,
            'blocks': 0,
            'turnovers': 2
        }
        
        score = calculate_fanduel_score(row_dict)
        expected = 2*3 + 5*2 + 3*1 + 8*1.2 + 5*1.5 + 1*3 + 0*3 - 2*1
        assert abs(score - expected) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

