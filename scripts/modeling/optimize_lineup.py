"""
Lineup optimizer for daily fantasy sports.

This script optimizes lineups based on:
- Predicted fantasy scores
- Player salaries
- Position requirements
- Salary cap constraints
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def optimize_fanduel_lineup(predictions_df, salary_df, salary_cap=60000):
    """
    Optimize FanDuel lineup.
    
    FanDuel requirements:
    - 1 PG (Point Guard)
    - 1 SG (Shooting Guard)
    - 1 SF (Small Forward)
    - 1 PF (Power Forward)
    - 1 C (Center)
    - 1 G (Guard - PG or SG)
    - 1 F (Forward - SF or PF)
    - 1 UTIL (Any position)
    - Total salary: $60,000
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with predictions
    salary_df : pd.DataFrame
        DataFrame with salaries
    salary_cap : int
        Salary cap (default: 60000 for FanDuel)
    
    Returns:
    --------
    pd.DataFrame
        Optimized lineup
    """
    from scripts.data_processing.load_salary_data import merge_salary_with_predictions
    
    # Merge predictions with salaries
    df = merge_salary_with_predictions(predictions_df, salary_df)
    
    if 'salary' not in df.columns:
        print("⚠ Salary data not available. Cannot optimize lineup.")
        return None
    
    if 'position' not in df.columns:
        print("⚠ Position data not available. Cannot optimize lineup.")
        print("   You may need to add position data manually.")
        return None
    
    # Filter to players with valid data
    df = df[
        df['salary'].notna() & 
        df['predicted_fantasy_score'].notna() &
        (df['salary'] > 0)
    ].copy()
    
    if len(df) == 0:
        print("⚠ No valid players with salary and prediction data")
        return None
    
    print(f"\nOptimizing FanDuel lineup from {len(df)} players...")
    print(f"Salary cap: ${salary_cap:,}")
    
    # Simple greedy algorithm (can be improved with more sophisticated optimization)
    # Sort by value (points per dollar)
    df = df.sort_values('points_per_dollar', ascending=False)
    
    # Build lineup
    lineup = []
    total_salary = 0
    total_points = 0
    
    positions_needed = {
        'PG': 1,
        'SG': 1,
        'SF': 1,
        'PF': 1,
        'C': 1,
        'G': 1,  # PG or SG
        'F': 1,  # SF or PF
        'UTIL': 1  # Any
    }
    
    for _, player in df.iterrows():
        pos = str(player.get('position', '')).upper()
        
        # Check if we can add this player
        if total_salary + player['salary'] > salary_cap:
            continue
        
        # Check position requirements
        added = False
        
        if pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            if positions_needed.get(pos, 0) > 0:
                lineup.append(player)
                positions_needed[pos] -= 1
                total_salary += player['salary']
                total_points += player['predicted_fantasy_score']
                added = True
            elif pos in ['PG', 'SG'] and positions_needed.get('G', 0) > 0:
                lineup.append(player)
                positions_needed['G'] -= 1
                total_salary += player['salary']
                total_points += player['predicted_fantasy_score']
                added = True
            elif pos in ['SF', 'PF'] and positions_needed.get('F', 0) > 0:
                lineup.append(player)
                positions_needed['F'] -= 1
                total_salary += player['salary']
                total_points += player['predicted_fantasy_score']
                added = True
            elif positions_needed.get('UTIL', 0) > 0:
                lineup.append(player)
                positions_needed['UTIL'] -= 1
                total_salary += player['salary']
                total_points += player['predicted_fantasy_score']
                added = True
        
        if len(lineup) >= 8:  # FanDuel requires 8 players
            break
    
    if len(lineup) < 8:
        print(f"⚠ Could not fill complete lineup. Only {len(lineup)} players selected.")
        return None
    
    lineup_df = pd.DataFrame(lineup)
    
    print(f"\n✓ Optimized Lineup:")
    print(f"  Total Salary: ${total_salary:,} / ${salary_cap:,}")
    print(f"  Total Predicted Points: {total_points:.2f}")
    print(f"  Remaining Salary: ${salary_cap - total_salary:,}")
    
    return lineup_df


if __name__ == "__main__":
    print("="*80)
    print("LINEUP OPTIMIZER")
    print("="*80)
    print("\nThis script optimizes daily fantasy lineups based on:")
    print("- Predicted fantasy scores")
    print("- Player salaries")
    print("- Position requirements")
    print("- Salary cap constraints")
    print("\nTo use:")
    print("1. Load salary data (see load_salary_data.py)")
    print("2. Get predictions for players")
    print("3. Run optimizer with salary cap constraints")
    print("\nNote: This is a basic greedy algorithm.")
    print("For production use, consider more sophisticated optimization (linear programming, etc.)")


