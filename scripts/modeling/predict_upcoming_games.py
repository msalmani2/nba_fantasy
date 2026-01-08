"""
Predict fantasy scores for upcoming games (next game for each player).

This script predicts what players will score in their NEXT game based on
their recent performance and trends.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.load_data import load_player_statistics
from scripts.modeling.predict_by_teams import filter_by_teams, predict_fantasy_scores


def predict_next_game_for_teams(team_names, top_n=20):
    """
    Predict fantasy scores for the next game for players on specified teams.
    
    This uses each player's most recent game data to predict their next performance.
    
    Parameters:
    -----------
    team_names : list
        List of team names
    top_n : int
        Number of top players to return
    
    Returns:
    --------
    pd.DataFrame
        Predictions for next games
    """
    print(f"\n{'='*80}")
    print(f"PREDICTING NEXT GAME FOR: {', '.join(team_names)}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading latest data...")
    df = load_player_statistics(save_raw=False)
    
    # Convert date
    df['gameDate'] = pd.to_datetime(df['gameDateTimeEst'], format='mixed', errors='coerce', utc=True)
    if pd.api.types.is_datetime64_any_dtype(df['gameDate']):
        if df['gameDate'].dt.tz is not None:
            df['gameDate'] = df['gameDate'].dt.tz_localize(None)
    
    # Filter by teams
    print(f"Filtering by teams: {', '.join(team_names)}")
    team_df = filter_by_teams(df, team_names)
    
    if len(team_df) == 0:
        print(f"\n⚠ No players found for teams: {team_names}")
        return None
    
    # Filter to recent players only (last 2 months to get current roster)
    max_date = team_df['gameDate'].max()
    cutoff_date = max_date - pd.DateOffset(months=2)
    
    print(f"\nFiltering to recent players (played in last 2 months)...")
    print(f"  Date range: {cutoff_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    recent_team_df = team_df[team_df['gameDate'] >= cutoff_date].copy()
    
    if len(recent_team_df) == 0:
        print(f"\n⚠ No recent games found for these teams")
        return None
    
    # Get most recent game for each player
    print("\nFinding most recent game for each player...")
    recent_team_df = recent_team_df.sort_values(['personId', 'gameDate'], ascending=[True, False])
    
    # Get the most recent game per player
    most_recent = recent_team_df.groupby('personId').first().reset_index()
    
    print(f"Found {len(most_recent)} current players")
    print(f"Most recent game dates: {most_recent['gameDate'].min()} to {most_recent['gameDate'].max()}")
    
    # Generate predictions based on their most recent performance
    print("\nGenerating predictions for next game...")
    df_with_predictions = predict_fantasy_scores(most_recent, use_ensemble=True)
    
    # Sort by predicted score
    df_with_predictions = df_with_predictions.sort_values('predicted_fantasy_score', ascending=False)
    
    # Format output
    result_df = df_with_predictions[[
        'firstName', 'lastName', 'playerteamName', 'gameDate', 
        'fantasyScore', 'predicted_fantasy_score'
    ]].copy()
    
    result_df['Player'] = result_df['firstName'] + ' ' + result_df['lastName']
    result_df = result_df.drop(['firstName', 'lastName'], axis=1)
    result_df = result_df.rename(columns={
        'playerteamName': 'Team',
        'predicted_fantasy_score': 'Predicted Next Game',
        'fantasyScore': 'Last Game Score',
        'gameDate': 'Last Game Date'
    })
    
    # Reorder columns
    cols = ['Player', 'Team', 'Last Game Date', 'Last Game Score', 'Predicted Next Game']
    result_df = result_df[cols]
    
    # Add a note that this is for next game
    result_df['Status'] = 'Next Game Prediction'
    
    # Limit to top N
    if top_n:
        result_df = result_df.head(top_n)
    
    return result_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict next game for teams')
    parser.add_argument('--teams', '-t', nargs='+', required=True, help='Team names')
    parser.add_argument('--top', '-n', type=int, default=20, help='Top N players')
    
    args = parser.parse_args()
    
    result_df = predict_next_game_for_teams(args.teams, args.top)
    
    if result_df is not None and len(result_df) > 0:
        print(f"\n{'='*80}")
        print("NEXT GAME PREDICTIONS (Based on Recent Performance)")
        print(f"{'='*80}\n")
        
        # Remove Status column for display
        display_df = result_df.drop('Status', axis=1)
        print(display_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("Note: Predictions are for each player's NEXT game based on their")
        print("most recent performance and trends. These are forward-looking predictions.")
        print(f"{'='*80}\n")
        
        # Save option
        save_file = f"next_game_predictions_{'_'.join(args.teams)}.csv"
        output_path = project_root / "models" / "predictions" / save_file
        result_df.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to: {output_path}")
    else:
        print("\nNo results found.")

