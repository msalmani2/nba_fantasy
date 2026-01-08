"""
Quick script to get predictions for recent games (December 2025) by team.

This script automatically filters to the most recent games and generates
predictions if needed.
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.load_data import load_player_statistics
from scripts.modeling.predict_by_teams import filter_by_teams, predict_fantasy_scores


def get_recent_team_predictions(team_names, months_back=1, top_n=20):
    """
    Get predictions for teams from recent games only.
    
    Parameters:
    -----------
    team_names : list
        List of team names
    months_back : int
        Number of months back to include (default: 1 for December 2025)
    top_n : int
        Number of top players to return
    
    Returns:
    --------
    pd.DataFrame
        Predictions for recent games
    """
    print(f"\n{'='*80}")
    print(f"GETTING RECENT PREDICTIONS FOR: {', '.join(team_names)}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading latest data...")
    df = load_player_statistics(save_raw=False)
    
    # Convert date
    df['gameDate'] = pd.to_datetime(df['gameDateTimeEst'], format='mixed', errors='coerce', utc=True)
    if pd.api.types.is_datetime64_any_dtype(df['gameDate']):
        if df['gameDate'].dt.tz is not None:
            df['gameDate'] = df['gameDate'].dt.tz_localize(None)
    
    # Filter to recent games
    max_date = df['gameDate'].max()
    cutoff_date = max_date - pd.DateOffset(months=months_back)
    
    print(f"Filtering to recent games:")
    print(f"  Date range: {cutoff_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    recent_df = df[df['gameDate'] >= cutoff_date].copy()
    print(f"  Recent games: {len(recent_df):,}")
    
    # Filter by teams
    print(f"\nFiltering by teams: {', '.join(team_names)}")
    team_df = filter_by_teams(recent_df, team_names)
    print(f"  Team games: {len(team_df):,}")
    
    if len(team_df) == 0:
        print("\n⚠ No games found for these teams in the recent period")
        return None
    
    # Generate predictions
    print("\nGenerating predictions...")
    df_with_predictions = predict_fantasy_scores(team_df, use_ensemble=True)
    
    # Get most recent game per player
    df_with_predictions = df_with_predictions.sort_values(['personId', 'gameDate'], ascending=[True, False])
    df_with_predictions = df_with_predictions.groupby('personId').first().reset_index()
    
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
        'predicted_fantasy_score': 'Predicted Score',
        'fantasyScore': 'Actual Score',
        'gameDate': 'Game Date'
    })
    
    # Reorder columns
    cols = ['Player', 'Team', 'Game Date', 'Actual Score', 'Predicted Score']
    result_df = result_df[cols]
    
    # Limit to top N
    if top_n:
        result_df = result_df.head(top_n)
    
    return result_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Get recent team predictions')
    parser.add_argument('--teams', '-t', nargs='+', required=True, help='Team names')
    parser.add_argument('--top', '-n', type=int, default=20, help='Top N players')
    parser.add_argument('--months', type=int, default=1, help='Months back (default: 1)')
    
    args = parser.parse_args()
    
    result_df = get_recent_team_predictions(args.teams, args.months, args.top)
    
    if result_df is not None and len(result_df) > 0:
        print(f"\n{'='*80}")
        print("TOP PLAYERS BY PREDICTED FANTASY SCORE")
        print(f"{'='*80}\n")
        print(result_df.to_string(index=False))
        print(f"\n{'='*80}\n")
    else:
        print("\nNo results found.")


