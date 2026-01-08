"""
Predict fantasy scores for players on specific teams.

Usage:
    python scripts/modeling/predict_by_teams.py --teams "Lakers" "Warriors" "Celtics"
    python scripts/modeling/predict_by_teams.py --teams "Los Angeles Lakers" "Golden State Warriors"
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.modeling.predict import predict_fantasy_scores, load_base_models, load_ensemble_model
from scripts.data_processing.load_data import load_player_statistics


def get_available_teams():
    """Get list of available team names from the dataset."""
    print("Loading dataset to get team names...")
    df = load_player_statistics(save_raw=False)
    
    # Get unique team names
    teams = sorted(df['playerteamName'].dropna().unique().tolist())
    return teams


def filter_by_teams(df, team_names):
    """
    Filter dataframe by team names (case-insensitive, partial matching).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with player statistics
    team_names : list
        List of team names to filter by
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    if 'playerteamName' not in df.columns:
        raise ValueError("DataFrame must contain 'playerteamName' column")
    
    # Create a mask for matching teams
    mask = pd.Series([False] * len(df))
    
    for team_name in team_names:
        # Case-insensitive partial matching
        team_mask = df['playerteamName'].str.contains(
            team_name, case=False, na=False, regex=False
        )
        mask = mask | team_mask
    
    filtered_df = df[mask].copy()
    
    return filtered_df


def predict_for_teams(team_names, use_latest_data=True, top_n=None, recent_only=True, months_back=3):
    """
    Get predictions for players on specified teams.
    
    Parameters:
    -----------
    team_names : list
        List of team names
    use_latest_data : bool
        Whether to use latest data or existing predictions
    top_n : int, optional
        Return only top N players by predicted score
    recent_only : bool
        If True, only show recent games (last N months)
    months_back : int
        Number of months back to include (default: 3)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions for players on specified teams
    """
    print(f"\n{'='*80}")
    print(f"PREDICTING FOR TEAMS: {', '.join(team_names)}")
    print(f"{'='*80}\n")
    
    if use_latest_data:
        print("Loading latest data...")
        df = load_player_statistics(save_raw=False)
        
        print("Filtering by teams...")
        df_filtered = filter_by_teams(df, team_names)
        
        if len(df_filtered) == 0:
            print(f"\n⚠ No players found for teams: {team_names}")
            print("\nAvailable teams:")
            available_teams = get_available_teams()
            for team in available_teams[:20]:  # Show first 20
                print(f"  - {team}")
            if len(available_teams) > 20:
                print(f"  ... and {len(available_teams) - 20} more")
            return None
        
        print(f"Found {len(df_filtered)} player-game records for specified teams")
        
        # Get unique players
        unique_players = df_filtered[['personId', 'firstName', 'lastName', 'playerteamName']].drop_duplicates()
        print(f"Unique players: {len(unique_players)}")
        
        # Make predictions
        print("\nGenerating predictions...")
        df_with_predictions = predict_fantasy_scores(df_filtered, use_ensemble=True)
        
    else:
        # Use existing predictions file
        predictions_file = project_root / "models" / "predictions" / "predictions.csv"
        if not predictions_file.exists():
            print("No existing predictions found. Generating new predictions...")
            return predict_for_teams(team_names, use_latest_data=True, top_n=top_n, 
                                   recent_only=recent_only, months_back=months_back)
        
        print("Loading existing predictions...")
        df_with_predictions = pd.read_csv(predictions_file)
        
        print("Filtering by teams...")
        df_with_predictions = filter_by_teams(df_with_predictions, team_names)
        
        if len(df_with_predictions) == 0:
            print(f"\n⚠ No players found for teams: {team_names}")
            return None
        
        # Convert date column if needed
        if 'gameDateTimeEst' in df_with_predictions.columns and 'gameDate' not in df_with_predictions.columns:
            df_with_predictions['gameDate'] = pd.to_datetime(
                df_with_predictions['gameDateTimeEst'], format='mixed', errors='coerce', utc=True
            )
            if pd.api.types.is_datetime64_any_dtype(df_with_predictions['gameDate']):
                if df_with_predictions['gameDate'].dt.tz is not None:
                    df_with_predictions['gameDate'] = df_with_predictions['gameDate'].dt.tz_localize(None)
    
    # Handle date column
    date_col = None
    if 'gameDate' in df_with_predictions.columns:
        date_col = 'gameDate'
    elif 'gameDateTimeEst' in df_with_predictions.columns:
        # Convert to datetime if needed
        df_with_predictions['gameDate'] = pd.to_datetime(
            df_with_predictions['gameDateTimeEst'], format='mixed', errors='coerce', utc=True
        )
        if pd.api.types.is_datetime64_any_dtype(df_with_predictions['gameDate']):
            if df_with_predictions['gameDate'].dt.tz is not None:
                df_with_predictions['gameDate'] = df_with_predictions['gameDate'].dt.tz_localize(None)
        date_col = 'gameDate'
    
    # Filter to recent games if requested
    if recent_only and date_col and pd.api.types.is_datetime64_any_dtype(df_with_predictions[date_col]):
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Get the most recent date in the dataset
        max_date = df_with_predictions[date_col].max()
        if pd.notna(max_date):
            # Calculate cutoff date (N months back from most recent date)
            cutoff_date = max_date - pd.DateOffset(months=months_back)
            
            # Filter to recent games
            before_count = len(df_with_predictions)
            df_with_predictions = df_with_predictions[
                df_with_predictions[date_col] >= cutoff_date
            ].copy()
            after_count = len(df_with_predictions)
            
            print(f"\nFiltered to recent games (last {months_back} months)")
            print(f"  Date range: {cutoff_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            print(f"  Records: {before_count:,} -> {after_count:,}")
            
            if len(df_with_predictions) == 0:
                print(f"\n⚠ No recent games found in the last {months_back} months")
                print(f"   Most recent date in dataset: {max_date.strftime('%Y-%m-%d')}")
                if not use_latest_data:
                    print(f"\n💡 Tip: Try running without --use-existing to generate predictions for latest data")
                    print(f"   Or use --all-dates to see all historical games")
                return None
            
            # Warn if data seems outdated (older than 6 months from today)
            from datetime import datetime
            today = pd.Timestamp.now()
            if max_date < (today - pd.DateOffset(months=6)):
                print(f"\n⚠ Warning: Most recent data is from {max_date.strftime('%Y-%m-%d')}")
                print(f"   This may be outdated. Consider generating new predictions.")
    
    # Sort by predicted fantasy score (descending)
    df_with_predictions = df_with_predictions.sort_values(
        'predicted_fantasy_score', ascending=False
    )
    
    # Get most recent prediction per player (if we have dates)
    if date_col and pd.api.types.is_datetime64_any_dtype(df_with_predictions[date_col]):
        # Get most recent prediction per player
        df_with_predictions = df_with_predictions.sort_values(['personId', date_col], ascending=[True, False])
        df_with_predictions = df_with_predictions.groupby('personId').first().reset_index()
        
        # Re-sort by predicted score
        df_with_predictions = df_with_predictions.sort_values(
            'predicted_fantasy_score', ascending=False
        )
    
    # Select relevant columns for output
    output_cols = ['firstName', 'lastName', 'playerteamName']
    
    # Add date column (prefer gameDate, fallback to gameDateTimeEst)
    if 'gameDate' in df_with_predictions.columns:
        output_cols.append('gameDate')
    elif 'gameDateTimeEst' in df_with_predictions.columns:
        output_cols.append('gameDateTimeEst')
    
    if 'fantasyScore' in df_with_predictions.columns:
        output_cols.append('fantasyScore')
    output_cols.append('predicted_fantasy_score')
    
    # Filter to available columns and remove duplicates
    available_cols = [col for col in output_cols if col in df_with_predictions.columns]
    result_df = df_with_predictions[available_cols].copy()
    
    # Add player name column
    if 'firstName' in result_df.columns and 'lastName' in result_df.columns:
        result_df['Player'] = result_df['firstName'] + ' ' + result_df['lastName']
        result_df = result_df.drop(['firstName', 'lastName'], axis=1)
        # Reorder columns
        cols = ['Player'] + [c for c in result_df.columns if c != 'Player']
        result_df = result_df[cols]
    
    # Rename columns for better display
    rename_dict = {
        'playerteamName': 'Team',
        'predicted_fantasy_score': 'Predicted Fantasy Score',
        'fantasyScore': 'Actual Fantasy Score',
    }
    
    # Handle date column (only one should exist)
    if 'gameDate' in result_df.columns:
        rename_dict['gameDate'] = 'Game Date'
    elif 'gameDateTimeEst' in result_df.columns:
        rename_dict['gameDateTimeEst'] = 'Game Date'
    
    result_df = result_df.rename(columns=rename_dict)
    
    # Limit to top N if specified
    if top_n:
        result_df = result_df.head(top_n)
    
    return result_df


def display_results(result_df, team_names):
    """Display prediction results in a formatted way."""
    if result_df is None or len(result_df) == 0:
        return
    
    print(f"\n{'='*80}")
    print(f"PREDICTIONS FOR: {', '.join(team_names)}")
    print(f"{'='*80}\n")
    
    # Display summary
    print(f"Total Players: {len(result_df)}")
    if 'Predicted Fantasy Score' in result_df.columns:
        print(f"Average Predicted Score: {result_df['Predicted Fantasy Score'].mean():.2f}")
        print(f"Max Predicted Score: {result_df['Predicted Fantasy Score'].max():.2f}")
        print(f"Min Predicted Score: {result_df['Predicted Fantasy Score'].min():.2f}")
    
    print(f"\n{'='*80}")
    print("TOP PLAYERS BY PREDICTED FANTASY SCORE")
    print(f"{'='*80}\n")
    
    # Display table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(result_df.to_string(index=False))
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Predict fantasy scores for players on specific teams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/modeling/predict_by_teams.py --teams Lakers Warriors
  python scripts/modeling/predict_by_teams.py --teams "Los Angeles Lakers" "Golden State Warriors" --top 20
  python scripts/modeling/predict_by_teams.py --teams Celtics --use-existing
  python scripts/modeling/predict_by_teams.py --list-teams
        """
    )
    
    parser.add_argument(
        '--teams', '-t',
        nargs='+',
        help='Team names to get predictions for (case-insensitive, partial matching)'
    )
    
    parser.add_argument(
        '--top', '-n',
        type=int,
        help='Show only top N players by predicted score'
    )
    
    parser.add_argument(
        '--use-existing',
        action='store_true',
        help='Use existing predictions file instead of generating new ones'
    )
    
    parser.add_argument(
        '--list-teams',
        action='store_true',
        help='List all available team names'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        help='Save results to CSV file (provide filename)'
    )
    
    parser.add_argument(
        '--all-dates',
        action='store_true',
        help='Include all historical games (default: only recent games from last 3 months)'
    )
    
    parser.add_argument(
        '--months',
        type=int,
        default=3,
        help='Number of months back to include (default: 3)'
    )
    
    parser.add_argument(
        '--next-game',
        action='store_true',
        help='Predict next game for each player (forward-looking predictions)'
    )
    
    args = parser.parse_args()
    
    if args.list_teams:
        print("\nAvailable Teams:")
        print("="*80)
        teams = get_available_teams()
        for i, team in enumerate(teams, 1):
            print(f"{i:3d}. {team}")
        print(f"\nTotal: {len(teams)} teams")
        return
    
    if not args.teams:
        parser.print_help()
        print("\n⚠ Error: Please specify team names using --teams")
        print("   Example: python scripts/modeling/predict_by_teams.py --teams Lakers Warriors")
        return
    
    # Use next game predictions if requested
    if args.next_game:
        from scripts.modeling.predict_upcoming_games import predict_next_game_for_teams
        result_df = predict_next_game_for_teams(args.teams, args.top)
    else:
        # Get predictions
        result_df = predict_for_teams(
            args.teams,
            use_latest_data=not args.use_existing,
            top_n=args.top,
            recent_only=not args.all_dates,
            months_back=args.months
        )
    
    if result_df is None or len(result_df) == 0:
        return
    
    # Display results
    display_results(result_df, args.teams)
    
    # Save if requested
    if args.save:
        output_file = project_root / "models" / "predictions" / args.save
        result_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()

