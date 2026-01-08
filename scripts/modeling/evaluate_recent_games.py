"""
Evaluate model performance on recent games.

This script:
1. Loads the latest dataset
2. Identifies the most recent games
3. Generates predictions for those games
4. Compares predictions vs actual results
5. Provides game-by-game analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.load_data import load_player_statistics
from scripts.modeling.predict import predict_fantasy_scores
from scripts.modeling.evaluate import calculate_metrics, print_metrics


def get_recent_games(df, days_back=7):
    """
    Get the most recent games from the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    days_back : int
        Number of days back to include
    
    Returns:
    --------
    pd.DataFrame
        Recent games
    """
    # Ensure date column exists
    if 'gameDate' not in df.columns and 'gameDateTimeEst' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDateTimeEst'], format='mixed', errors='coerce', utc=True)
        if pd.api.types.is_datetime64_any_dtype(df['gameDate']):
            if df['gameDate'].dt.tz is not None:
                df['gameDate'] = df['gameDate'].dt.tz_localize(None)
    
    # Get most recent date
    max_date = df['gameDate'].max()
    cutoff_date = max_date - pd.Timedelta(days=days_back)
    
    recent_games = df[df['gameDate'] >= cutoff_date].copy()
    
    return recent_games, max_date, cutoff_date


def evaluate_recent_games(days_back=7, min_games=3):
    """
    Evaluate model on recent games.
    
    Parameters:
    -----------
    days_back : int
        Number of days back to include
    min_games : int
        Minimum number of games to analyze
    
    Returns:
    --------
    dict
        Evaluation results
    """
    print("="*80)
    print("MODEL EVALUATION ON RECENT GAMES")
    print("="*80)
    
    # Load latest data
    print("\n1. Loading latest dataset...")
    df = load_player_statistics(save_raw=False)
    
    print(f"   Total records: {len(df):,}")
    
    # Get recent games
    print(f"\n2. Identifying recent games (last {days_back} days)...")
    recent_games, max_date, cutoff_date = get_recent_games(df, days_back)
    
    print(f"   Date range: {cutoff_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"   Recent games: {len(recent_games):,} player-game records")
    
    if len(recent_games) == 0:
        print("\n⚠ No recent games found!")
        return None
    
    # Get unique games
    if 'gameId' in recent_games.columns:
        unique_games = recent_games['gameId'].unique()
        print(f"   Unique games: {len(unique_games)}")
    else:
        # Group by date and teams
        unique_games = recent_games.groupby(['gameDate', 'playerteamName', 'opponentteamName']).size().reset_index()
        print(f"   Unique game matchups: {len(unique_games)}")
    
    if len(unique_games) < min_games:
        print(f"\n⚠ Only {len(unique_games)} games found. Consider increasing days_back.")
        print("   Continuing with available games...")
    
    # Generate predictions
    print(f"\n3. Generating predictions for recent games...")
    print("   (This may take a few minutes...)")
    
    df_with_predictions = predict_fantasy_scores(recent_games, use_ensemble=True)
    
    # Calculate overall metrics
    print(f"\n4. Calculating overall performance metrics...")
    y_true = df_with_predictions['fantasyScore'].values
    y_pred = df_with_predictions['predicted_fantasy_score'].values
    
    overall_metrics = calculate_metrics(y_true, y_pred)
    print_metrics(y_true, y_pred, "Overall Recent Games")
    
    # Game-by-game analysis
    print(f"\n5. Game-by-Game Analysis")
    print("="*80)
    
    game_analyses = []
    
    if 'gameId' in df_with_predictions.columns:
        # Group by gameId
        for game_id in df_with_predictions['gameId'].unique()[:20]:  # Limit to 20 games
            game_df = df_with_predictions[df_with_predictions['gameId'] == game_id].copy()
            
            if len(game_df) < 2:  # Need at least 2 players
                continue
            
            # Get game info
            game_info = game_df.iloc[0]
            game_date = game_info.get('gameDate', game_info.get('gameDateTimeEst', 'Unknown'))
            team1 = game_info.get('playerteamName', 'Unknown')
            team2 = game_info.get('opponentteamName', 'Unknown')
            
            # Calculate game metrics
            game_true = game_df['fantasyScore'].values
            game_pred = game_df['predicted_fantasy_score'].values
            game_metrics = calculate_metrics(game_true, game_pred)
            
            game_analyses.append({
                'game_id': game_id,
                'date': game_date,
                'team1': team1,
                'team2': team2,
                'players': len(game_df),
                'metrics': game_metrics,
                'data': game_df
            })
    else:
        # Group by date and teams
        grouped = df_with_predictions.groupby(['gameDate', 'playerteamName', 'opponentteamName'])
        for (game_date, team1, team2), game_df in list(grouped)[:20]:  # Limit to 20 games
            if len(game_df) < 2:
                continue
            
            game_true = game_df['fantasyScore'].values
            game_pred = game_df['predicted_fantasy_score'].values
            game_metrics = calculate_metrics(game_true, game_pred)
            
            game_analyses.append({
                'game_id': f"{team1} vs {team2}",
                'date': game_date,
                'team1': team1,
                'team2': team2,
                'players': len(game_df),
                'metrics': game_metrics,
                'data': game_df
            })
    
    # Sort by date (most recent first)
    game_analyses.sort(key=lambda x: x['date'] if isinstance(x['date'], pd.Timestamp) else pd.Timestamp.min, reverse=True)
    
    # Display game-by-game results
    for i, game in enumerate(game_analyses, 1):
        print(f"\n{'='*80}")
        print(f"GAME {i}: {game['team1']} vs {game['team2']}")
        print(f"Date: {game['date']}")
        print(f"Players: {game['players']}")
        print(f"{'='*80}")
        
        # Game-level metrics
        print(f"\nGame-Level Metrics:")
        print(f"  MAE:  {game['metrics']['MAE']:.2f}")
        print(f"  RMSE: {game['metrics']['RMSE']:.2f}")
        print(f"  R²:   {game['metrics']['R2']:.4f}")
        
        # Player-level details
        game_df = game['data'].copy()
        game_df = game_df.sort_values('predicted_fantasy_score', ascending=False)
        
        # Format player data
        if 'firstName' in game_df.columns and 'lastName' in game_df.columns:
            game_df['Player'] = game_df['firstName'] + ' ' + game_df['lastName']
        
        display_cols = ['Player']
        if 'playerteamName' in game_df.columns:
            display_cols.append('playerteamName')
        display_cols.extend(['fantasyScore', 'predicted_fantasy_score'])
        
        available_cols = [c for c in display_cols if c in game_df.columns]
        player_df = game_df[available_cols].copy()
        
        player_df = player_df.rename(columns={
            'playerteamName': 'Team',
            'fantasyScore': 'Actual',
            'predicted_fantasy_score': 'Predicted'
        })
        
        # Calculate error
        player_df['Error'] = player_df['Actual'] - player_df['Predicted']
        player_df['Abs Error'] = player_df['Error'].abs()
        
        print(f"\nPlayer-by-Player Performance:")
        print(player_df.to_string(index=False))
        
        # Top errors
        print(f"\nLargest Errors:")
        top_errors = player_df.nlargest(3, 'Abs Error')[['Player', 'Team', 'Actual', 'Predicted', 'Error']]
        print(top_errors.to_string(index=False))
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    game_maes = [g['metrics']['MAE'] for g in game_analyses]
    game_rmses = [g['metrics']['RMSE'] for g in game_analyses]
    game_r2s = [g['metrics']['R2'] for g in game_analyses]
    
    print(f"Games Analyzed: {len(game_analyses)}")
    print(f"\nAverage Game-Level Metrics:")
    print(f"  MAE:  {np.mean(game_maes):.2f} (std: {np.std(game_maes):.2f})")
    print(f"  RMSE: {np.mean(game_rmses):.2f} (std: {np.std(game_rmses):.2f})")
    print(f"  R²:   {np.mean(game_r2s):.4f} (std: {np.std(game_r2s):.4f})")
    
    # Save detailed results
    results_path = project_root / "documentation" / "results" / "recent_games_evaluation.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary dataframe
    summary_data = []
    for game in game_analyses:
        summary_data.append({
            'Game': f"{game['team1']} vs {game['team2']}",
            'Date': game['date'],
            'Players': game['players'],
            'MAE': game['metrics']['MAE'],
            'RMSE': game['metrics']['RMSE'],
            'R2': game['metrics']['R2']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_path, index=False)
    print(f"\n✓ Summary saved to: {results_path}")
    
    # Save detailed player-level results
    detailed_path = project_root / "documentation" / "results" / "recent_games_detailed.csv"
    all_players = []
    for game in game_analyses:
        game_df = game['data'].copy()
        if 'firstName' in game_df.columns and 'lastName' in game_df.columns:
            game_df['Player'] = game_df['firstName'] + ' ' + game_df['lastName']
        game_df['Game'] = f"{game['team1']} vs {game['team2']}"
        game_df['Game_Date'] = game['date']
        all_players.append(game_df)
    
    if all_players:
        detailed_df = pd.concat(all_players, ignore_index=True)
        detailed_df.to_csv(detailed_path, index=False)
        print(f"✓ Detailed results saved to: {detailed_path}")
    
    return {
        'overall_metrics': overall_metrics,
        'game_analyses': game_analyses,
        'summary': summary_df
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model on recent games')
    parser.add_argument('--days', type=int, default=7, help='Days back to include (default: 7)')
    parser.add_argument('--min-games', type=int, default=3, help='Minimum games to analyze')
    
    args = parser.parse_args()
    
    results = evaluate_recent_games(days_back=args.days, min_games=args.min_games)


