"""
Optimize lineups from player_data.csv file.

This script:
1. Loads players from CSV with salaries
2. Generates predictions for each player
3. Creates optimal lineups for:
   - FanDuel format: 8 players, $50,000 cap
   - DraftKings format: 9 players (2PG, 2SG, 2SF, 2PF, 1C), $60,000 cap
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
import re

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.load_data import load_player_statistics
from scripts.modeling.predict import predict_fantasy_scores


def parse_salary(salary_str):
    """Parse salary string like '$10,000' to integer."""
    if pd.isna(salary_str) or salary_str == '':
        return 0
    # Remove $ and commas
    salary_str = str(salary_str).replace('$', '').replace(',', '').strip()
    try:
        return int(salary_str)
    except:
        return 0


def parse_position(position_str):
    """Parse position string like 'PG/SG' to list of positions."""
    if pd.isna(position_str) or position_str == '':
        return []
    positions = str(position_str).split('/')
    return [p.strip() for p in positions]


def match_player_to_dataset(player_name, team, df):
    """
    Match a player from CSV to the dataset.
    
    Parameters:
    -----------
    player_name : str
        Player name from CSV
    team : str
        Team abbreviation
    df : pd.DataFrame
        Full dataset
    
    Returns:
    --------
    pd.DataFrame
        Matching player records
    """
    # Normalize player name
    name_parts = player_name.split()
    first_name = name_parts[0] if len(name_parts) > 0 else ''
    last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
    
    # Try exact match first
    if 'firstName' in df.columns and 'lastName' in df.columns:
        # Exact match
        exact_match = df[
            (df['firstName'].str.contains(first_name, case=False, na=False)) &
            (df['lastName'].str.contains(last_name, case=False, na=False))
        ]
        
        if len(exact_match) > 0:
            # Filter by team if possible
            if 'playerteamName' in df.columns:
                team_match = exact_match[
                    exact_match['playerteamName'].str.contains(team, case=False, na=False)
                ]
                if len(team_match) > 0:
                    return team_match
            
            return exact_match
    
    return pd.DataFrame()


def load_and_prepare_players(csv_path):
    """
    Load players from CSV and prepare for prediction.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to player_data.csv
    
    Returns:
    --------
    pd.DataFrame
        Prepared player data with salaries
    """
    print("Loading player data from CSV...")
    df = pd.read_csv(csv_path)
    
    # Parse salaries
    df['salary'] = df['Salary'].apply(parse_salary)
    
    # Parse positions
    df['positions'] = df['Position'].apply(parse_position)
    
    # Clean up player names
    df['player_name'] = df['Player Name'].str.strip()
    
    # Parse team and opponent
    df['team'] = df['Team'].str.strip()
    df['opponent'] = df['Opponent'].str.strip()
    
    # Filter out players with no salary or invalid data
    df = df[df['salary'] > 0].copy()
    
    # Filter out players with status indicating they won't play (O = Out, etc.)
    # Keep Q (Questionable) players as they might play
    if 'Status' in df.columns:
        df = df[~df['Status'].str.contains('O', case=False, na=False)].copy()
    
    print(f"Loaded {len(df)} players from CSV")
    return df


def get_player_predictions(csv_players_df):
    """
    Get predictions for players in CSV.
    
    Parameters:
    -----------
    csv_players_df : pd.DataFrame
        Players from CSV
    
    Returns:
    --------
    pd.DataFrame
        Players with predictions
    """
    print("\nLoading dataset to match players...")
    full_df = load_player_statistics(save_raw=False)
    
    # Convert date
    if 'gameDateTimeEst' in full_df.columns:
        full_df['gameDate'] = pd.to_datetime(full_df['gameDateTimeEst'], format='mixed', errors='coerce', utc=True)
        if pd.api.types.is_datetime64_any_dtype(full_df['gameDate']):
            if full_df['gameDate'].dt.tz is not None:
                full_df['gameDate'] = full_df['gameDate'].dt.tz_localize(None)
    
    # Get most recent game for each player
    full_df = full_df.sort_values(['personId', 'gameDate'], ascending=[True, False])
    recent_players = full_df.groupby('personId').first().reset_index()
    
    print(f"Dataset loaded: {len(recent_players):,} unique players")
    
    # Match players
    print("\nMatching players and generating predictions...")
    matched_players = []
    # Store CSV data by personId (more stable than name)
    csv_data_by_personid = {}
    
    for idx, row in csv_players_df.iterrows():
        player_name = row['player_name']
        team = row['team']
        
        # Match to dataset
        matches = match_player_to_dataset(player_name, team, recent_players)
        
        if len(matches) > 0:
            # Use most recent match
            player_data = matches.iloc[0:1].copy()
            person_id = player_data['personId'].iloc[0]
            
            # Store all CSV data by personId
            csv_data_by_personid[person_id] = {
                'salary': row['salary'],
                'positions': row['positions'],
                'team': str(row['team']),
                'opponent': str(row['opponent']),
                'name': player_name,
                'fppg': row.get('FPPG', 0),
                'matchup': str(row.get('Matchup', '')) if 'Matchup' in row else ''
            }
            
            matched_players.append(player_data)
        else:
            print(f"  ⚠ Could not match: {player_name} ({team})")
    
    if len(matched_players) == 0:
        print("\n⚠ No players matched! Check player names and teams.")
        return pd.DataFrame()
    
    matched_df = pd.concat(matched_players, ignore_index=True)
    print(f"\nMatched {len(matched_df)} players")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions_df = predict_fantasy_scores(matched_df, use_ensemble=True)
    
    # Add CSV data back using personId mapping (personId should be preserved)
    if 'personId' in predictions_df.columns:
        predictions_df['salary'] = predictions_df['personId'].map(lambda x: csv_data_by_personid.get(x, {}).get('salary', 0))
        predictions_df['positions'] = predictions_df['personId'].map(lambda x: csv_data_by_personid.get(x, {}).get('positions', []))
        predictions_df['csv_team'] = predictions_df['personId'].map(lambda x: csv_data_by_personid.get(x, {}).get('team', ''))
        predictions_df['csv_opponent'] = predictions_df['personId'].map(lambda x: csv_data_by_personid.get(x, {}).get('opponent', ''))
        predictions_df['csv_name'] = predictions_df['personId'].map(lambda x: csv_data_by_personid.get(x, {}).get('name', ''))
    else:
        # Fallback: try to use csv_name if personId is lost
        print("⚠ Warning: personId not found, using fallback method")
        predictions_df['salary'] = predictions_df.get('csv_salary', 0)
        predictions_df['positions'] = []
    
    # Ensure positions are lists
    predictions_df['positions'] = predictions_df['positions'].apply(
        lambda x: x if isinstance(x, list) and len(x) > 0 else []
    )
    
    # Debug: check positions
    valid_pos_count = predictions_df['positions'].apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
    print(f"Players with valid positions: {valid_pos_count}/{len(predictions_df)}")
    predictions_df['csv_team'] = predictions_df['csv_team']
    predictions_df['csv_opponent'] = predictions_df['csv_opponent']
    predictions_df['csv_name'] = predictions_df['csv_name']
    
    # Calculate value (points per $1000)
    predictions_df['value'] = predictions_df['predicted_fantasy_score'] / (predictions_df['salary'] / 1000)
    
    return predictions_df


def optimize_fanduel_lineup(players_df, salary_cap=50000, num_lineups=2):
    """
    Optimize FanDuel lineups.
    
    FanDuel requirements:
    - 1 PG, 1 SG, 1 SF, 1 PF, 1 C
    - 1 G (PG or SG)
    - 1 F (SF or PF)
    - 1 UTIL (any position)
    - Total: 8 players
    - Salary cap: $50,000
    
    Parameters:
    -----------
    players_df : pd.DataFrame
        Players with predictions and salaries
    salary_cap : int
        Salary cap (default: 50000)
    num_lineups : int
        Number of lineups to generate
    
    Returns:
    --------
    list
        List of optimal lineups (DataFrames)
    """
    print(f"\n{'='*80}")
    print(f"OPTIMIZING FANDUEL LINEUPS (${salary_cap:,} cap)")
    print(f"{'='*80}")
    
    # Filter valid players
    def has_valid_positions(pos_list):
        if isinstance(pos_list, list):
            return len(pos_list) > 0
        return False
    
    valid = players_df[
        (players_df['salary'] > 0) &
        (players_df['predicted_fantasy_score'].notna()) &
        (players_df['positions'].apply(has_valid_positions))
    ].copy()
    
    print(f"Valid players for FanDuel: {len(valid)}")
    
    if len(valid) < 8:
        print(f"⚠ Not enough players ({len(valid)} < 8)")
        return []
    
    # Sort by value
    valid = valid.sort_values('value', ascending=False).reset_index(drop=True)
    
    lineups = []
    
    for lineup_num in range(num_lineups):
        print(f"\nGenerating lineup {lineup_num + 1}...")
        
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
            'UTIL': 1
        }
        
        # Create a copy for this lineup
        available = valid.copy()
        
        # Remove players already used in previous lineups
        for prev_lineup in lineups:
            used_ids = prev_lineup['personId'].values if 'personId' in prev_lineup.columns else []
            available = available[~available['personId'].isin(used_ids)]
        
        for _, player in available.iterrows():
            if len(lineup) >= 8:
                break
            
            if total_salary + player['salary'] > salary_cap:
                continue
            
            pos_list = player['positions']
            added = False
            
            # Try to fill required positions first
            for pos in pos_list:
                if pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                    if positions_needed.get(pos, 0) > 0:
                        lineup.append(player)
                        positions_needed[pos] -= 1
                        total_salary += player['salary']
                        total_points += player['predicted_fantasy_score']
                        added = True
                        break
                    elif pos in ['PG', 'SG'] and positions_needed.get('G', 0) > 0:
                        lineup.append(player)
                        positions_needed['G'] -= 1
                        total_salary += player['salary']
                        total_points += player['predicted_fantasy_score']
                        added = True
                        break
                    elif pos in ['SF', 'PF'] and positions_needed.get('F', 0) > 0:
                        lineup.append(player)
                        positions_needed['F'] -= 1
                        total_salary += player['salary']
                        total_points += player['predicted_fantasy_score']
                        added = True
                        break
            
            # If still need UTIL
            if not added and positions_needed.get('UTIL', 0) > 0:
                lineup.append(player)
                positions_needed['UTIL'] -= 1
                total_salary += player['salary']
                total_points += player['predicted_fantasy_score']
                added = True
        
        if len(lineup) < 8:
            print(f"  ⚠ Could not complete lineup ({len(lineup)}/8 players)")
            continue
        
        lineup_df = pd.DataFrame(lineup)
        lineups.append(lineup_df)
        
        print(f"  ✓ Lineup {lineup_num + 1} created:")
        print(f"    Salary: ${total_salary:,} / ${salary_cap:,}")
        print(f"    Predicted Points: {total_points:.2f}")
        print(f"    Remaining: ${salary_cap - total_salary:,}")
    
    return lineups


def optimize_draftkings_lineup(players_df, salary_cap=60000, num_lineups=2):
    """
    Optimize DraftKings lineups with improved algorithm.
    
    DraftKings requirements:
    - 2 PG
    - 2 SG
    - 2 SF
    - 2 PF
    - 1 C
    - Total: 9 players
    - Salary cap: $60,000
    
    Parameters:
    -----------
    players_df : pd.DataFrame
        Players with predictions and salaries
    salary_cap : int
        Salary cap (default: 60000)
    num_lineups : int
        Number of lineups to generate
    
    Returns:
    --------
    list
        List of optimal lineups (DataFrames)
    """
    print(f"\n{'='*80}")
    print(f"OPTIMIZING DRAFTKINGS LINEUPS (${salary_cap:,} cap)")
    print(f"{'='*80}")
    
    # Filter valid players
    def has_valid_positions(pos_list):
        if isinstance(pos_list, list):
            return len(pos_list) > 0
        return False
    
    valid = players_df[
        (players_df['salary'] > 0) &
        (players_df['predicted_fantasy_score'].notna()) &
        (players_df['positions'].apply(has_valid_positions))
    ].copy()
    
    print(f"Valid players for DraftKings: {len(valid)}")
    
    if len(valid) < 9:
        print(f"⚠ Not enough players ({len(valid)} < 9)")
        return []
    
    # Sort by value (points per $1000)
    valid = valid.sort_values('value', ascending=False).reset_index(drop=True)
    
    lineups = []
    
    for lineup_num in range(num_lineups):
        print(f"\nGenerating lineup {lineup_num + 1}...")
        
        lineup = []
        total_salary = 0
        total_points = 0
        
        positions_needed = {
            'PG': 2,
            'SG': 2,
            'SF': 2,
            'PF': 2,
            'C': 1
        }
        
        # Create a copy for this lineup
        available = valid.copy()
        
        # Remove players already used in previous lineups
        for prev_lineup in lineups:
            used_ids = prev_lineup['personId'].values if 'personId' in prev_lineup.columns else []
            available = available[~available['personId'].isin(used_ids)]
        
        # First pass: Fill required positions with best value players
        # Sort by value, but prioritize filling all positions
        for _, player in available.iterrows():
            if len(lineup) >= 9:
                break
            
            if total_salary + player['salary'] > salary_cap:
                continue
            
            pos_list = player['positions']
            added = False
            
            # Try to fill required positions - check all positions the player can play
            for pos in pos_list:
                if pos in positions_needed and positions_needed[pos] > 0:
                    lineup.append(player)
                    positions_needed[pos] -= 1
                    total_salary += player['salary']
                    total_points += player['predicted_fantasy_score']
                    added = True
                    break
            
            if not added:
                continue
        
        # If we still need positions, be more aggressive
        if len(lineup) < 9:
            remaining_needed = sum(positions_needed.values())
            if remaining_needed > 0:
                # Get all remaining available players sorted by value
                used_ids = [p['personId'] for p in lineup if 'personId' in p] if lineup else []
                available_remaining = available[
                    (~available['personId'].isin(used_ids)) &
                    (available['salary'] <= (salary_cap - total_salary))
                ].copy()
                
                # Sort by value
                available_remaining = available_remaining.sort_values('value', ascending=False)
                
                # Try to fill remaining positions
                for _, player in available_remaining.iterrows():
                    if len(lineup) >= 9:
                        break
                    
                    if total_salary + player['salary'] > salary_cap:
                        continue
                    
                    pos_list = player['positions']
                    added = False
                    
                    # Check if player can fill any remaining position
                    for pos in pos_list:
                        if pos in positions_needed and positions_needed[pos] > 0:
                            lineup.append(player)
                            positions_needed[pos] -= 1
                            total_salary += player['salary']
                            total_points += player['predicted_fantasy_score']
                            added = True
                            break
                    
                    if not added:
                        continue
        
        # If we still need positions, try to fill with any available position
        if len(lineup) < 9:
            remaining_needed = sum(positions_needed.values())
            if remaining_needed > 0:
                # Sort by remaining salary space and value
                remaining_salary = salary_cap - total_salary
                available_remaining = available[
                    (~available['personId'].isin([p['personId'] for p in lineup if 'personId' in p])) &
                    (available['salary'] <= remaining_salary)
                ].copy()
                
                # Try to fill remaining positions more flexibly
                for _, player in available_remaining.iterrows():
                    if len(lineup) >= 9:
                        break
                    
                    if total_salary + player['salary'] > salary_cap:
                        continue
                    
                    pos_list = player['positions']
                    added = False
                    
                    # Check if player can fill any remaining position
                    for pos in pos_list:
                        if pos in positions_needed and positions_needed[pos] > 0:
                            lineup.append(player)
                            positions_needed[pos] -= 1
                            total_salary += player['salary']
                            total_points += player['predicted_fantasy_score']
                            added = True
                            break
                    
                    if not added:
                        continue
        
        if len(lineup) < 9:
            print(f"  ⚠ Could not complete lineup ({len(lineup)}/9 players)")
            print(f"    Still need: {dict((k, v) for k, v in positions_needed.items() if v > 0)}")
            continue
        
        lineup_df = pd.DataFrame(lineup)
        lineups.append(lineup_df)
        
        print(f"  ✓ Lineup {lineup_num + 1} created:")
        print(f"    Salary: ${total_salary:,} / ${salary_cap:,}")
        print(f"    Predicted Points: {total_points:.2f}")
        print(f"    Remaining: ${salary_cap - total_salary:,}")
    
    return lineups


def print_lineup(lineup_df, format_name, lineup_num):
    """Print a formatted lineup."""
    print(f"\n{'='*80}")
    print(f"{format_name} LINEUP #{lineup_num}")
    print(f"{'='*80}\n")
    
    # Sort by position
    if format_name == "FANDUEL":
        position_order = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    else:
        position_order = ['PG', 'SG', 'SF', 'PF', 'C']
    
    display_data = []
    for _, player in lineup_df.iterrows():
        pos_list = player['positions'] if isinstance(player['positions'], list) else eval(str(player['positions']))
        pos_str = '/'.join(pos_list[:2]) if len(pos_list) > 0 else 'N/A'
        
        display_data.append({
            'Position': pos_str,
            'Player': player.get('csv_name', player.get('firstName', '') + ' ' + player.get('lastName', '')),
            'Team': player.get('csv_team', ''),
            'Opponent': player.get('csv_opponent', ''),
            'Salary': f"${player['salary']:,}",
            'Predicted': f"{player['predicted_fantasy_score']:.2f}",
            'Value': f"{player['value']:.2f}"
        })
    
    display_df = pd.DataFrame(display_data)
    print(display_df.to_string(index=False))
    
    total_salary = lineup_df['salary'].sum()
    total_points = lineup_df['predicted_fantasy_score'].sum()
    
    print(f"\nTotal Salary: ${total_salary:,}")
    print(f"Total Predicted Points: {total_points:.2f}")
    print(f"Average Value: {lineup_df['value'].mean():.2f} pts/$1k")


def optimize_by_game(players_df):
    """Optimize lineups separately for each game."""
    print(f"\n{'='*80}")
    print("OPTIMIZING LINEUPS BY GAME")
    print(f"{'='*80}")
    
    # Group by matchup - use the original CSV matchup if available
    # Otherwise create from team/opponent
    if 'Matchup' not in players_df.columns and 'csv_opponent' in players_df.columns:
        # Ensure we have string values
        players_df['csv_team'] = players_df['csv_team'].fillna('').astype(str)
        players_df['csv_opponent'] = players_df['csv_opponent'].fillna('').astype(str)
        players_df['Matchup'] = players_df['csv_team'] + ' @ ' + players_df['csv_opponent']
    
    if 'Matchup' not in players_df.columns:
        print("⚠ Cannot determine games. Using all players together.")
        return {'All Games': optimize_fanduel_lineup(players_df, salary_cap=50000, num_lineups=2)}
    
    games = players_df.groupby('Matchup')
    all_lineups = {}
    
    for game_name, game_players in games:
        print(f"\n\n{'='*80}")
        print(f"GAME: {game_name}")
        print(f"{'='*80}")
        print(f"Players in game: {len(game_players)}")
        
        game_lineups = optimize_fanduel_lineup(game_players, salary_cap=50000, num_lineups=2)
        all_lineups[game_name] = game_lineups
        
        for i, lineup in enumerate(game_lineups, 1):
            print_lineup(lineup, "FANDUEL", i)
    
    return all_lineups


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize lineups from CSV')
    parser.add_argument('--csv', type=str, default='player_data.csv', help='Path to player_data.csv')
    parser.add_argument('--output', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    csv_path = project_root / args.csv
    if not csv_path.exists():
        csv_path = Path(args.csv)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Load and prepare players
    csv_players = load_and_prepare_players(csv_path)
    
    if len(csv_players) == 0:
        print("No valid players found in CSV")
        sys.exit(1)
    
    # Get predictions
    players_with_predictions = get_player_predictions(csv_players)
    
    if len(players_with_predictions) == 0:
        print("No predictions generated")
        sys.exit(1)
    
    # Optimize full roster (DraftKings format) - ALL PLAYERS
    print("\n" + "="*80)
    print("GENERATING FULL ROSTER LINEUPS (DraftKings Format)")
    print("Using ALL available players from CSV")
    print("="*80)
    # Use improved algorithm
    from scripts.modeling.optimize_draftkings_simple import optimize_draftkings_lineup_simple
    dk_lineups = optimize_draftkings_lineup_simple(players_with_predictions, salary_cap=60000, num_lineups=2)
    
    if len(dk_lineups) > 0:
        for i, lineup in enumerate(dk_lineups, 1):
            print_lineup(lineup, "DRAFTKINGS", i)
    else:
        print("\n⚠ Could not generate complete lineups. Try adjusting the algorithm or check player availability.")
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "models" / "lineups"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DraftKings lineups
    for i, lineup in enumerate(dk_lineups, 1):
        filename = output_dir / f"draftkings_lineup_{i}.csv"
        lineup[['csv_name', 'csv_team', 'csv_opponent', 'salary', 'predicted_fantasy_score', 'value']].to_csv(filename, index=False)
        print(f"\n✓ Saved: {filename}")

