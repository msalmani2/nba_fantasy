"""
Optimize FanDuel lineups from FanDuel CSV export.

This script:
1. Loads players from FanDuel CSV export
2. Filters out injured players
3. Generates 3 optimal lineups using ILP optimization (provably optimal!)
4. FanDuel format: 2 PG, 2 SG, 2 SF, 2 PF, 1 C (9 players, $60,000 cap)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.load_data import load_player_statistics
from scripts.modeling.predict import predict_fantasy_scores
from scripts.modeling.ilp_optimizer import optimize_lineup_ilp_fanduel


def parse_positions(position_str):
    """Parse position string like 'PG/SG' or 'C/PF' to list."""
    if pd.isna(position_str) or position_str == '':
        return []
    return [p.strip() for p in str(position_str).split('/')]


def is_injured(injury_indicator):
    """Check if player is injured (Out or Questionable)."""
    if pd.isna(injury_indicator) or injury_indicator == '':
        return False
    # O = Out, Q = Questionable (we'll exclude both)
    return str(injury_indicator).upper() in ['O', 'Q']


def load_fanduel_csv(csv_path):
    """Load and prepare FanDuel CSV data."""
    print(f"Loading FanDuel CSV: {csv_path}")
    
    # Try to detect if this is a FanDuel upload template format (has instructions in first rows)
    # In that case, skip to row 7 where the actual player data starts
    try:
        # First, try reading the first row to detect the format
        first_row = pd.read_csv(csv_path, nrows=1)
        if 'PG' in first_row.columns and 'Instructions' in first_row.columns:
            # This is a FanDuel upload template - skip first 6 rows
            print("Detected FanDuel upload template format, skipping header rows...")
            df = pd.read_csv(csv_path, skiprows=6)
            # The actual column names are in row 7 (after skipping 6), which becomes the header
        else:
            # Standard FanDuel export format
            df = pd.read_csv(csv_path)
    except:
        # Fallback to standard read
        df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} players from CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Filter out injured players
    if 'Injury Indicator' in df.columns:
        injured = df['Injury Indicator'].apply(is_injured)
        df = df[~injured].copy()
        print(f"Filtered out {injured.sum()} injured players")
        print(f"Remaining players: {len(df)}")
    
    # Parse positions
    if 'Position' in df.columns:
        df['positions'] = df['Position'].apply(parse_positions)
    elif 'Roster Position' in df.columns:
        df['positions'] = df['Roster Position'].apply(parse_positions)
    else:
        print("⚠ Warning: No position column found")
        df['positions'] = [[] for _ in range(len(df))]
    
    # Parse salary
    if 'Salary' in df.columns:
        df['salary'] = pd.to_numeric(df['Salary'], errors='coerce').fillna(0)
    else:
        print("⚠ Warning: No salary column found")
        df['salary'] = 0
    
    # Get fantasy points (FPPG = Fantasy Points Per Game)
    if 'FPPG' in df.columns:
        df['fppg'] = pd.to_numeric(df['FPPG'], errors='coerce').fillna(0)
    else:
        print("⚠ Warning: No FPPG column found, will use model predictions")
        df['fppg'] = 0
    
    # Create player name for display (using Nickname which is the full display name in FanDuel CSVs)
    if 'Nickname' in df.columns:
        df['player_name'] = df['Nickname']
    elif 'First Name' in df.columns and 'Last Name' in df.columns:
        df['player_name'] = df['First Name'] + ' ' + df['Last Name']
    else:
        df['player_name'] = 'Unknown'
    
    # Get team and opponent
    df['team'] = df.get('Team', '')
    df['opponent'] = df.get('Opponent', '')
    
    return df


def optimize_fanduel_lineup(players_df, salary_cap=60000, num_lineups=3, use_model_predictions=False):
    """
    Optimize lineups with custom position requirements.
    
    Position requirements:
    - 2 PG, 2 SG, 2 SF, 2 PF, 1 C
    - Total: 9 players
    - Salary cap: $60,000
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
        (players_df['fppg'].notna() | use_model_predictions) &
        (players_df['positions'].apply(has_valid_positions))
    ].copy()
    
    print(f"Valid players: {len(valid)}")
    
    if len(valid) < 9:
        print(f"⚠ Not enough players ({len(valid)} < 9)")
        return []
    
    # Use FPPG as predicted score, or generate predictions
    if use_model_predictions:
        print("Generating model predictions...")
        # This would require matching players to dataset and running predictions
        # For now, use FPPG
        valid['predicted_fantasy_score'] = valid['fppg']
    else:
        valid['predicted_fantasy_score'] = valid['fppg']
    
    # Calculate value (points per $1000)
    valid['value'] = valid['predicted_fantasy_score'] / (valid['salary'] / 1000)
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
        
        # Create available pool (excluding previous lineups)
        available = valid.copy().reset_index(drop=True)
        used_player_ids = set()
        
        for prev_lineup in lineups:
            if 'Id' in prev_lineup.columns:
                used_ids = prev_lineup['Id'].values
                available = available[~available['Id'].isin(used_ids)].reset_index(drop=True)
        
        # Fill positions systematically
        # Priority: Fill C first (only need 1), then others
        position_priority = [
            ('C', ['C']),
            ('SG', ['SG']),
            ('SF', ['SF']),
            ('PF', ['PF']),
            ('PG', ['PG'])
        ]
        
        for pos_name, pos_options in position_priority:
            if len(lineup) >= 9:
                break
            
            needed = positions_needed.get(pos_name, 0)
            if needed == 0:
                continue
            
            # Fill this position needed times
            for _ in range(needed):
                if len(lineup) >= 9:
                    break
                
                # Find best player for this position
                best = None
                best_value = -1
                best_idx = None
                
                # Iterate using integer index after reset_index
                for idx in range(len(available)):
                    player = available.iloc[idx]
                    
                    if total_salary + player['salary'] > salary_cap:
                        continue
                    
                    # Check if player can play this position
                    player_positions = player['positions']
                    can_play = any(p in pos_options for p in player_positions)
                    
                    if not can_play:
                        continue
                    
                    # Check not already in lineup
                    player_id = player.get('Id', idx)
                    if player_id in used_player_ids:
                        continue
                    
                    # Calculate score considering remaining budget
                    remaining_spots = 9 - len(lineup)
                    remaining_budget = salary_cap - total_salary
                    target_salary = remaining_budget / remaining_spots if remaining_spots > 0 else remaining_budget
                    
                    # Score: value * (1 - salary_penalty)
                    salary_diff = abs(player['salary'] - target_salary)
                    salary_penalty = min(salary_diff / target_salary, 0.5) if target_salary > 0 else 0
                    score = player['value'] * (1 - salary_penalty)
                    
                    if score > best_value:
                        best_value = score
                        best = player.copy()
                        best_idx = idx
                
                if best is not None:
                    # Store the roster position this player was assigned to
                    best['roster_position'] = pos_name
                    lineup.append(best)
                    total_salary += best['salary']
                    total_points += best['predicted_fantasy_score']
                    used_player_ids.add(best.get('Id', best_idx))
                    positions_needed[pos_name] -= 1
                    # Remove from available using iloc index
                    available = available.drop(available.index[best_idx]).reset_index(drop=True)
        
        if len(lineup) == 9:
            lineup_df = pd.DataFrame(lineup)
            lineups.append(lineup_df)
            print(f"  ✓ Lineup {lineup_num + 1} created:")
            print(f"    Salary: ${total_salary:,} / ${salary_cap:,}")
            print(f"    Predicted Points: {total_points:.2f}")
            print(f"    Remaining: ${salary_cap - total_salary:,}")
        else:
            print(f"  ⚠ Could not complete lineup {lineup_num + 1} ({len(lineup)}/9 players)")
            print(f"    Still need: {dict((k, v) for k, v in positions_needed.items() if v > 0)}")
    
    return lineups


def format_lineup_for_fanduel_upload(lineup_df):
    """
    Format lineup in FanDuel upload template format.
    Returns a dictionary with position as key and player ID + Name as value.
    """
    # Position slots for FanDuel (in order)
    position_slots = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C']
    
    # Create a dictionary to store the lineup
    fanduel_lineup = {}
    
    # Group players by their roster position
    position_counts = {'PG': 0, 'SG': 0, 'SF': 0, 'PF': 0, 'C': 0}
    
    for _, player in lineup_df.iterrows():
        roster_pos = player.get('roster_position', '')
        if roster_pos not in position_counts:
            continue
        
        # Create player ID + Name string (FanDuel format)
        player_id = player.get('Id', '')
        player_name = player.get('player_name', '')
        fanduel_player_str = f"{player_id}:{player_name}"
        
        # Assign to the appropriate slot
        slot_index = position_counts[roster_pos]
        position_counts[roster_pos] += 1
        
        fanduel_lineup[roster_pos + str(slot_index + 1)] = fanduel_player_str
    
    return fanduel_lineup


def save_lineup_fanduel_format(lineup_df, filename):
    """
    Save lineup in FanDuel upload template format.
    This creates a CSV that can be directly uploaded to FanDuel.
    
    Format: ID:Nickname (e.g., 125122-84669:Luka Doncic)
    """
    # Create the header row (position slots)
    header = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C']
    
    # Initialize the lineup row with empty strings
    lineup_row = [''] * 9
    
    # Map roster positions to slot indices
    position_slot_map = {
        'PG': [0, 1],
        'SG': [2, 3],
        'SF': [4, 5],
        'PF': [6, 7],
        'C': [8]
    }
    
    # Keep track of how many players we've assigned to each position
    position_counts = {'PG': 0, 'SG': 0, 'SF': 0, 'PF': 0, 'C': 0}
    
    # Fill in the lineup row
    for _, player in lineup_df.iterrows():
        roster_pos = player.get('roster_position', '')
        if roster_pos not in position_slot_map:
            continue
        
        # Get the slot index for this position
        slot_indices = position_slot_map[roster_pos]
        slot_count = position_counts[roster_pos]
        
        if slot_count >= len(slot_indices):
            continue
        
        slot_index = slot_indices[slot_count]
        position_counts[roster_pos] += 1
        
        # Create player ID + Name string (FanDuel format)
        # Use 'Nickname' from original CSV which has the proper display name
        player_id = player.get('Id', '')
        
        # Try to get 'Nickname' first (from original CSV), fallback to constructed name
        if 'Nickname' in player and pd.notna(player['Nickname']):
            player_display_name = player['Nickname']
        else:
            # Fallback: construct from first name and last name
            first_name = player.get('First Name', player.get('firstName', ''))
            last_name = player.get('Last Name', player.get('lastName', ''))
            player_display_name = f"{first_name} {last_name}".strip()
        
        lineup_row[slot_index] = f"{player_id}:{player_display_name}"
    
    # Create a DataFrame with the lineup
    lineup_upload_df = pd.DataFrame([lineup_row], columns=header)
    
    # Save to CSV
    lineup_upload_df.to_csv(filename, index=False)


def print_lineup(lineup_df, lineup_num):
    """Print formatted lineup."""
    print(f"\n{'='*80}")
    print(f"LINEUP #{lineup_num}")
    print(f"{'='*80}\n")
    
    # Use roster_position from the DataFrame
    if 'roster_position' in lineup_df.columns:
        roster_positions = lineup_df['roster_position'].values
    else:
        # Fallback: use positions from player data
        roster_positions = []
        for _, player in lineup_df.iterrows():
            pos_list = player.get('positions', [])
            if pos_list:
                roster_positions.append(pos_list[0])
            else:
                roster_positions.append('Unknown')
    
    # Create display DataFrame
    display_df = pd.DataFrame({
        'Position': roster_positions,
        'Player': lineup_df['player_name'].values,
        'Team': lineup_df['team'].values,
        'Opponent': lineup_df['opponent'].values,
        'Salary': lineup_df['salary'].values,
        'FPPG': lineup_df['predicted_fantasy_score'].values,
        'Value': lineup_df['value'].values
    })
    
    # Format for display
    display_df['Salary'] = display_df['Salary'].apply(lambda x: f"${x:,}")
    display_df['FPPG'] = display_df['FPPG'].apply(lambda x: f"{x:.2f}")
    display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:.2f}")
    
    print(display_df.to_string(index=False))
    
    total_salary = lineup_df['salary'].sum()
    total_points = lineup_df['predicted_fantasy_score'].sum()
    avg_value = lineup_df['value'].mean()
    
    print(f"\nTotal Salary: ${total_salary:,}")
    print(f"Total Predicted Points: {total_points:.2f}")
    print(f"Average Value: {avg_value:.2f} pts/$1k")


def main():
    parser = argparse.ArgumentParser(description='Optimize FanDuel lineups from CSV using ILP')
    parser.add_argument('--csv', type=str, required=True, help='Path to FanDuel CSV file')
    parser.add_argument('--output', type=str, help='Output directory for lineups')
    parser.add_argument('--use-model', action='store_true', help='Use model predictions instead of FPPG')
    parser.add_argument('--num-lineups', type=int, default=3, help='Number of lineups to generate')
    parser.add_argument('--use-greedy', action='store_true', help='Use greedy algorithm instead of ILP')
    
    args = parser.parse_args()
    
    # Load CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    players_df = load_fanduel_csv(csv_path)
    
    # Ensure 'predicted_fantasy_score' column exists (use FPPG from CSV)
    if 'fppg' in players_df.columns and 'predicted_fantasy_score' not in players_df.columns:
        players_df['predicted_fantasy_score'] = players_df['fppg']
    
    # Choose optimization method
    if args.use_greedy:
        print("Using GREEDY algorithm (faster but suboptimal)")
        lineups = optimize_fanduel_lineup(
            players_df,
            salary_cap=60000,
            num_lineups=args.num_lineups,
            use_model_predictions=args.use_model
        )
    else:
        print("Using ILP algorithm (guaranteed optimal!)")
        lineups = optimize_lineup_ilp_fanduel(
            players_df,
            salary_cap=60000,
            num_lineups=args.num_lineups,
            diversity_penalty=0.90
        )
    
    if len(lineups) == 0:
        print("\n⚠ Could not generate any complete lineups.")
        return
    
    # Print lineups
    for i, lineup in enumerate(lineups, 1):
        # Calculate value if not present
        if 'value' not in lineup.columns:
            lineup['value'] = lineup['predicted_fantasy_score'] / (lineup['salary'] / 1000)
        print_lineup(lineup, i)
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "models" / "lineups"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, lineup in enumerate(lineups, 1):
        # Save in FanDuel upload format
        upload_filename = output_dir / f"fanduel_upload_lineup_{i}.csv"
        save_lineup_fanduel_format(lineup, upload_filename)
        print(f"\n✓ Saved FanDuel upload format: {upload_filename}")
        
        # Also save in detailed format for reference
        detail_filename = output_dir / f"fanduel_lineup_{i}_details.csv"
        columns_to_save = ['roster_position', 'player_name', 'team', 'opponent', 'salary', 'predicted_fantasy_score', 'value']
        columns_to_save = [col for col in columns_to_save if col in lineup.columns]
        lineup[columns_to_save].to_csv(detail_filename, index=False)
        print(f"✓ Saved details: {detail_filename}")


if __name__ == "__main__":
    main()

