"""
Simplified DraftKings lineup optimizer that ensures complete lineups.
"""

import pandas as pd
import numpy as np

def optimize_draftkings_lineup_simple(players_df, salary_cap=60000, num_lineups=2):
    """
    Optimize DraftKings lineups with a simpler, more reliable algorithm.
    
    Requirements:
    - 2 PG, 2 SG, 2 SF, 2 PF, 1 C
    - Total: 9 players
    - Salary cap: $60,000
    """
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
    
    if len(valid) < 9:
        return []
    
    # Sort by value
    valid = valid.sort_values('value', ascending=False).reset_index(drop=True)
    
    lineups = []
    
    for lineup_num in range(num_lineups):
        lineup = []
        total_salary = 0
        total_points = 0
        
        positions_needed = {'PG': 2, 'SG': 2, 'SF': 2, 'PF': 2, 'C': 1}
        
        # Create available pool (excluding previous lineups)
        available = valid.copy().reset_index(drop=True)
        for prev_lineup in lineups:
            if 'personId' in prev_lineup.columns:
                used_ids = prev_lineup['personId'].values
                available = available[~available['personId'].isin(used_ids)].reset_index(drop=True)
        
        # Fill positions with balanced salary approach
        # Strategy: Fill expensive positions first, save budget for remaining spots
        used_player_ids = set()
        
        # Calculate average salary per remaining spot
        def get_avg_salary_per_spot():
            remaining_spots = 9 - len(lineup)
            if remaining_spots == 0:
                return 0
            remaining_budget = salary_cap - total_salary
            return remaining_budget / remaining_spots
        
        # Fill positions - prioritize by scarcity and value
        position_order = ['C', 'SG', 'SF', 'PF', 'PG']  # Fill C first (only need 1)
        
        for pos in position_order:
            needed = positions_needed.get(pos, 0)
            for _ in range(needed):
                if len(lineup) >= 9:
                    break
                
                # Calculate target salary (leave room for remaining positions)
                remaining_spots = 9 - len(lineup)
                remaining_budget = salary_cap - total_salary
                target_salary = remaining_budget / remaining_spots if remaining_spots > 0 else remaining_budget
                
                # Find best player for this position (considering value and salary)
                best = None
                best_score = -1
                best_idx = None
                
                for idx in available.index:
                    player = available.loc[idx]
                    
                    if total_salary + player['salary'] > salary_cap:
                        continue
                    
                    # Check if player can play this position
                    if pos not in player['positions']:
                        continue
                    
                    # Check not already in lineup
                    player_id = player.get('personId', idx)
                    if player_id in used_player_ids:
                        continue
                    
                    # Score: value * (1 - salary_penalty)
                    # Prefer players close to target salary
                    salary_diff = abs(player['salary'] - target_salary)
                    salary_penalty = min(salary_diff / target_salary, 0.5) if target_salary > 0 else 0
                    score = player['value'] * (1 - salary_penalty)
                    
                    if score > best_score:
                        best_score = score
                        best = player.copy()
                        best_idx = idx
                
                if best is not None:
                    lineup.append(best)
                    total_salary += best['salary']
                    total_points += best['predicted_fantasy_score']
                    used_player_ids.add(best.get('personId', best_idx))
                    positions_needed[pos] -= 1
                    # Remove from available
                    available = available.drop(best_idx).reset_index(drop=True)
        
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
            print(f"    Remaining salary: ${salary_cap - total_salary:,}")
            print(f"    Available players: {len(available)}")
            # Show available players for missing positions
            for pos, needed in positions_needed.items():
                if needed > 0:
                    pos_players = available[available['positions'].apply(lambda x: pos in x if isinstance(x, list) else False)]
                    affordable = pos_players[pos_players['salary'] <= (salary_cap - total_salary)]
                    print(f"      {pos} ({needed} needed): {len(affordable)} affordable players available")
    
    return lineups

