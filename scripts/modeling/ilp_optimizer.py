"""
Integer Linear Programming (ILP) based lineup optimizer.

This module uses PuLP to find provably optimal lineups under constraints.
Much more powerful than greedy algorithms.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from pulp import *

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def optimize_lineup_ilp_fanduel(players_df, salary_cap=60000, num_lineups=3, diversity_penalty=0.90):
    """
    Optimize FanDuel lineup using Integer Linear Programming.
    
    Position requirements: 2 PG, 2 SG, 2 SF, 2 PF, 1 C (9 players, $60,000 cap)
    
    Parameters:
    -----------
    players_df : pd.DataFrame
        Players with positions, salaries, and predicted scores
    salary_cap : int
        Maximum total salary
    num_lineups : int
        Number of diverse lineups to generate
    diversity_penalty : float
        Penalty for using same players (0.9 = 10% penalty)
    
    Returns:
    --------
    list
        List of optimal lineups (DataFrames)
    """
    print(f"\n{'='*80}")
    print(f"ILP OPTIMIZATION - FANDUEL FORMAT (${salary_cap:,} cap)")
    print(f"{'='*80}")
    
    # Filter valid players
    def has_valid_positions(pos_list):
        return isinstance(pos_list, list) and len(pos_list) > 0
    
    valid = players_df[
        (players_df['salary'] > 0) &
        (players_df['predicted_fantasy_score'].notna()) &
        (players_df['positions'].apply(has_valid_positions))
    ].copy().reset_index(drop=True)
    
    print(f"Valid players: {len(valid)}")
    
    if len(valid) < 9:
        print(f"⚠ Not enough players ({len(valid)} < 9)")
        return []
    
    lineups = []
    used_players = {}  # Track players used in previous lineups
    
    for lineup_num in range(num_lineups):
        print(f"\nGenerating lineup {lineup_num + 1}...")
        
        # Create the optimization problem
        prob = LpProblem(f"FanDuel_Lineup_{lineup_num+1}", LpMaximize)
        
        # Decision variables: binary (1 if player is selected, 0 otherwise)
        player_vars = {}
        for idx in valid.index:
            player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')
        
        # Objective function: Maximize total predicted points
        # Apply diversity penalty for players used in previous lineups
        prob += lpSum([
            player_vars[idx] * valid.loc[idx, 'predicted_fantasy_score'] * 
            (diversity_penalty if idx in used_players else 1.0)
            for idx in valid.index
        ]), "Total_Points"
        
        # Constraint 1: Salary cap
        prob += lpSum([
            player_vars[idx] * valid.loc[idx, 'salary']
            for idx in valid.index
        ]) <= salary_cap, "Salary_Cap"
        
        # Constraint 2: Exactly 9 players
        prob += lpSum([player_vars[idx] for idx in valid.index]) == 9, "Total_Players"
        
        # Constraint 3: Position requirements
        positions_needed = {'PG': 2, 'SG': 2, 'SF': 2, 'PF': 2, 'C': 1}
        
        for pos, count in positions_needed.items():
            # Players who can play this position
            can_play_pos = valid.index[valid['positions'].apply(lambda x: pos in x)]
            prob += lpSum([player_vars[idx] for idx in can_play_pos]) == count, f"Position_{pos}"
        
        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output
        
        # Extract solution
        if prob.status == 1:  # Optimal solution found
            selected_indices = [idx for idx in valid.index if player_vars[idx].varValue == 1]
            lineup_df = valid.loc[selected_indices].copy()
            
            # Calculate lineup stats
            total_salary = lineup_df['salary'].sum()
            total_points = lineup_df['predicted_fantasy_score'].sum()
            
            # Assign roster positions based on positions needed
            lineup_df['roster_position'] = ''
            positions_assigned = {pos: 0 for pos in positions_needed.keys()}
            
            for idx in lineup_df.index:
                player_positions = lineup_df.loc[idx, 'positions']
                for pos in ['C', 'PG', 'SG', 'SF', 'PF']:  # Priority order
                    if pos in player_positions and positions_assigned[pos] < positions_needed[pos]:
                        lineup_df.loc[idx, 'roster_position'] = pos
                        positions_assigned[pos] += 1
                        break
            
            lineups.append(lineup_df)
            
            # Track used players for diversity
            for idx in selected_indices:
                used_players[idx] = used_players.get(idx, 0) + 1
            
            print(f"  ✓ Lineup {lineup_num + 1} created (ILP Optimal):")
            print(f"    Salary: ${total_salary:,} / ${salary_cap:,}")
            print(f"    Predicted Points: {total_points:.2f}")
            print(f"    Remaining: ${salary_cap - total_salary:,}")
        else:
            print(f"  ⚠ Could not find optimal solution (Status: {LpStatus[prob.status]})")
    
    return lineups


def optimize_lineup_ilp_draftkings(players_df, salary_cap=60000, num_lineups=3, diversity_penalty=0.90):
    """
    Optimize DraftKings lineup using Integer Linear Programming.
    
    Position requirements: 2 PG, 2 SG, 2 SF, 2 PF, 1 C (9 players, $60,000 cap)
    Same as FanDuel in this case.
    
    Parameters:
    -----------
    players_df : pd.DataFrame
        Players with positions, salaries, and predicted scores
    salary_cap : int
        Maximum total salary
    num_lineups : int
        Number of diverse lineups to generate
    diversity_penalty : float
        Penalty for using same players
    
    Returns:
    --------
    list
        List of optimal lineups (DataFrames)
    """
    # Same structure as FanDuel for now
    return optimize_lineup_ilp_fanduel(players_df, salary_cap, num_lineups, diversity_penalty)


def optimize_with_risk_constraint(players_df, salary_cap=60000, max_risk=10.0, min_points=250.0):
    """
    Optimize lineup with risk constraints.
    
    Parameters:
    -----------
    players_df : pd.DataFrame
        Players with positions, salaries, predicted scores, and std_dev
    salary_cap : int
        Maximum total salary
    max_risk : float
        Maximum total portfolio std dev
    min_points : float
        Minimum required predicted points
    
    Returns:
    --------
    pd.DataFrame
        Optimal risk-adjusted lineup
    """
    print(f"\n{'='*80}")
    print(f"RISK-ADJUSTED ILP OPTIMIZATION")
    print(f"Max Risk: {max_risk:.2f}, Min Points: {min_points:.2f}")
    print(f"{'='*80}")
    
    # Filter valid players
    def has_valid_positions(pos_list):
        return isinstance(pos_list, list) and len(pos_list) > 0
    
    valid = players_df[
        (players_df['salary'] > 0) &
        (players_df['predicted_fantasy_score'].notna()) &
        (players_df['positions'].apply(has_valid_positions))
    ].copy().reset_index(drop=True)
    
    # If no std_dev, add default
    if 'std_dev' not in valid.columns:
        valid['std_dev'] = valid['predicted_fantasy_score'] * 0.10
    
    # Create problem
    prob = LpProblem("Risk_Adjusted_Lineup", LpMaximize)
    
    # Decision variables
    player_vars = {}
    for idx in valid.index:
        player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')
    
    # Objective: Maximize risk-adjusted points (points - penalty * risk)
    risk_penalty = 1.0
    prob += lpSum([
        player_vars[idx] * (
            valid.loc[idx, 'predicted_fantasy_score'] - 
            risk_penalty * valid.loc[idx, 'std_dev']
        )
        for idx in valid.index
    ]), "Risk_Adjusted_Points"
    
    # Constraints
    prob += lpSum([player_vars[idx] * valid.loc[idx, 'salary'] for idx in valid.index]) <= salary_cap
    prob += lpSum([player_vars[idx] for idx in valid.index]) == 9
    
    # Minimum points constraint
    prob += lpSum([
        player_vars[idx] * valid.loc[idx, 'predicted_fantasy_score']
        for idx in valid.index
    ]) >= min_points, "Min_Points"
    
    # Maximum risk constraint (simplified - sum of std devs)
    prob += lpSum([
        player_vars[idx] * valid.loc[idx, 'std_dev']
        for idx in valid.index
    ]) <= max_risk, "Max_Risk"
    
    # Position requirements
    positions_needed = {'PG': 2, 'SG': 2, 'SF': 2, 'PF': 2, 'C': 1}
    for pos, count in positions_needed.items():
        can_play_pos = valid.index[valid['positions'].apply(lambda x: pos in x)]
        prob += lpSum([player_vars[idx] for idx in can_play_pos]) == count, f"Position_{pos}"
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))
    
    if prob.status == 1:
        selected_indices = [idx for idx in valid.index if player_vars[idx].varValue == 1]
        lineup_df = valid.loc[selected_indices].copy()
        
        total_salary = lineup_df['salary'].sum()
        total_points = lineup_df['predicted_fantasy_score'].sum()
        total_risk = lineup_df['std_dev'].sum()
        
        print(f"\n  ✓ Risk-adjusted lineup created:")
        print(f"    Salary: ${total_salary:,} / ${salary_cap:,}")
        print(f"    Predicted Points: {total_points:.2f}")
        print(f"    Total Risk (std dev): {total_risk:.2f}")
        print(f"    Risk-Adjusted Score: {total_points - total_risk:.2f}")
        
        return lineup_df
    else:
        print(f"  ⚠ Could not find solution (Status: {LpStatus[prob.status]})")
        return pd.DataFrame()


def compare_greedy_vs_ilp(players_df, salary_cap=60000):
    """
    Compare greedy algorithm vs ILP optimization.
    
    Parameters:
    -----------
    players_df : pd.DataFrame
        Players data
    salary_cap : int
        Salary cap
    
    Returns:
    --------
    dict
        Comparison results
    """
    print(f"\n{'='*80}")
    print("COMPARING GREEDY VS ILP OPTIMIZATION")
    print(f"{'='*80}")
    
    # ILP optimization
    ilp_lineups = optimize_lineup_ilp_fanduel(players_df, salary_cap, num_lineups=1)
    
    if len(ilp_lineups) > 0:
        ilp_lineup = ilp_lineups[0]
        ilp_points = ilp_lineup['predicted_fantasy_score'].sum()
        ilp_salary = ilp_lineup['salary'].sum()
        
        print(f"\nILP Lineup:")
        print(f"  Points: {ilp_points:.2f}")
        print(f"  Salary: ${ilp_salary:,}")
        print(f"  Efficiency: {ilp_points / (ilp_salary / 1000):.2f} pts/$1k")
        
        return {
            'ilp_points': ilp_points,
            'ilp_salary': ilp_salary,
            'ilp_lineup': ilp_lineup
        }
    else:
        print("⚠ Could not generate ILP lineup")
        return {}


if __name__ == "__main__":
    print("ILP Optimizer module loaded successfully!")
    print("\nAvailable functions:")
    print("  - optimize_lineup_ilp_fanduel()")
    print("  - optimize_lineup_ilp_draftkings()")
    print("  - optimize_with_risk_constraint()")
    print("  - compare_greedy_vs_ilp()")
    print("\nRequires: pulp library")

