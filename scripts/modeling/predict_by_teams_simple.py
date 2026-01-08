"""
Simple interactive script to get predictions by team names.

This is a simpler version that's easier to use interactively.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.modeling.predict_by_teams import predict_for_teams, display_results, get_available_teams


def main():
    print("\n" + "="*80)
    print("NBA FANTASY SCORE PREDICTIONS BY TEAM")
    print("="*80)
    print("\nEnter team names (separated by commas, or 'list' to see all teams)")
    print("Example: Lakers, Warriors, Celtics")
    print("Or: list")
    print("\nType 'quit' or 'exit' to stop\n")
    
    while True:
        user_input = input("Enter team names: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'list':
            print("\nAvailable Teams:")
            print("="*80)
            teams = get_available_teams()
            for i, team in enumerate(teams, 1):
                print(f"{i:3d}. {team}")
            print(f"\nTotal: {len(teams)} teams\n")
            continue
        
        if not user_input:
            continue
        
        # Parse team names
        team_names = [name.strip() for name in user_input.split(',')]
        
        # Ask for top N
        top_n_input = input("Show top N players? (press Enter for all, or enter a number): ").strip()
        top_n = None
        if top_n_input:
            try:
                top_n = int(top_n_input)
            except ValueError:
                print("Invalid number, showing all players")
        
        # Ask for date range
        months_input = input("How many months back? (press Enter for 3 months, or 'all' for all dates): ").strip()
        recent_only = True
        months_back = 3
        if months_input.lower() == 'all':
            recent_only = False
        elif months_input:
            try:
                months_back = int(months_input)
            except ValueError:
                print("Invalid number, using 3 months")
        
        # Get predictions
        try:
            result_df = predict_for_teams(
                team_names,
                use_latest_data=False,  # Use existing predictions for speed
                top_n=top_n,
                recent_only=recent_only,
                months_back=months_back
            )
            
            if result_df is not None and len(result_df) > 0:
                display_results(result_df, team_names)
                
                # Ask if user wants to save
                save_input = input("Save results to CSV? (y/n): ").strip().lower()
                if save_input == 'y':
                    filename = input("Enter filename (or press Enter for default): ").strip()
                    if not filename:
                        filename = f"predictions_{'_'.join(team_names)}.csv"
                    
                    output_file = project_root / "models" / "predictions" / filename
                    result_df.to_csv(output_file, index=False)
                    print(f"\n✓ Results saved to: {output_file}\n")
            else:
                print("\nNo players found for the specified teams.\n")
                
        except Exception as e:
            print(f"\n⚠ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

