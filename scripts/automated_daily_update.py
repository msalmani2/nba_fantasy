"""
Automated daily update pipeline for NBA fantasy predictions.

This script:
1. Downloads latest dataset from Kaggle
2. Recalculates fantasy scores
3. Runs feature engineering on new data
4. Retrains models (weekly) or uses existing models (daily)
5. Generates predictions
6. Saves updated predictions and reports

Usage:
  python automated_daily_update.py --full  # Full retrain (weekly)
  python automated_daily_update.py         # Quick update (daily)
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
from datetime import datetime, timedelta
import kagglehub

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.load_data import load_player_statistics
from scripts.utils.fantasy_scoring import add_fantasy_score_column
from scripts.data_processing.feature_engineering import add_all_features
from scripts.data_processing.train_test_split import clean_and_split_data
from scripts.modeling.train_models import train_all_models
from scripts.modeling.predict import predict_fantasy_scores, save_predictions


def check_if_update_needed(last_update_file):
    """
    Check if data needs to be updated (once per day).
    
    Parameters:
    -----------
    last_update_file : Path
        Path to file storing last update timestamp
    
    Returns:
    --------
    bool
        True if update is needed
    """
    if not last_update_file.exists():
        return True
    
    with open(last_update_file, 'r') as f:
        last_update = f.read().strip()
    
    try:
        last_date = datetime.fromisoformat(last_update)
        now = datetime.now()
        
        # Update if last update was more than 23 hours ago
        return (now - last_date).total_seconds() > 23 * 3600
    except:
        return True


def update_dataset(force=False):
    """
    Update dataset from Kaggle.
    
    Parameters:
    -----------
    force : bool
        Force download even if recently updated
    
    Returns:
    --------
    Path
        Path to updated dataset
    """
    print(f"\n{'='*80}")
    print("UPDATING DATASET FROM KAGGLE")
    print(f"{'='*80}")
    
    path = kagglehub.dataset_download(
        'eoinamoore/historical-nba-data-and-player-box-scores',
        force_download=force
    )
    
    print(f"✓ Dataset updated: {path}")
    return Path(path)


def quick_daily_update():
    """
    Quick daily update without retraining models.
    """
    print(f"\n{'='*80}")
    print("QUICK DAILY UPDATE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Step 1: Update dataset
    try:
        dataset_path = update_dataset(force=False)
    except Exception as e:
        print(f"⚠ Dataset update failed: {e}")
        print("Using existing dataset...")
        dataset_path = None
    
    # Step 2: Load data
    print("\nLoading player statistics...")
    df = load_player_statistics()
    print(f"✓ Loaded {len(df):,} game records")
    
    # Step 3: Calculate fantasy scores
    print("\nCalculating fantasy scores...")
    df = add_fantasy_score_column(df)
    print("✓ Fantasy scores calculated")
    
    # Step 4: Generate predictions for recent players
    print("\nGenerating predictions...")
    # Filter to last 7 days for quick update
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    recent_date = df['gameDate'].max() - timedelta(days=7)
    recent_df = df[df['gameDate'] >= recent_date].copy()
    
    print(f"Processing {len(recent_df):,} recent games...")
    
    # Add features and predict
    recent_df = add_all_features(recent_df)
    predictions = predict_fantasy_scores(recent_df, use_ensemble=True)
    
    # Step 5: Save predictions
    output_path = project_root / "models" / "predictions" / f"daily_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    save_predictions(predictions, output_path)
    
    print(f"\n✓ Daily update complete!")
    print(f"Predictions saved: {output_path}")
    
    return predictions


def full_weekly_retrain():
    """
    Full weekly retrain with model updates.
    """
    print(f"\n{'='*80}")
    print("FULL WEEKLY RETRAIN")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Step 1: Update dataset (force download)
    dataset_path = update_dataset(force=True)
    
    # Step 2: Load full data
    print("\nLoading player statistics...")
    df = load_player_statistics()
    print(f"✓ Loaded {len(df):,} game records")
    
    # Step 3: Calculate fantasy scores
    print("\nCalculating fantasy scores...")
    df = add_fantasy_score_column(df)
    
    # Step 4: Feature engineering
    print("\nRunning feature engineering...")
    df = add_all_features(df)
    
    # Step 5: Data preprocessing
    print("\nPreprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = clean_and_split_data(df)
    
    # Step 6: Retrain models
    print("\nRetraining models...")
    models = train_all_models()
    
    # Step 7: Generate predictions
    print("\nGenerating full predictions...")
    predictions = predict_fantasy_scores(df, use_ensemble=True)
    
    # Step 8: Save everything
    output_path = project_root / "models" / "predictions" / f"weekly_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    save_predictions(predictions, output_path)
    
    print(f"\n✓ Weekly retrain complete!")
    print(f"Models and predictions updated: {output_path}")
    
    return predictions, models


def generate_update_report(predictions):
    """
    Generate a report of the update.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        Predictions DataFrame
    
    Returns:
    --------
    str
        Report text
    """
    report = []
    report.append("="*80)
    report.append("NBA FANTASY PREDICTION - UPDATE REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY:")
    report.append(f"  Total predictions: {len(predictions):,}")
    report.append(f"  Unique players: {predictions['personId'].nunique():,}")
    report.append(f"  Date range: {predictions['gameDate'].min()} to {predictions['gameDate'].max()}")
    report.append("")
    
    # Top predictions
    if 'predicted_fantasy_score' in predictions.columns:
        report.append("TOP 10 PREDICTIONS (Recent Games):")
        top_10 = predictions.nlargest(10, 'predicted_fantasy_score')
        for idx, row in top_10.iterrows():
            player_name = f"{row.get('firstName', '')} {row.get('lastName', '')}".strip()
            score = row['predicted_fantasy_score']
            date = row.get('gameDate', 'N/A')
            report.append(f"  {player_name}: {score:.2f} pts ({date})")
        report.append("")
    
    report.append("="*80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Automated daily update pipeline')
    parser.add_argument('--full', action='store_true', help='Full retrain (weekly)')
    parser.add_argument('--force', action='store_true', help='Force dataset download')
    parser.add_argument('--report', action='store_true', help='Generate and save report')
    
    args = parser.parse_args()
    
    # Check if update is needed
    last_update_file = project_root / "data" / ".last_update"
    
    if not args.force and not check_if_update_needed(last_update_file):
        print("✓ Data is up to date (updated within last 23 hours)")
        print("Use --force to update anyway")
        return
    
    # Run appropriate update
    try:
        if args.full:
            predictions, models = full_weekly_retrain()
        else:
            predictions = quick_daily_update()
        
        # Generate report
        if args.report:
            report = generate_update_report(predictions)
            print("\n" + report)
            
            # Save report
            report_path = project_root / "documentation" / "results" / f"update_report_{datetime.now().strftime('%Y%m%d')}.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\n✓ Report saved: {report_path}")
        
        # Update timestamp
        last_update_file.parent.mkdir(parents=True, exist_ok=True)
        with open(last_update_file, 'w') as f:
            f.write(datetime.now().isoformat())
        
        print(f"\n✅ Update complete!")
        
    except Exception as e:
        print(f"\n❌ Update failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

