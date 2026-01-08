"""
Prediction pipeline for NBA Fantasy Score Prediction.

This script loads trained models and makes predictions on new data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_processing.load_data import load_player_statistics
from scripts.utils.fantasy_scoring import add_fantasy_score_column
from scripts.data_processing.feature_engineering import create_all_features, get_feature_list
from scripts.data_processing.train_test_split import clean_data, encode_categorical_features


def load_config():
    """Load configuration."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_ensemble_model():
    """Load the best ensemble model."""
    models_path = project_root / "models" / "saved"
    ensemble_file = models_path / "ensemble_models.pkl"
    
    if ensemble_file.exists():
        with open(ensemble_file, 'rb') as f:
            ensemble_models = pickle.load(f)
        return ensemble_models
    else:
        raise FileNotFoundError("Ensemble models not found. Please train models first.")


def load_base_models():
    """Load base models for ensemble."""
    models_path = project_root / "models" / "saved"
    
    models = {}
    model_files = {
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'LightGBM': 'lightgbm.pkl',
        'CatBoost': 'catboost.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = models_path / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
    
    return models


def load_feature_list():
    """Load the feature list used during training."""
    processed_path = project_root / "data" / "processed" / "preprocessed"
    feature_file = processed_path / "feature_list.pkl"
    
    if feature_file.exists():
        with open(feature_file, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError("Feature list not found. Please run preprocessing first.")


def preprocess_new_data(df):
    """
    Preprocess new data to match training data format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        New data with player statistics
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for prediction
    """
    # Apply fantasy scoring
    df = add_fantasy_score_column(df)
    
    # Create features
    df = create_all_features(df)
    
    # Clean data
    df = clean_data(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    return df


def predict_fantasy_scores(new_data, use_ensemble=True):
    """
    Predict fantasy scores for new data.
    
    Parameters:
    -----------
    new_data : pd.DataFrame
        New player statistics data
    use_ensemble : bool
        Whether to use ensemble model (default: True)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions added
    """
    print("Preprocessing new data...")
    df = preprocess_new_data(new_data.copy())
    
    # Load feature list
    feature_list = load_feature_list()
    
    # Get features
    X = df[feature_list].fillna(0)
    
    if use_ensemble:
        print("Loading ensemble model...")
        ensemble_models = load_ensemble_model()
        base_models = load_base_models()
        
        # Use weighted averaging (best performing typically)
        if 'weighted_averaging' in ensemble_models and ensemble_models['weighted_averaging'] is not None:
            weights = ensemble_models['weighted_averaging']
            predictions = []
            for name, model in base_models.items():
                pred = model.predict(X)
                predictions.append(pred)
            predictions = np.array(predictions)
            df['predicted_fantasy_score'] = np.average(predictions, axis=0, weights=weights)
        else:
            # Fallback to simple averaging
            predictions = []
            for name, model in base_models.items():
                pred = model.predict(X)
                predictions.append(pred)
            df['predicted_fantasy_score'] = np.mean(predictions, axis=0)
    else:
        # Use single best model (XGBoost typically)
        print("Loading XGBoost model...")
        models_path = project_root / "models" / "saved"
        with open(models_path / "xgboost.pkl", 'rb') as f:
            model = pickle.load(f)
        df['predicted_fantasy_score'] = model.predict(X)
    
    return df


def save_predictions(df, output_path=None):
    """Save predictions to file."""
    if output_path is None:
        predictions_path = project_root / "models" / "predictions"
        predictions_path.mkdir(parents=True, exist_ok=True)
        output_path = predictions_path / "predictions.csv"
    
    # Save relevant columns
    output_cols = ['firstName', 'lastName', 'personId', 'gameId', 'gameDateTimeEst',
                   'playerteamName', 'opponentteamName', 'fantasyScore', 'predicted_fantasy_score']
    
    available_cols = [col for col in output_cols if col in df.columns]
    df[available_cols].to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Example: Load latest data and make predictions
    print("Loading latest data...")
    df = load_player_statistics(save_raw=False)
    
    # Make predictions
    df_with_predictions = predict_fantasy_scores(df, use_ensemble=True)
    
    # Save predictions
    save_predictions(df_with_predictions)
    
    print("\nPredictions complete!")
    print(f"Total predictions: {len(df_with_predictions)}")
    print(f"\nSample predictions:")
    print(df_with_predictions[['firstName', 'lastName', 'fantasyScore', 'predicted_fantasy_score']].head(10))


