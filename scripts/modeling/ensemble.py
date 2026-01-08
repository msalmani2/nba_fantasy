"""
Ensemble models for NBA Fantasy Score Prediction.

This module implements:
- Averaging Ensemble
- Weighted Averaging Ensemble
- Stacking Ensemble
- Blending Ensemble
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.modeling.evaluate import print_metrics, calculate_metrics, compare_models


def load_config():
    """Load configuration."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_preprocessed_data():
    """Load preprocessed data."""
    processed_path = project_root / "data" / "processed" / "preprocessed"
    X_train = pd.read_csv(processed_path / "X_train.csv")
    X_val = pd.read_csv(processed_path / "X_val.csv")
    X_test = pd.read_csv(processed_path / "X_test.csv")
    y_train = pd.read_csv(processed_path / "y_train.csv").squeeze()
    y_val = pd.read_csv(processed_path / "y_val.csv").squeeze()
    y_test = pd.read_csv(processed_path / "y_test.csv").squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_base_models():
    """Load trained base models."""
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
            print(f"Loaded {name}")
        else:
            print(f"Warning: {filename} not found")
    
    return models


def averaging_ensemble(models, X):
    """Simple averaging ensemble."""
    predictions = []
    for name, model in models.items():
        pred = model.predict(X)
        predictions.append(pred)
    
    return np.mean(predictions, axis=0)


def weighted_averaging_ensemble(models, X, weights=None):
    """Weighted averaging ensemble."""
    if weights is None:
        # Equal weights
        weights = [1.0 / len(models)] * len(models)
    
    predictions = []
    for name, model in models.items():
        pred = model.predict(X)
        predictions.append(pred)
    
    weighted_pred = np.average(predictions, axis=0, weights=weights)
    return weighted_pred


def optimize_weights(models, X_val, y_val):
    """Optimize weights for weighted averaging using validation set."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("Warning: scipy not available. Using equal weights.")
        return [1.0 / len(models)] * len(models)
    
    predictions = []
    for name, model in models.items():
        pred = model.predict(X_val)
        predictions.append(pred)
    predictions = np.array(predictions)
    
    def objective(weights):
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        mse = np.mean((y_val - weighted_pred) ** 2)
        return mse
    
    # Constraint: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * len(models)
    initial_weights = [1.0 / len(models)] * len(models)
    
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result.x


def stacking_ensemble(models, X_train, y_train, X_val, y_val, meta_learner=None):
    """Stacking ensemble with meta-learner."""
    if meta_learner is None:
        meta_learner = Ridge(alpha=1.0)
    
    print("Generating meta-features...")
    # Generate meta-features on validation set
    meta_features = []
    for name, model in models.items():
        pred = model.predict(X_val)
        meta_features.append(pred)
    
    meta_features = np.column_stack(meta_features)
    
    # Train meta-learner
    print("Training meta-learner...")
    meta_learner.fit(meta_features, y_val)
    
    return meta_learner


def blending_ensemble(models, X_train, y_train, X_val, y_val):
    """Blending ensemble - similar to stacking but simpler."""
    # Generate predictions on validation set
    val_predictions = []
    for name, model in models.items():
        pred = model.predict(X_val)
        val_predictions.append(pred)
    
    val_predictions = np.column_stack(val_predictions)
    
    # Learn optimal blend weights
    from sklearn.linear_model import LinearRegression
    blender = LinearRegression()
    blender.fit(val_predictions, y_val)
    
    return blender


def evaluate_ensemble_methods(models, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate all ensemble methods."""
    results = {}
    
    print("=" * 60)
    print("ENSEMBLE MODEL EVALUATION")
    print("=" * 60)
    
    # Averaging Ensemble
    print("\n" + "-" * 60)
    print("Averaging Ensemble")
    print("-" * 60)
    avg_pred_val = averaging_ensemble(models, X_val)
    avg_pred_test = averaging_ensemble(models, X_test)
    avg_metrics_val = calculate_metrics(y_val, avg_pred_val)
    avg_metrics_test = calculate_metrics(y_test, avg_pred_test)
    results['Averaging Ensemble'] = avg_metrics_test
    print_metrics(y_val, avg_pred_val, "Averaging (Validation)")
    print_metrics(y_test, avg_pred_test, "Averaging (Test)")
    
    # Weighted Averaging Ensemble
    print("\n" + "-" * 60)
    print("Weighted Averaging Ensemble")
    print("-" * 60)
    weights = optimize_weights(models, X_val, y_val)
    print(f"Optimized weights: {dict(zip(models.keys(), weights))}")
    weighted_pred_val = weighted_averaging_ensemble(models, X_val, weights)
    weighted_pred_test = weighted_averaging_ensemble(models, X_test, weights)
    weighted_metrics_test = calculate_metrics(y_test, weighted_pred_test)
    results['Weighted Averaging'] = weighted_metrics_test
    print_metrics(y_val, weighted_pred_val, "Weighted Averaging (Validation)")
    print_metrics(y_test, weighted_pred_test, "Weighted Averaging (Test)")
    
    # Stacking Ensemble
    print("\n" + "-" * 60)
    print("Stacking Ensemble")
    print("-" * 60)
    meta_learner = stacking_ensemble(models, X_train, y_train, X_val, y_val)
    
    # Generate meta-features for test set
    test_meta_features = []
    for name, model in models.items():
        pred = model.predict(X_test)
        test_meta_features.append(pred)
    test_meta_features = np.column_stack(test_meta_features)
    
    stacking_pred_test = meta_learner.predict(test_meta_features)
    stacking_metrics_test = calculate_metrics(y_test, stacking_pred_test)
    results['Stacking Ensemble'] = stacking_metrics_test
    print_metrics(y_test, stacking_pred_test, "Stacking (Test)")
    
    # Blending Ensemble
    print("\n" + "-" * 60)
    print("Blending Ensemble")
    print("-" * 60)
    blender = blending_ensemble(models, X_train, y_train, X_val, y_val)
    
    test_blend_features = []
    for name, model in models.items():
        pred = model.predict(X_test)
        test_blend_features.append(pred)
    test_blend_features = np.column_stack(test_blend_features)
    
    blending_pred_test = blender.predict(test_blend_features)
    blending_metrics_test = calculate_metrics(y_test, blending_pred_test)
    results['Blending Ensemble'] = blending_metrics_test
    print_metrics(y_test, blending_pred_test, "Blending (Test)")
    
    return results, {
        'averaging': None,
        'weighted_averaging': weights,
        'stacking': meta_learner,
        'blending': blender
    }


if __name__ == "__main__":
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    # Load base models
    models = load_base_models()
    
    if len(models) == 0:
        print("No base models found. Please train base models first.")
        sys.exit(1)
    
    # Evaluate ensemble methods
    results, ensemble_models = evaluate_ensemble_methods(
        models, X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Compare ensemble results
    print("\n" + "=" * 60)
    print("ENSEMBLE COMPARISON")
    print("=" * 60)
    comparison_df = compare_models(results)
    print(comparison_df)
    
    print("\nBest ensemble (by RMSE):", comparison_df['RMSE'].idxmin())
    print("Best RMSE:", comparison_df['RMSE'].min())
    
    # Save ensemble models
    models_path = project_root / "models" / "saved"
    with open(models_path / "ensemble_models.pkl", 'wb') as f:
        pickle.dump(ensemble_models, f)
    
    with open(models_path / "ensemble_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print("\nEnsemble models and results saved.")

