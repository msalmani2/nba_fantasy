"""
Train all advanced models.

This script trains Random Forest, XGBoost, LightGBM, and CatBoost models.
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.modeling.random_forest import train_random_forest, load_preprocessed_data

# Try importing advanced models, skip if not available
try:
    from scripts.modeling.xgboost_model import train_xgboost
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"Warning: XGBoost not available: {e}")
    XGBOOST_AVAILABLE = False

try:
    from scripts.modeling.lightgbm_model import train_lightgbm
    LIGHTGBM_AVAILABLE = True
except Exception as e:
    print(f"Warning: LightGBM not available: {e}")
    LIGHTGBM_AVAILABLE = False

try:
    from scripts.modeling.catboost_model import train_catboost
    CATBOOST_AVAILABLE = True
except Exception as e:
    print(f"Warning: CatBoost not available: {e}")
    CATBOOST_AVAILABLE = False
from scripts.modeling.evaluate import print_metrics, compare_models
import pickle
import yaml


def load_config():
    """Load configuration."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_all_models():
    """Train all advanced models."""
    config = load_config()
    X_train, X_val, y_train, y_val = load_preprocessed_data()
    
    models = {}
    results = {}
    
    print("=" * 60)
    print("TRAINING ALL ADVANCED MODELS")
    print("=" * 60)
    
    # Random Forest
    print("\n" + "-" * 60)
    rf_model = train_random_forest(X_train, y_train, config)
    rf_pred = rf_model.predict(X_val)
    from scripts.modeling.evaluate import calculate_metrics
    rf_metrics = calculate_metrics(y_val, rf_pred)
    models['Random Forest'] = rf_model
    results['Random Forest'] = rf_metrics
    print_metrics(y_val, rf_pred, "Random Forest")
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("\n" + "-" * 60)
        try:
            xgb_model = train_xgboost(X_train, y_train, X_val, y_val, config)
            xgb_pred = xgb_model.predict(X_val)
            xgb_metrics = calculate_metrics(y_val, xgb_pred)
            models['XGBoost'] = xgb_model
            results['XGBoost'] = xgb_metrics
            print_metrics(y_val, xgb_pred, "XGBoost")
        except Exception as e:
            print(f"XGBoost training failed: {e}")
    else:
        print("\nSkipping XGBoost (not available)")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n" + "-" * 60)
        try:
            lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, config)
            lgb_pred = lgb_model.predict(X_val)
            lgb_metrics = calculate_metrics(y_val, lgb_pred)
            models['LightGBM'] = lgb_model
            results['LightGBM'] = lgb_metrics
            print_metrics(y_val, lgb_pred, "LightGBM")
        except Exception as e:
            print(f"LightGBM training failed: {e}")
    else:
        print("\nSkipping LightGBM (not available)")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        print("\n" + "-" * 60)
        try:
            cb_model = train_catboost(X_train, y_train, X_val, y_val, config)
            cb_pred = cb_model.predict(X_val)
            cb_metrics = calculate_metrics(y_val, cb_pred)
            models['CatBoost'] = cb_model
            results['CatBoost'] = cb_metrics
            print_metrics(y_val, cb_pred, "CatBoost")
        except Exception as e:
            print(f"CatBoost training failed: {e}")
    else:
        print("\nSkipping CatBoost (not available)")
    
    # Save models
    models_path = project_root / "models" / "saved"
    models_path.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        model_file = models_path / f"{name.lower().replace(' ', '_')}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nSaved {name} to {model_file}")
    
    # Save results
    results_file = models_path / "advanced_models_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    comparison_df = compare_models(results)
    print(comparison_df)
    
    print("\nBest model (by RMSE):", comparison_df['RMSE'].idxmin())
    print("Best RMSE:", comparison_df['RMSE'].min())
    
    return models, results


if __name__ == "__main__":
    models, results = train_all_models()

