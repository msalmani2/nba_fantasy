"""
Baseline models for NBA Fantasy Score Prediction.

This module implements simple baseline models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from scripts.modeling.evaluate import print_metrics, plot_predictions_vs_actual, calculate_metrics


def load_preprocessed_data():
    """Load preprocessed training and validation data."""
    processed_path = project_root / "data" / "processed" / "preprocessed"
    
    X_train = pd.read_csv(processed_path / "X_train.csv")
    X_val = pd.read_csv(processed_path / "X_val.csv")
    y_train = pd.read_csv(processed_path / "y_train.csv").squeeze()
    y_val = pd.read_csv(processed_path / "y_val.csv").squeeze()
    
    return X_train, X_val, y_train, y_val


def train_linear_regression(X_train, y_train):
    """Train Linear Regression model."""
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(X_train, y_train, alpha=1.0):
    """Train Ridge Regression model."""
    print("Training Ridge Regression...")
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_lasso_regression(X_train, y_train, alpha=1.0):
    """Train Lasso Regression model."""
    print("Training Lasso Regression...")
    model = Lasso(alpha=alpha, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=10):
    """Train Decision Tree Regressor."""
    print("Training Decision Tree...")
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_all_baseline_models(X_train, X_val, y_train, y_val):
    """
    Train all baseline models and evaluate them.
    
    Returns:
    --------
    dict
        Dictionary of trained models and their metrics
    """
    models = {}
    results = {}
    
    # Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    lr_pred = lr_model.predict(X_val)
    lr_metrics = calculate_metrics(y_val, lr_pred)
    models['Linear Regression'] = lr_model
    results['Linear Regression'] = lr_metrics
    print_metrics(y_val, lr_pred, "Linear Regression")
    
    # Ridge Regression
    ridge_model = train_ridge_regression(X_train, y_train, alpha=1.0)
    ridge_pred = ridge_model.predict(X_val)
    ridge_metrics = calculate_metrics(y_val, ridge_pred)
    models['Ridge Regression'] = ridge_model
    results['Ridge Regression'] = ridge_metrics
    print_metrics(y_val, ridge_pred, "Ridge Regression")
    
    # Lasso Regression
    lasso_model = train_lasso_regression(X_train, y_train, alpha=1.0)
    lasso_pred = lasso_model.predict(X_val)
    lasso_metrics = calculate_metrics(y_val, lasso_pred)
    models['Lasso Regression'] = lasso_model
    results['Lasso Regression'] = lasso_metrics
    print_metrics(y_val, lasso_pred, "Lasso Regression")
    
    # Decision Tree
    dt_model = train_decision_tree(X_train, y_train, max_depth=10)
    dt_pred = dt_model.predict(X_val)
    dt_metrics = calculate_metrics(y_val, dt_pred)
    models['Decision Tree'] = dt_model
    results['Decision Tree'] = dt_metrics
    print_metrics(y_val, dt_pred, "Decision Tree")
    
    return models, results


def save_models(models, results):
    """Save trained models and results."""
    models_path = project_root / "models" / "saved"
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Save each model
    for name, model in models.items():
        model_file = models_path / f"baseline_{name.lower().replace(' ', '_')}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} to {model_file}")
    
    # Save results
    results_file = models_path / "baseline_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {results_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE MODELS TRAINING")
    print("=" * 60)
    
    # Load data
    X_train, X_val, y_train, y_val = load_preprocessed_data()
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Train all baseline models
    models, results = train_all_baseline_models(X_train, X_val, y_train, y_val)
    
    # Save models
    save_models(models, results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE MODELS SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    print("\nBest model (by RMSE):", results_df['RMSE'].idxmin())
    print("Best RMSE:", results_df['RMSE'].min())


