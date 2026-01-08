"""LightGBM model for NBA Fantasy Score Prediction."""

import sys
from pathlib import Path
import pandas as pd
import pickle
import yaml
import lightgbm as lgb

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.modeling.evaluate import print_metrics


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
    y_train = pd.read_csv(processed_path / "y_train.csv").squeeze()
    y_val = pd.read_csv(processed_path / "y_val.csv").squeeze()
    return X_train, X_val, y_train, y_val


def train_lightgbm(X_train, y_train, X_val, y_val, config=None):
    """Train LightGBM model with early stopping."""
    if config is None:
        config = load_config()
    
    lgb_config = config['models_config']['lightgbm']
    
    print("Training LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=lgb_config['n_estimators'],
        num_leaves=lgb_config['num_leaves'],
        learning_rate=lgb_config['learning_rate'],
        feature_fraction=lgb_config['feature_fraction'],
        random_state=lgb_config['random_state'],
        n_jobs=-1,
        verbosity=-1
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)])
    
    return model


if __name__ == "__main__":
    config = load_config()
    X_train, X_val, y_train, y_val = load_preprocessed_data()
    
    model = train_lightgbm(X_train, y_train, X_val, y_val, config)
    pred = model.predict(X_val)
    print_metrics(y_val, pred, "LightGBM")
    
    # Save model
    models_path = project_root / "models" / "saved"
    models_path.mkdir(parents=True, exist_ok=True)
    with open(models_path / "lightgbm.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nFeature importances (top 10):")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.nlargest(10))


