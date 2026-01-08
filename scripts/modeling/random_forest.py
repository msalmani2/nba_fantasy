"""Random Forest model for NBA Fantasy Score Prediction."""

import sys
from pathlib import Path
import pandas as pd
import pickle
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.ensemble import RandomForestRegressor
from scripts.modeling.evaluate import print_metrics, calculate_metrics


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


def train_random_forest(X_train, y_train, config=None):
    """Train Random Forest model."""
    if config is None:
        config = load_config()
    
    rf_config = config['models_config']['random_forest']
    
    print("Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=rf_config['n_estimators'],
        max_depth=rf_config['max_depth'],
        min_samples_split=rf_config['min_samples_split'],
        min_samples_leaf=rf_config.get('min_samples_leaf', 2),
        random_state=rf_config['random_state'],
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    config = load_config()
    X_train, X_val, y_train, y_val = load_preprocessed_data()
    
    model = train_random_forest(X_train, y_train, config)
    pred = model.predict(X_val)
    metrics = print_metrics(y_val, pred, "Random Forest")
    
    # Save model
    models_path = project_root / "models" / "saved"
    models_path.mkdir(parents=True, exist_ok=True)
    with open(models_path / "random_forest.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nFeature importances (top 10):")
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(importances.nlargest(10))


