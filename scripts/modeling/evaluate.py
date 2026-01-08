"""
Model evaluation utilities.

This module provides functions for evaluating model performance
with various metrics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def print_metrics(y_true, y_pred, model_name="Model"):
    """
    Print evaluation metrics in a formatted way.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print("-" * 40)
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R²:   {metrics['R2']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print("-" * 40)
    
    return metrics


def plot_predictions_vs_actual(y_true, y_pred, model_name="Model", save_path=None):
    """
    Create visualization of predictions vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Fantasy Score')
    axes[0].set_ylabel('Predicted Fantasy Score')
    axes[0].set_title(f'{model_name}: Predictions vs Actual')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Fantasy Score')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residuals Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_models(results_dict, save_path=None):
    """
    Compare multiple models' performance.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics dicts as values
    save_path : str, optional
        Path to save the comparison plot
    """
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_dict).T
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        comparison_df[metric].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return comparison_df


