"""
Evaluation utilities for wind power prediction models.
Computes metrics and generates visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE, handling zero values."""
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float
) -> float:
    """
    Compute pinball loss for a single quantile.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Quantile predictions
    quantile : float
        Quantile level
        
    Returns:
    --------
    float
        Pinball loss
    """
    errors = y_true - y_pred
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)


def coverage_probability(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> float:
    """
    Compute coverage probability (fraction of true values within bounds).
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    lower_bound : np.ndarray
        Lower uncertainty bounds
    upper_bound : np.ndarray
        Upper uncertainty bounds
        
    Returns:
    --------
    float
        Coverage probability (0-1)
    """
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    return coverage


def evaluate_probabilistic(
    y_true: np.ndarray,
    quantile_preds: np.ndarray,
    quantiles: List[float] = [0.1, 0.5, 0.9]
) -> Dict[str, float]:
    """
    Evaluate probabilistic predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    quantile_preds : np.ndarray
        Quantile predictions (n_samples, n_quantiles)
    quantiles : List[float]
        List of quantiles
        
    Returns:
    --------
    dict
        Probabilistic metrics
    """
    metrics = {}
    
    # Pinball loss for each quantile
    for i, q in enumerate(quantiles):
        metrics[f'pinball_loss_q{q}'] = pinball_loss(y_true, quantile_preds[:, i], q)
    
    # Average pinball loss
    metrics['mean_pinball_loss'] = np.mean([
        metrics[f'pinball_loss_q{q}'] for q in quantiles
    ])
    
    # Coverage for 80% interval (0.1 to 0.9 quantiles)
    if 0.1 in quantiles and 0.9 in quantiles:
        lower_idx = quantiles.index(0.1)
        upper_idx = quantiles.index(0.9)
        metrics['coverage_80'] = coverage_probability(
            y_true,
            quantile_preds[:, lower_idx],
            quantile_preds[:, upper_idx]
        )
    
    # Median prediction metrics
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    median_pred = quantile_preds[:, median_idx]
    metrics.update({
        f'MAE_median': mean_absolute_error(y_true, median_pred),
        f'RMSE_median': np.sqrt(mean_squared_error(y_true, median_pred)),
        f'R2_median': r2_score(y_true, median_pred)
    })
    
    return metrics


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Predictions vs Ground Truth",
    sample_size: int = 1000
):
    """
    Plot predictions against ground truth.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    save_path : Path, optional
        Path to save figure
    title : str
        Plot title
    sample_size : int
        Number of points to plot (for large datasets)
    """
    if len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(y_true_plot, y_pred_plot, alpha=0.5, s=10)
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('Ground Truth')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Time series (if sequential)
    n_plot = min(500, len(y_true_plot))
    axes[1].plot(y_true_plot[:n_plot], label='Ground Truth', alpha=0.7)
    axes[1].plot(y_pred_plot[:n_plot], label='Predicted', alpha=0.7)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Time Series Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_uncertainty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Predictions with Uncertainty Bounds",
    sample_size: int = 500
):
    """
    Plot predictions with uncertainty bounds.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values (median/mean)
    lower_bound : np.ndarray
        Lower uncertainty bounds
    upper_bound : np.ndarray
        Upper uncertainty bounds
    save_path : Path, optional
        Path to save figure
    title : str
        Plot title
    sample_size : int
        Number of points to plot
    """
    if len(y_true) > sample_size:
        indices = np.arange(0, len(y_true), len(y_true) // sample_size)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
        lower_plot = lower_bound[indices]
        upper_plot = upper_bound[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        lower_plot = lower_bound
        upper_plot = upper_bound
    
    plt.figure(figsize=(15, 6))
    x_axis = np.arange(len(y_true_plot))
    
    plt.fill_between(x_axis, lower_plot, upper_plot, alpha=0.3, label='Uncertainty bounds')
    plt.plot(x_axis, y_true_plot, label='Ground Truth', linewidth=1.5)
    plt.plot(x_axis, y_pred_plot, label='Predicted', linewidth=1.5, linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Power (kW)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_error_vs_wind_speed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wind_speed: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Error vs Wind Speed"
):
    """
    Plot prediction error as a function of wind speed.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    wind_speed : np.ndarray
        Wind speed values
    save_path : Path, optional
        Path to save figure
    title : str
        Plot title
    """
    errors = np.abs(y_true - y_pred)
    
    # Sample for visualization
    sample_size = min(5000, len(errors))
    indices = np.random.choice(len(errors), sample_size, replace=False)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(wind_speed[indices], errors[indices], alpha=0.3, s=10)
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Absolute Error (kW)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

