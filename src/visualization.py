import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_scatter_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                           title: str = "Predicted vs Actual pKd",
                           save_path: Optional[str] = None,
                           dpi: int = 300) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    min_val = min(y_true.min(), y_pred.min()) - 0.5
    max_val = max(y_true.max(), y_pred.max()) + 0.5
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect prediction')
    ax.plot([min_val, max_val], [min_val + 1, max_val + 1], 'r--', lw=1, alpha=0.5, label='±1 log unit')
    ax.plot([min_val, max_val], [min_val - 1, max_val - 1], 'r--', lw=1, alpha=0.5)
    
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    line = slope * y_true + intercept
    ax.plot(y_true, line, 'b-', lw=2, label=f'Linear fit (R²={r_value**2:.3f})')
    
    ax.set_xlabel('Actual pKd', fontsize=12)
    ax.set_ylabel('Predicted pKd', fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    from src.evaluation.metrics import calculate_mse, calculate_rmse, calculate_mae
    mse = calculate_mse(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    
    textstr = f'MSE: {mse:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r_value**2:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {save_path}")
    
    return fig


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                          title: str = "Error Distribution",
                          save_path: Optional[str] = None,
                          dpi: int = 300) -> plt.Figure:
    
    errors = y_pred - y_true
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    n, bins, patches = ax1.hist(errors, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    mu = np.mean(errors)
    sigma = np.std(errors)
    x = np.linspace(errors.min(), errors.max(), 100)
    ax1.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
             'r-', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
    
    ax1.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero error')
    ax1.set_xlabel('Prediction Error (pKd)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Error Distribution', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved error distribution plot to {save_path}")
    
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                  title: str = "Residual Analysis",
                  save_path: Optional[str] = None,
                  dpi: int = 300) -> plt.Figure:
    
    residuals = y_pred - y_true
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Predicted pKd', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residuals vs Predicted Values', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(y_true, residuals, alpha=0.6, color='orange')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Actual pKd', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residuals vs Actual Values', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    absolute_residuals = np.abs(residuals)
    ax3.scatter(y_pred, absolute_residuals, alpha=0.6, color='green')
    z = np.polyfit(y_pred, absolute_residuals, 1)
    p = np.poly1d(z)
    ax3.plot(y_pred, p(y_pred), "r--", linewidth=2)
    ax3.set_xlabel('Predicted pKd', fontsize=12)
    ax3.set_ylabel('|Residuals|', fontsize=12)
    ax3.set_title('Absolute Residuals vs Predicted', fontsize=13)
    ax3.grid(True, alpha=0.3)
    
    standardized_residuals = residuals / np.std(residuals)
    ax4.hist(standardized_residuals, bins=30, density=True, alpha=0.7, color='purple', edgecolor='black')
    x = np.linspace(-4, 4, 100)
    ax4.plot(x, 1/np.sqrt(2*np.pi) * np.exp(-0.5*x**2), 'r-', linewidth=2, label='Standard Normal')
    ax4.set_xlabel('Standardized Residuals', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Standardized Residual Distribution', fontsize=13)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved residual analysis plot to {save_path}")
    
    return fig


def plot_stratified_performance(performance_dict: Dict[str, Dict[str, float]],
                              metric: str = 'rmse',
                              title: str = "Performance by Category",
                              save_path: Optional[str] = None,
                              dpi: int = 300) -> plt.Figure:
    
    categories = []
    values = []
    counts = []
    
    for category, metrics in performance_dict.items():
        if category != 'overall' and metric in metrics:
            categories.append(category)
            values.append(metrics[metric])
            if 'count' in metrics:
                counts.append(metrics['count'])
            else:
                counts.append(None)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(categories, values, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        if count is not None:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={count}', ha='center', va='bottom', fontsize=9)
    
    if 'overall' in performance_dict and metric in performance_dict['overall']:
        ax.axhline(y=performance_dict['overall'][metric], color='red', 
                  linestyle='--', linewidth=2, label=f'Overall {metric.upper()}')
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved stratified performance plot to {save_path}")
    
    return fig


def plot_learning_curves(history: Dict[str, List[float]],
                       title: str = "Training History",
                       save_path: Optional[str] = None,
                       dpi: int = 300) -> plt.Figure:
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if 'val_rmse' in history:
        ax2.plot(epochs, history['val_rmse'], 'g-', label='Validation RMSE')
    if 'val_mae' in history:
        ax2.plot(epochs, history['val_mae'], 'm-', label='Validation MAE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title('Validation Metrics', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved learning curves to {save_path}")
    
    return fig


def create_all_visualizations(y_true: np.ndarray, y_pred: np.ndarray,
                            output_dir: str, prefix: str = "",
                            performance_dict: Optional[Dict] = None,
                            history: Optional[Dict] = None,
                            config: Optional[Dict] = None) -> Dict[str, str]:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    saved_plots = {}
    
    dpi = config['visualization']['dpi'] if config else 300
    formats = config['visualization']['formats'] if config else ['png']
    
    for fmt in formats:
        scatter_path = output_dir / f"{prefix}scatter_plot.{fmt}"
        plot_scatter_predictions(y_true, y_pred, save_path=str(scatter_path), dpi=dpi)
        saved_plots[f'scatter_{fmt}'] = str(scatter_path)
        
        error_path = output_dir / f"{prefix}error_distribution.{fmt}"
        plot_error_distribution(y_true, y_pred, save_path=str(error_path), dpi=dpi)
        saved_plots[f'error_dist_{fmt}'] = str(error_path)
        
        residual_path = output_dir / f"{prefix}residual_analysis.{fmt}"
        plot_residuals(y_true, y_pred, save_path=str(residual_path), dpi=dpi)
        saved_plots[f'residuals_{fmt}'] = str(residual_path)
        
        if performance_dict and 'metrics_by_length' in performance_dict:
            perf_path = output_dir / f"{prefix}performance_by_length.{fmt}"
            plot_stratified_performance(
                performance_dict['metrics_by_length'],
                save_path=str(perf_path), dpi=dpi
            )
            saved_plots[f'perf_by_length_{fmt}'] = str(perf_path)
        
        if history:
            curves_path = output_dir / f"{prefix}learning_curves.{fmt}"
            plot_learning_curves(history, save_path=str(curves_path), dpi=dpi)
            saved_plots[f'learning_curves_{fmt}'] = str(curves_path)
    
    plt.close('all')
    
    return saved_plots