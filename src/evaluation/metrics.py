import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error using sklearn."""
    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error using sklearn."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error using sklearn."""
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² (coefficient of determination) using sklearn."""
    return r2_score(y_true, y_pred)


def calculate_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Calculate Pearson correlation coefficient."""
    if len(y_true) < 3:
        return np.nan, np.nan
    
    # Use scipy for p-value calculation
    correlation, p_value = stats.pearsonr(y_true, y_pred)
    return correlation, p_value


def calculate_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Calculate Spearman rank correlation coefficient."""
    if len(y_true) < 3:
        return np.nan, np.nan
    
    # Use scipy for p-value calculation
    correlation, p_value = stats.spearmanr(y_true, y_pred)
    return correlation, p_value


def calculate_fraction_within_threshold(y_true: np.ndarray, y_pred: np.ndarray,
                                      threshold: float) -> float:
    within_threshold = np.abs(y_true - y_pred) < threshold
    return np.mean(within_threshold)


def calculate_relative_error(y_true: np.ndarray, y_pred: np.ndarray,
                           epsilon: float = 1e-8) -> np.ndarray:
    return np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         denormalize_fn: Optional[callable] = None) -> Dict[str, float]:
    
    # If denormalize_fn is None, assume data is already on original scale
    if denormalize_fn is not None:
        y_true_original = denormalize_fn(y_true)
        y_pred_original = denormalize_fn(y_pred)
        # Also calculate normalized metrics
        metrics_normalized = {
            'mse_normalized': calculate_mse(y_true, y_pred),
            'rmse_normalized': calculate_rmse(y_true, y_pred),
            'mae_normalized': calculate_mae(y_true, y_pred),
            'r2_normalized': calculate_r2(y_true, y_pred)
        }
    else:
        y_true_original = y_true
        y_pred_original = y_pred
        metrics_normalized = {}
    
    metrics = {}
    
    # Primary metrics on original pKd scale using sklearn
    metrics['mse'] = calculate_mse(y_true_original, y_pred_original)
    metrics['rmse'] = calculate_rmse(y_true_original, y_pred_original)
    metrics['mae'] = calculate_mae(y_true_original, y_pred_original)
    metrics['r2'] = calculate_r2(y_true_original, y_pred_original)
    
    # Additional sklearn metrics
    metrics['explained_variance'] = explained_variance_score(y_true_original, y_pred_original)
    metrics['max_error'] = max_error(y_true_original, y_pred_original)
    
    # Mean Absolute Percentage Error (if no zeros in y_true)
    if np.all(y_true_original != 0):
        metrics['mape'] = mean_absolute_percentage_error(y_true_original, y_pred_original)
    else:
        metrics['mape'] = np.nan
    
    # Correlation metrics
    pearson_r, pearson_p = calculate_pearson(y_true_original, y_pred_original)
    metrics['pearson_r'] = pearson_r
    metrics['pearson_p'] = pearson_p
    
    spearman_r, spearman_p = calculate_spearman(y_true_original, y_pred_original)
    metrics['spearman_r'] = spearman_r
    metrics['spearman_p'] = spearman_p
    
    # Domain-specific metrics
    metrics['fraction_within_1'] = calculate_fraction_within_threshold(
        y_true_original, y_pred_original, 1.0
    )
    metrics['fraction_within_2'] = calculate_fraction_within_threshold(
        y_true_original, y_pred_original, 2.0
    )
    
    relative_errors = calculate_relative_error(y_true_original, y_pred_original)
    metrics['mean_relative_error'] = np.mean(relative_errors)
    metrics['median_relative_error'] = np.median(relative_errors)
    
    # Add normalized metrics if applicable
    metrics.update(metrics_normalized)
    
    return metrics


def calculate_stratified_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               stratify_by: np.ndarray, 
                               strata_names: Optional[Dict] = None,
                               denormalize_fn: Optional[callable] = None) -> Dict[str, Dict[str, float]]:
    
    unique_strata = np.unique(stratify_by)
    stratified_metrics = {}
    
    for stratum in unique_strata:
        mask = stratify_by == stratum
        stratum_name = strata_names.get(stratum, f"stratum_{stratum}") if strata_names else f"stratum_{stratum}"
        
        if np.sum(mask) < 3:
            logger.warning(f"Stratum {stratum_name} has fewer than 3 samples, skipping metrics")
            continue
        
        stratum_metrics = calculate_all_metrics(
            y_true[mask], y_pred[mask], denormalize_fn
        )
        stratified_metrics[stratum_name] = stratum_metrics
    
    overall_metrics = calculate_all_metrics(y_true, y_pred, denormalize_fn)
    stratified_metrics['overall'] = overall_metrics
    
    return stratified_metrics


def analyze_errors_by_range(y_true: np.ndarray, y_pred: np.ndarray,
                          num_bins: int = 5) -> Dict[str, Dict[str, float]]:
    
    bin_edges = np.percentile(y_true, np.linspace(0, 100, num_bins + 1))
    bin_indices = np.digitize(y_true, bin_edges[1:-1])
    
    error_analysis = {}
    
    for i in range(num_bins):
        mask = bin_indices == i
        
        if np.sum(mask) == 0:
            continue
        
        bin_range = f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
        
        errors = y_pred[mask] - y_true[mask]
        
        error_analysis[bin_range] = {
            'count': int(np.sum(mask)),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'mse': float(calculate_mse(y_true[mask], y_pred[mask])),
            'rmse': float(calculate_rmse(y_true[mask], y_pred[mask])),
            'mae': float(calculate_mae(y_true[mask], y_pred[mask])),
            'r2': float(calculate_r2(y_true[mask], y_pred[mask])),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors))
        }
    
    return error_analysis


def print_metrics_summary(metrics: Dict[str, float]):
    logger.info("\n=== Evaluation Metrics ===")
    
    # Primary metrics
    if 'mse' in metrics:
        logger.info(f"MSE: {metrics['mse']:.4f}")
    if 'rmse' in metrics:
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
    if 'mae' in metrics:
        logger.info(f"MAE: {metrics['mae']:.4f}")
    if 'r2' in metrics:
        logger.info(f"R²: {metrics['r2']:.4f}")
    
    # Additional sklearn metrics
    if 'explained_variance' in metrics:
        logger.info(f"Explained Variance: {metrics['explained_variance']:.4f}")
    if 'max_error' in metrics:
        logger.info(f"Max Error: {metrics['max_error']:.4f}")
    if 'mape' in metrics and not np.isnan(metrics['mape']):
        logger.info(f"MAPE: {metrics['mape']:.2%}")
    
    # Correlation metrics
    if 'pearson_r' in metrics:
        logger.info(f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    if 'spearman_r' in metrics:
        logger.info(f"Spearman ρ: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.2e})")
    
    # Domain-specific metrics
    if 'fraction_within_1' in metrics:
        logger.info(f"Fraction within 1 log unit: {metrics['fraction_within_1']:.3f}")
    if 'fraction_within_2' in metrics:
        logger.info(f"Fraction within 2 log units: {metrics['fraction_within_2']:.3f}")
    if 'mean_relative_error' in metrics:
        logger.info(f"Mean relative error: {metrics['mean_relative_error']:.3f}")


def create_performance_report(y_true: np.ndarray, y_pred: np.ndarray,
                            sequence_lengths: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                            denormalize_fn: Optional[callable] = None) -> Dict:
    
    report = {
        'overall_metrics': calculate_all_metrics(y_true, y_pred, denormalize_fn),
        'error_by_range': analyze_errors_by_range(
            denormalize_fn(y_true) if denormalize_fn else y_true,
            denormalize_fn(y_pred) if denormalize_fn else y_pred
        )
    }
    
    if sequence_lengths is not None:
        lengths1, lengths2 = sequence_lengths
        total_lengths = lengths1 + lengths2
        
        length_bins = [0, 200, 500, 1000, float('inf')]
        length_categories = np.digitize(total_lengths, length_bins)
        length_names = {
            0: 'very_short_<200',
            1: 'short_200-500',
            2: 'medium_500-1000',
            3: 'long_>1000'
        }
        
        report['metrics_by_length'] = calculate_stratified_metrics(
            y_true, y_pred, length_categories, length_names, denormalize_fn
        )
    
    return report