#!/usr/bin/env python3
"""
Investigate R² calculation issue - compare normalized vs unnormalized metrics
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

def calculate_r2_manual(y_true, y_pred):
    """Calculate R² manually with detailed steps."""
    # Calculate components
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Calculate R²
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Also calculate using correlation approach
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    r2_from_corr = correlation ** 2
    
    return {
        'r2': r2,
        'r2_from_correlation': r2_from_corr,
        'ss_residual': ss_res,
        'ss_total': ss_tot,
        'mean_y_true': np.mean(y_true),
        'std_y_true': np.std(y_true),
        'mean_y_pred': np.mean(y_pred),
        'std_y_pred': np.std(y_pred),
        'correlation': correlation
    }

def main():
    # Load predictions (you'll need to specify the correct path)
    # This assumes you have saved predictions from validation or test
    pred_file = 'validation_results/validation_predictions.csv'
    
    if Path(pred_file).exists():
        df = pd.read_csv(pred_file)
        y_true = df['pkd'].values
        y_pred = df['pkd_predicted'].values
        
        if 'pkd_normalized_predicted' in df.columns:
            y_true_norm = (y_true - y_true.mean()) / y_true.std()
            y_pred_norm = df['pkd_normalized_predicted'].values
        else:
            y_true_norm = None
            y_pred_norm = None
    else:
        # Generate synthetic data to demonstrate the issue
        print("No prediction file found. Using synthetic data to demonstrate...")
        np.random.seed(42)
        
        # Create synthetic data with good RMSE but potentially poor R²
        n = 1000
        y_true = np.random.uniform(3, 12, n)  # pKd values
        
        # Add predictions with constant bias and some noise
        # This creates good RMSE but poor R² scenario
        bias = 2.0  # Constant offset
        noise = np.random.normal(0, 0.5, n)
        y_pred = np.ones_like(y_true) * y_true.mean() + noise  # Predictions around mean
        
        # Alternative: predictions with good correlation but different scale
        # y_pred = 0.5 * y_true + 3 + noise
        
        y_true_norm = (y_true - y_true.mean()) / y_true.std()
        y_pred_norm = (y_pred - y_pred.mean()) / y_pred.std()
    
    print("="*70)
    print("R² INVESTIGATION - Normalized vs Unnormalized")
    print("="*70)
    
    # Calculate metrics on original scale
    print("\n1. ORIGINAL pKd SCALE:")
    print("-"*40)
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    
    # Detailed R² calculation
    r2_details = calculate_r2_manual(y_true, y_pred)
    print(f"\nR² Calculation Details:")
    print(f"  Mean of y_true:  {r2_details['mean_y_true']:.4f}")
    print(f"  Std of y_true:   {r2_details['std_y_true']:.4f}")
    print(f"  Mean of y_pred:  {r2_details['mean_y_pred']:.4f}")
    print(f"  Std of y_pred:   {r2_details['std_y_pred']:.4f}")
    print(f"  Correlation:     {r2_details['correlation']:.4f}")
    print(f"  SS_residual:    {r2_details['ss_residual']:.2f}")
    print(f"  SS_total:       {r2_details['ss_total']:.2f}")
    print(f"  R² = 1 - SS_res/SS_tot = {r2_details['r2']:.4f}")
    print(f"  R² from corr²:  {r2_details['r2_from_correlation']:.4f}")
    
    # Check for common R² issues
    print("\n2. DIAGNOSTIC CHECKS:")
    print("-"*40)
    
    # Check 1: Constant predictions
    pred_variance = np.var(y_pred)
    if pred_variance < 0.01:
        print("⚠️  WARNING: Predictions have very low variance (nearly constant)")
        print(f"   Prediction variance: {pred_variance:.6f}")
    
    # Check 2: Scale mismatch
    scale_ratio = np.std(y_pred) / np.std(y_true)
    if scale_ratio < 0.5 or scale_ratio > 2.0:
        print("⚠️  WARNING: Large scale mismatch between predictions and targets")
        print(f"   Scale ratio (std_pred/std_true): {scale_ratio:.3f}")
    
    # Check 3: Systematic bias
    mean_error = np.mean(y_pred - y_true)
    if abs(mean_error) > 0.5:
        print("⚠️  WARNING: Large systematic bias in predictions")
        print(f"   Mean error: {mean_error:.3f}")
    
    # Check 4: R² vs RMSE consistency
    # For good R², RMSE should be much smaller than std of y_true
    rmse_ratio = rmse / np.std(y_true)
    expected_r2 = 1 - rmse_ratio**2
    print(f"\nExpected R² from RMSE/std ratio: {expected_r2:.4f}")
    if abs(expected_r2 - r2_details['r2']) > 0.1:
        print("⚠️  WARNING: R² calculation may have numerical issues")
    
    # Calculate on normalized scale if available
    if y_true_norm is not None and y_pred_norm is not None:
        print("\n3. NORMALIZED SCALE:")
        print("-"*40)
        r2_norm = calculate_r2_manual(y_true_norm, y_pred_norm)
        print(f"R² on normalized scale: {r2_norm['r2']:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Scatter plot
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')
    ax.plot([y_true.min(), y_true.max()], [y_true.mean(), y_true.mean()], 'g--', label='Mean prediction')
    ax.set_xlabel('True pKd')
    ax.set_ylabel('Predicted pKd')
    ax.set_title(f'Predictions vs True Values (R²={r2_details["r2"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residual plot
    ax = axes[0, 1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.axhline(y=mean_error, color='g', linestyle='--', label=f'Mean error: {mean_error:.3f}')
    ax.set_xlabel('True pKd')
    ax.set_ylabel('Residual (Pred - True)')
    ax.set_title('Residual Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distribution comparison
    ax = axes[1, 0]
    ax.hist(y_true, bins=30, alpha=0.5, label='True', density=True)
    ax.hist(y_pred, bins=30, alpha=0.5, label='Predicted', density=True)
    ax.set_xlabel('pKd')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    
    # Plot 4: Q-Q plot
    ax = axes[1, 1]
    quantiles = np.percentile(y_true, np.linspace(0, 100, 101))
    pred_quantiles = np.percentile(y_pred, np.linspace(0, 100, 101))
    ax.scatter(quantiles, pred_quantiles, alpha=0.6)
    ax.plot([quantiles.min(), quantiles.max()], [quantiles.min(), quantiles.max()], 'r--')
    ax.set_xlabel('True Quantiles')
    ax.set_ylabel('Predicted Quantiles')
    ax.set_title('Q-Q Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('r2_investigation.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: r2_investigation.png")
    
    # Explain the R² issue
    print("\n4. EXPLANATION:")
    print("-"*70)
    print("R² can be poor even with good RMSE when:")
    print("1. Predictions are clustered around the mean (low variance)")
    print("2. There's a systematic bias in predictions")
    print("3. The model captures relative differences but not absolute scale")
    print("4. The test set has different distribution than training set")
    print("\nR² measures proportion of variance explained, while RMSE measures")
    print("absolute prediction error. They can diverge when predictions don't")
    print("capture the full variance of the target distribution.")

if __name__ == '__main__':
    main()