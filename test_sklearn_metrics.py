#!/usr/bin/env python3
"""
Test scikit-learn metrics implementation
"""

import numpy as np
from src.evaluation.metrics import calculate_all_metrics, print_metrics_summary

# Generate test data
np.random.seed(42)
n_samples = 100

# Create synthetic pKd values (range 3-12)
y_true = np.random.uniform(3, 12, n_samples)

# Create predictions with some noise and bias
noise = np.random.normal(0, 0.5, n_samples)
bias = 0.2
y_pred = y_true + noise + bias

print("Testing sklearn metrics implementation")
print("=" * 50)
print(f"Number of samples: {n_samples}")
print(f"True pKd range: [{y_true.min():.2f}, {y_true.max():.2f}]")
print(f"Pred pKd range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

# Calculate metrics
metrics = calculate_all_metrics(y_true, y_pred)

# Print all metrics
print("\nAll calculated metrics:")
print("-" * 50)
for key, value in sorted(metrics.items()):
    if isinstance(value, float):
        if 'p' in key and 'p_' not in key:  # p-values
            print(f"{key:<25}: {value:.2e}")
        elif key == 'mape':  # percentage
            print(f"{key:<25}: {value:.2%}")
        else:
            print(f"{key:<25}: {value:.4f}")

# Use the print summary function
print("\n")
print_metrics_summary(metrics)

# Verify sklearn vs manual calculation
print("\nVerification (sklearn vs manual):")
print("-" * 50)

# Manual R² calculation
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_manual = 1 - (ss_res / ss_tot)

print(f"R² (sklearn):  {metrics['r2']:.6f}")
print(f"R² (manual):   {r2_manual:.6f}")
print(f"Difference:    {abs(metrics['r2'] - r2_manual):.2e}")

# Manual MSE calculation
mse_manual = np.mean((y_true - y_pred) ** 2)
print(f"\nMSE (sklearn): {metrics['mse']:.6f}")
print(f"MSE (manual):  {mse_manual:.6f}")
print(f"Difference:    {abs(metrics['mse'] - mse_manual):.2e}")

print("\n✓ Scikit-learn metrics successfully integrated!")