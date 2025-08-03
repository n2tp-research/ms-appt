#!/usr/bin/env python3
"""
Verify prediction and denormalization process
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    # Step 1: Load normalization parameters
    norm_path = Path('checkpoints/normalization_params.json')
    if not norm_path.exists():
        print("ERROR: normalization_params.json not found!")
        return
    
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)
    
    print("="*60)
    print("PREDICTION FLOW VERIFICATION")
    print("="*60)
    
    print("\n1. NORMALIZATION PARAMETERS FROM TRAINING:")
    print(f"   Mean: {norm_params['mean']:.4f}")
    print(f"   Std:  {norm_params['std']:.4f}")
    
    # Step 2: Show what model sees and outputs
    print("\n2. MODEL PROCESS:")
    print("   - Model was trained on NORMALIZED pKd values")
    print("   - Model outputs NORMALIZED predictions")
    print("   - These predictions should be centered around 0 with std ≈ 1")
    
    # Step 3: Demonstrate denormalization
    print("\n3. DENORMALIZATION FORMULA:")
    print("   pKd = normalized_pred * std + mean")
    print(f"   pKd = normalized_pred * {norm_params['std']:.4f} + {norm_params['mean']:.4f}")
    
    # Step 4: Example predictions
    print("\n4. EXAMPLE PREDICTIONS:")
    example_normalized = np.array([-2, -1, 0, 1, 2])
    example_denormalized = example_normalized * norm_params['std'] + norm_params['mean']
    
    print("   Normalized (model output) → Denormalized (actual pKd)")
    for n, d in zip(example_normalized, example_denormalized):
        print(f"   {n:+6.2f} → {d:6.2f}")
    
    # Step 5: Expected pKd range
    print(f"\n5. EXPECTED pKd RANGE:")
    print(f"   For normalized values in [-3, +3]:")
    min_pkd = -3 * norm_params['std'] + norm_params['mean']
    max_pkd = +3 * norm_params['std'] + norm_params['mean']
    print(f"   pKd range: [{min_pkd:.2f}, {max_pkd:.2f}]")
    
    # Step 6: Common issues
    print("\n6. COMMON ISSUES TO CHECK:")
    print("   ✓ Model checkpoint matches normalization parameters")
    print("   ✓ Test data has similar distribution to training data")
    print("   ✓ No accidental double normalization/denormalization")
    print("   ✓ Metrics calculated on original pKd scale (not normalized)")
    
    # Step 7: How metrics should be calculated
    print("\n7. CORRECT METRIC CALCULATION:")
    print("   1. y_true = df['pkd'].values  # Original pKd values")
    print("   2. y_pred = denormalized predictions")
    print("   3. Calculate MSE, RMSE, R² on these original scale values")
    
    print("\n" + "="*60)
    print("If predictions seem wrong, check:")
    print("1. Run: python check_data_distribution.py")
    print("   → This shows if test distribution differs from training")
    print("2. Check model predictions are roughly N(0,1) distributed")
    print("3. Verify no preprocessing differences between train/test")

if __name__ == '__main__':
    main()