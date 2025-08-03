#!/usr/bin/env python3
"""
Diagnostic script to check prediction and denormalization flow
"""

import numpy as np
import json
from pathlib import Path

def check_normalization_params(checkpoint_dir):
    """Check if normalization parameters exist and are reasonable."""
    norm_params_path = Path(checkpoint_dir) / 'normalization_params.json'
    
    if not norm_params_path.exists():
        print("❌ ERROR: normalization_params.json not found!")
        print(f"   Expected at: {norm_params_path}")
        return None
    
    with open(norm_params_path, 'r') as f:
        params = json.load(f)
    
    print("✓ Normalization parameters found:")
    print(f"  Mean: {params['mean']:.4f}")
    print(f"  Std:  {params['std']:.4f}")
    
    # Check if parameters are reasonable for pKd values
    if params['mean'] < 0 or params['mean'] > 20:
        print("⚠️  WARNING: Mean seems unusual for pKd values (expected 0-20)")
    if params['std'] < 0.1 or params['std'] > 10:
        print("⚠️  WARNING: Std seems unusual for pKd values")
    
    return params

def simulate_prediction_flow(norm_params):
    """Simulate the prediction and denormalization flow."""
    print("\n" + "="*60)
    print("SIMULATING PREDICTION FLOW")
    print("="*60)
    
    # Simulate some normalized predictions from model
    # Model outputs should be centered around 0 with std ~1 if trained on normalized data
    normalized_predictions = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    
    print("\n1. Model outputs (normalized):")
    print(f"   {normalized_predictions}")
    
    # Denormalize
    denormalized = normalized_predictions * norm_params['std'] + norm_params['mean']
    
    print("\n2. After denormalization:")
    print(f"   {denormalized}")
    print(f"   Range: [{denormalized.min():.2f}, {denormalized.max():.2f}]")
    
    # Check if denormalized values are in reasonable pKd range
    if np.any(denormalized < 0) or np.any(denormalized > 20):
        print("⚠️  WARNING: Some denormalized values are outside typical pKd range (0-20)")
    
    # Show the inverse operation
    print("\n3. Verification (should match original normalized values):")
    renormalized = (denormalized - norm_params['mean']) / norm_params['std']
    print(f"   {renormalized}")
    print(f"   Max difference: {np.max(np.abs(renormalized - normalized_predictions)):.2e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check prediction flow')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory containing normalization_params.json')
    args = parser.parse_args()
    
    print("PREDICTION FLOW DIAGNOSTIC")
    print("="*60)
    
    # Check normalization parameters
    norm_params = check_normalization_params(args.checkpoint_dir)
    
    if norm_params:
        # Simulate prediction flow
        simulate_prediction_flow(norm_params)
        
        # Show expected pKd range after denormalization
        print("\n" + "="*60)
        print("EXPECTED pKd RANGES")
        print("="*60)
        
        # For normalized values in [-3, 3] range
        norm_range = np.array([-3, -2, -1, 0, 1, 2, 3])
        pkd_range = norm_range * norm_params['std'] + norm_params['mean']
        
        print("\nNormalized → pKd mapping:")
        for n, p in zip(norm_range, pkd_range):
            print(f"  {n:+.1f} → {p:.2f}")
        
        print(f"\nTypical pKd range after denormalization:")
        print(f"  [{pkd_range.min():.2f}, {pkd_range.max():.2f}]")
        
        # Recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print("1. Ensure model was trained with these same normalization parameters")
        print("2. Check that model outputs are roughly centered around 0")
        print("3. Verify test data has similar pKd distribution as training data")
        print("4. If predictions seem wrong, check:")
        print("   - Model checkpoint matches normalization parameters")
        print("   - No mismatch between training and inference preprocessing")

if __name__ == '__main__':
    main()