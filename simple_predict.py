#!/usr/bin/env python3
"""
Simple prediction script with explicit denormalization
"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_normalization_params(checkpoint_dir):
    """Load normalization parameters from training."""
    norm_path = Path(checkpoint_dir) / 'normalization_params.json'
    with open(norm_path, 'r') as f:
        return json.load(f)

def predict_and_denormalize(model, dataloader, norm_params, device='cuda'):
    """Make predictions and properly denormalize them."""
    model.eval()
    
    all_predictions_normalized = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get embeddings and make predictions
            # ... (model forward pass) ...
            # predictions = model(embeddings1, embeddings2, seq1, seq2)
            
            # Collect NORMALIZED predictions from model
            # all_predictions_normalized.extend(predictions.cpu().numpy())
            pass
    
    # Convert to numpy array
    predictions_normalized = np.array(all_predictions_normalized)
    
    # DENORMALIZE: convert from normalized to actual pKd scale
    predictions_pkd = predictions_normalized * norm_params['std'] + norm_params['mean']
    
    return predictions_normalized, predictions_pkd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Dir with norm params')
    parser.add_argument('--has-labels', action='store_true', help='Input has pKd labels')
    args = parser.parse_args()
    
    # Load normalization parameters
    norm_params = load_normalization_params(args.checkpoint_dir)
    logger.info(f"Loaded normalization params: mean={norm_params['mean']:.4f}, std={norm_params['std']:.4f}")
    
    # Load data
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} samples")
    
    # Make predictions (simplified - actual implementation would use MS_APPT_Predictor)
    # predictions_normalized, predictions_pkd = predict_and_denormalize(model, dataloader, norm_params)
    
    # For demonstration, simulate some predictions
    n = len(df)
    predictions_normalized = np.random.randn(n)  # Simulated normalized predictions
    predictions_pkd = predictions_normalized * norm_params['std'] + norm_params['mean']
    
    # Save results
    results_df = df.copy()
    results_df['pkd_predicted'] = predictions_pkd
    results_df['pkd_normalized_predicted'] = predictions_normalized
    
    # If we have labels, calculate metrics on ORIGINAL scale
    if args.has_labels and 'pkd' in df.columns:
        from sklearn.metrics import mean_squared_error, r2_score
        
        y_true = df['pkd'].values  # Original pKd values
        y_pred = predictions_pkd    # Denormalized predictions
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        logger.info("\nMetrics (on original pKd scale):")
        logger.info(f"  MSE:  {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        
        # Show distribution of predictions
        logger.info("\nPrediction Statistics:")
        logger.info(f"  Normalized predictions: mean={predictions_normalized.mean():.3f}, std={predictions_normalized.std():.3f}")
        logger.info(f"  pKd predictions: mean={predictions_pkd.mean():.3f}, std={predictions_pkd.std():.3f}")
        logger.info(f"  pKd predictions range: [{predictions_pkd.min():.2f}, {predictions_pkd.max():.2f}]")
        
        # Compare with true values
        logger.info(f"\nTrue pKd values:")
        logger.info(f"  Mean: {y_true.mean():.3f}, std: {y_true.std():.3f}")
        logger.info(f"  Range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    
    # Save output
    output_columns = ['protein1_sequence', 'protein2_sequence', 'pkd_predicted']
    if args.has_labels and 'pkd' in df.columns:
        output_columns.append('pkd')
        results_df['error'] = results_df['pkd_predicted'] - results_df['pkd']
        output_columns.append('error')
    
    results_df[output_columns].to_csv(args.output, index=False)
    logger.info(f"\nSaved predictions to: {args.output}")

if __name__ == '__main__':
    main()