#!/usr/bin/env python3
"""
Simple and clear prediction script for MS-APPT
"""

import torch
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
import logging
from tqdm import tqdm

from src.models import MS_APPT, ESM2EmbeddingExtractor
from src.data import ProteinDataPreprocessor
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleProteinDataset(Dataset):
    """Simple dataset that just returns sequences."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'protein1_sequence': row['protein1_sequence'],
            'protein2_sequence': row['protein2_sequence'],
            'index': idx
        }


def collate_fn(batch):
    """Simple collate function."""
    return {
        'protein1_sequences': [item['protein1_sequence'] for item in batch],
        'protein2_sequences': [item['protein2_sequence'] for item in batch],
        'indices': torch.tensor([item['index'] for item in batch])
    }


def predict_test_data(test_csv_path, checkpoint_path, config_path, device='cuda'):
    """
    Simple prediction function:
    1. Load test data (with pkd values for evaluation)
    2. Clean sequences
    3. Get ESM embeddings (on-the-fly or cached)
    4. Run model → get normalized predictions
    5. Denormalize → get actual pKd predictions
    6. Compare with true pKd values
    """
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load normalization parameters from training
    checkpoint_dir = Path(checkpoint_path).parent
    norm_params_path = checkpoint_dir / 'normalization_params.json'
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    
    logger.info(f"Normalization params from training: mean={norm_params['mean']:.4f}, std={norm_params['std']:.4f}")
    
    # Load test data
    logger.info(f"Loading test data from {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Check what columns we have
    logger.info(f"Test data columns: {list(test_df.columns)}")
    has_true_pkd = 'pkd' in test_df.columns
    
    # Clean sequences (but DON'T normalize pkd values!)
    preprocessor = ProteinDataPreprocessor(config)
    test_df['protein1_sequence'] = test_df['protein1_sequence'].apply(preprocessor.clean_sequence)
    test_df['protein2_sequence'] = test_df['protein2_sequence'].apply(preprocessor.clean_sequence)
    
    # Validate sequences
    valid_mask = (
        test_df['protein1_sequence'].apply(preprocessor.validate_sequence) &
        test_df['protein2_sequence'].apply(preprocessor.validate_sequence)
    )
    
    if not valid_mask.all():
        logger.warning(f"Removing {(~valid_mask).sum()} invalid sequences")
        test_df = test_df[valid_mask].copy()
    
    # Create simple dataset and dataloader
    dataset = SimpleProteinDataset(test_df)
    dataloader = DataLoader(
        dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize models
    logger.info("Loading ESM-2 embedding extractor...")
    embedding_extractor = ESM2EmbeddingExtractor(
        model_name=config['model']['encoder']['model_name'],
        cache_dir=config['data']['cache_dir'],
        device=device
    )
    
    logger.info("Loading MS-APPT model...")
    model = MS_APPT(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Make predictions
    logger.info("Making predictions...")
    all_predictions_normalized = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Get embeddings (on-the-fly or from cache)
            embeddings1 = embedding_extractor.get_batch_embeddings(batch['protein1_sequences'])
            embeddings2 = embedding_extractor.get_batch_embeddings(batch['protein2_sequences'])
            
            # Pad embeddings
            max_len1 = max(e.shape[0] for e in embeddings1)
            max_len2 = max(e.shape[0] for e in embeddings2)
            
            batch_size = len(embeddings1)
            padded_embeddings1 = torch.zeros(batch_size, max_len1, embeddings1[0].shape[1])
            padded_embeddings2 = torch.zeros(batch_size, max_len2, embeddings2[0].shape[1])
            
            for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
                padded_embeddings1[i, :emb1.shape[0]] = emb1
                padded_embeddings2[i, :emb2.shape[0]] = emb2
            
            padded_embeddings1 = padded_embeddings1.to(device)
            padded_embeddings2 = padded_embeddings2.to(device)
            
            # Run model - get NORMALIZED predictions
            predictions_normalized = model(
                padded_embeddings1, padded_embeddings2,
                batch['protein1_sequences'], batch['protein2_sequences']
            )
            
            all_predictions_normalized.extend(predictions_normalized.cpu().numpy())
            all_indices.extend(batch['indices'].cpu().numpy())
    
    # Sort predictions by original index
    predictions_normalized = np.array(all_predictions_normalized)
    indices = np.array(all_indices)
    sorted_idx = np.argsort(indices)
    predictions_normalized = predictions_normalized[sorted_idx]
    
    # DENORMALIZE predictions
    predictions_pkd = predictions_normalized * norm_params['std'] + norm_params['mean']
    
    logger.info(f"\nPrediction statistics:")
    logger.info(f"  Normalized: mean={predictions_normalized.mean():.3f}, std={predictions_normalized.std():.3f}")
    logger.info(f"  pKd scale: mean={predictions_pkd.mean():.3f}, std={predictions_pkd.std():.3f}")
    logger.info(f"  pKd range: [{predictions_pkd.min():.2f}, {predictions_pkd.max():.2f}]")
    
    # Save predictions
    results_df = test_df.copy()
    results_df['pkd_predicted'] = predictions_pkd
    
    # Calculate metrics if we have true values
    if has_true_pkd:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import pearsonr, spearmanr
        
        y_true = test_df['pkd'].values
        y_pred = predictions_pkd
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  MSE:  {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
        logger.info(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")
        
        # Add error column
        results_df['pkd_actual'] = y_true
        results_df['error'] = y_pred - y_true
        
        logger.info(f"\nTrue pKd statistics:")
        logger.info(f"  Mean: {y_true.mean():.3f}, std: {y_true.std():.3f}")
        logger.info(f"  Range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    
    return results_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Simple MS-APPT prediction')
    parser.add_argument('--test', type=str, default='test.csv',
                       help='Test CSV file with protein sequences')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Model checkpoint path')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Config file path')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run predictions
    results_df = predict_test_data(
        args.test,
        args.checkpoint,
        args.config,
        args.device
    )
    
    # Save results
    output_columns = ['protein1_sequence', 'protein2_sequence', 'pkd_predicted']
    if 'pkd_actual' in results_df.columns:
        output_columns.extend(['pkd_actual', 'error'])
    
    results_df[output_columns].to_csv(args.output, index=False)
    logger.info(f"\nSaved predictions to: {args.output}")


if __name__ == '__main__':
    main()