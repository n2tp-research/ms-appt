#!/usr/bin/env python3
"""
Run comprehensive validation benchmark on the best trained model.
"""

import argparse
import logging
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime

from src.models import MS_APPT, ESM2EmbeddingExtractor
from src.data import ProteinDataPreprocessor, create_dataloaders
from src.evaluation import calculate_all_metrics, print_metrics_summary, create_performance_report
from src.visualization import create_all_visualizations
from src.utils import save_json_safe


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_model_and_data(config_path: str, checkpoint_path: str, data_path: str, 
                       device: str = 'cuda', val_split: float = 0.1):
    """Load model, preprocessor, and prepare data."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = ProteinDataPreprocessor(config)
    
    # Load normalization parameters
    checkpoint_dir = Path(checkpoint_path).parent
    norm_params_path = checkpoint_dir / 'normalization_params.json'
    if norm_params_path.exists():
        preprocessor.load_normalization_params(str(norm_params_path))
    else:
        logging.warning("Normalization parameters not found. Will fit on data.")
    
    # Load and preprocess data
    logging.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df = preprocessor.preprocess_dataset(df, fit_normalization=False)
    
    # Split data
    _, val_df = preprocessor.train_val_split(df, val_split=val_split, 
                                           random_state=config['data']['random_seed'])
    
    # Create dataloader
    _, val_loader = create_dataloaders(val_df, val_df, config)
    
    # Initialize models
    logging.info("Loading ESM-2 embedding extractor...")
    embedding_extractor = ESM2EmbeddingExtractor(
        model_name=config['model']['encoder']['model_name'],
        cache_dir=config['data']['cache_dir'],
        device=device
    )
    
    logging.info("Loading MS-APPT model...")
    model = MS_APPT(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logging.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        logging.info("Training metrics at checkpoint:")
        for k, v in checkpoint['metrics'].items():
            if isinstance(v, (int, float)):
                logging.info(f"  {k}: {v:.4f}")
    
    return model, embedding_extractor, preprocessor, val_loader, val_df, config


def run_validation(model, embedding_extractor, val_loader, device='cuda'):
    """Run validation and return predictions."""
    model.eval()
    embedding_extractor.set_model_eval()
    
    all_predictions = []
    all_targets = []
    all_indices = []
    
    logging.info("Running validation...")
    from tqdm import tqdm
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Get embeddings
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
            
            # Get predictions
            predictions = model(
                padded_embeddings1, padded_embeddings2,
                batch['protein1_sequences'], batch['protein2_sequences']
            )
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch['pkd_normalized'].numpy())
            all_indices.extend(batch['indices'].numpy())
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_indices = np.array(all_indices)
    
    # Sort by original indices
    sorted_idx = np.argsort(all_indices)
    all_predictions = all_predictions[sorted_idx]
    all_targets = all_targets[sorted_idx]
    
    return all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description='Run validation benchmark on MS-APPT model')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='train.csv',
                       help='Path to data file (will use validation split)')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Directory for output files')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save individual predictions to CSV')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model and data
    model, embedding_extractor, preprocessor, val_loader, val_df, config = load_model_and_data(
        args.config, args.checkpoint, args.data, args.device
    )
    
    # Run validation
    predictions_norm, targets_norm = run_validation(
        model, embedding_extractor, val_loader, args.device
    )
    
    # Denormalize predictions
    predictions = preprocessor.denormalize_pkd(predictions_norm)
    targets = preprocessor.denormalize_pkd(targets_norm)
    
    # Calculate comprehensive metrics
    logger.info("\nCalculating metrics...")
    metrics = calculate_all_metrics(targets, predictions)
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION BENCHMARK RESULTS")
    print("="*70)
    print(f"Dataset: {args.data}")
    print(f"Model: {args.checkpoint}")
    print(f"Validation samples: {len(val_df)}")
    print("="*70)
    
    # Print all metrics
    print("\nOverall Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            if 'p' in metric and 'p_' not in metric:  # p-values
                print(f"  {metric}: {value:.2e}")
            else:
                print(f"  {metric}: {value:.4f}")
    
    # Performance by pKd range
    print("\nPerformance by pKd Range:")
    error_by_range = analyze_errors_by_range(targets, predictions)
    print(f"{'Range':<15} {'Count':<8} {'RMSE':<8} {'MAE':<8} {'R²':<8}")
    print("-" * 50)
    for range_name, stats in error_by_range.items():
        print(f"{range_name:<15} {stats['count']:<8} {stats['rmse']:<8.3f} "
              f"{stats['mae']:<8.3f} {stats.get('r2', 0):<8.3f}")
    
    # Save detailed results
    results = {
        'timestamp': datetime.now().isoformat(),
        'config_path': args.config,
        'checkpoint_path': args.checkpoint,
        'data_path': args.data,
        'num_samples': len(val_df),
        'metrics': metrics,
        'error_by_range': error_by_range
    }
    
    save_json_safe(results, output_dir / 'validation_results.json', indent=2)
    logger.info(f"\nSaved detailed results to {output_dir / 'validation_results.json'}")
    
    # Save predictions if requested
    if args.save_predictions:
        pred_df = val_df.copy()
        pred_df['pkd_predicted'] = predictions
        pred_df['pkd_normalized_predicted'] = predictions_norm
        pred_df['error'] = predictions - targets
        pred_df['abs_error'] = np.abs(pred_df['error'])
        
        pred_df.to_csv(output_dir / 'validation_predictions.csv', index=False)
        logger.info(f"Saved predictions to {output_dir / 'validation_predictions.csv'}")
    
    # Create visualizations if requested
    if args.create_plots:
        logger.info("\nCreating visualizations...")
        
        # Get sequence lengths for stratified analysis
        seq_lengths1 = val_df['protein1_sequence'].str.len().values
        seq_lengths2 = val_df['protein2_sequence'].str.len().values
        
        # Create performance report
        performance_report = create_performance_report(
            targets, predictions,
            sequence_lengths=(seq_lengths1, seq_lengths2)
        )
        
        # Create plots
        saved_plots = create_all_visualizations(
            targets, predictions,
            output_dir=output_dir,
            prefix="validation_",
            performance_dict=performance_report,
            config=config
        )
        
        logger.info(f"Saved visualizations to {output_dir}")
    
    print("\n" + "="*70)
    print("Validation benchmark complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


def analyze_errors_by_range(y_true: np.ndarray, y_pred: np.ndarray, num_bins: int = 5):
    """Analyze errors by pKd range."""
    from src.evaluation.metrics import analyze_errors_by_range as analyze_fn
    return analyze_fn(y_true, y_pred, num_bins)


if __name__ == '__main__':
    main()