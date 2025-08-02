#!/usr/bin/env python3
import argparse
import logging
import yaml
import torch
import pandas as pd
from pathlib import Path
import json
import sys
import os
import random
import numpy as np
from datetime import datetime

from src.models import MS_APPT, ESM2EmbeddingExtractor
from src.data import ProteinDataPreprocessor, create_dataloaders
from src.training import MS_APPT_Trainer
from src.evaluation import print_metrics_summary, create_performance_report
from src.visualization import create_all_visualizations


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(log_dir: str, log_level: str = 'INFO'):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def main():
    parser = argparse.ArgumentParser(description='Train MS-APPT model')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to config file')
    parser.add_argument('--train-data', type=str, default='train.csv',
                       help='Path to training data CSV')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Overrides config.')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs. Overrides config.')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size. Overrides config.')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate. Overrides config.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.device:
        config['hardware']['device'] = args.device
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['optimizer']['learning_rate'] = args.learning_rate
    
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
        config['hardware']['device'] = 'cpu'
    
    set_seed(
        args.seed,
        deterministic=config['reproducibility']['deterministic'],
        benchmark=config['reproducibility']['benchmark']
    )
    config['reproducibility']['seed'] = args.seed
    
    log_file = setup_logging(config['logging']['log_dir'], config['logging']['log_level'])
    logger = logging.getLogger(__name__)
    logger.info(f"Starting MS-APPT training")
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Device: {device}")
    
    logger.info("Loading and preprocessing data...")
    preprocessor = ProteinDataPreprocessor(config)
    
    df = pd.read_csv(args.train_data)
    logger.info(f"Loaded {len(df)} samples from {args.train_data}")
    
    df = preprocessor.preprocess_dataset(df, fit_normalization=True)
    
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    preprocessor.save_normalization_params(str(checkpoint_dir / 'normalization_params.json'))
    
    train_df, val_df = preprocessor.train_val_split(
        df, 
        val_split=config['data']['validation_split'],
        random_state=config['data']['random_seed']
    )
    
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(train_df, val_df, config)
    
    logger.info("Initializing ESM-2 embedding extractor...")
    embedding_extractor = ESM2EmbeddingExtractor(
        model_name=config['model']['encoder']['model_name'],
        cache_dir=config['data']['cache_dir'],
        device=device
    )
    
    cache_stats = embedding_extractor.get_cache_stats()
    logger.info(f"Embedding cache stats: {cache_stats}")
    
    logger.info("Initializing MS-APPT model...")
    model = MS_APPT(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    logger.info("Initializing trainer...")
    trainer = MS_APPT_Trainer(model, embedding_extractor, config, device=device)
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    with open(checkpoint_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f)
    
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['max_epochs']
    )
    
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Loading best model for final evaluation...")
    best_checkpoint = checkpoint_dir / 'best_model.pt'
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info("Running final validation...")
        final_metrics = trainer.validate(val_loader)
        
        logger.info("\nFinal Validation Metrics:")
        print_metrics_summary(final_metrics)
        
        with torch.no_grad():
            all_predictions = []
            all_targets = []
            
            for batch in val_loader:
                protein1_embeddings, protein2_embeddings = trainer._get_embeddings_batch(
                    batch['protein1_sequences'], batch['protein2_sequences']
                )
                
                predictions = model(
                    protein1_embeddings, protein2_embeddings,
                    batch['protein1_sequences'], batch['protein2_sequences']
                )
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['pkd_normalized'].numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        y_true_denorm = preprocessor.denormalize_pkd(all_targets)
        y_pred_denorm = preprocessor.denormalize_pkd(all_predictions)
        
        seq_lengths1 = val_df['protein1_sequence'].str.len().values
        seq_lengths2 = val_df['protein2_sequence'].str.len().values
        
        performance_report = create_performance_report(
            y_true_denorm, y_pred_denorm,
            sequence_lengths=(seq_lengths1, seq_lengths2)
        )
        
        with open(checkpoint_dir / 'validation_performance.json', 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        logger.info("Creating visualizations...")
        vis_dir = Path(config['visualization']['output_dir'])
        saved_plots = create_all_visualizations(
            y_true_denorm, y_pred_denorm,
            output_dir=vis_dir,
            prefix="validation_",
            performance_dict=performance_report,
            history=history,
            config=config
        )
        
        logger.info(f"Saved visualizations: {saved_plots}")
    
    logger.info("Training completed successfully!")
    logger.info(f"Model checkpoints saved to: {checkpoint_dir}")
    logger.info(f"Best model: {best_checkpoint}")


if __name__ == '__main__':
    main()