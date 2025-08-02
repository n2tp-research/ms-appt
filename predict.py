#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
import torch

from src.inference import MS_APPT_Predictor


def main():
    parser = argparse.ArgumentParser(description='MS-APPT Inference Script')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to config.yml file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output CSV file')
    parser.add_argument('--has-labels', action='store_true',
                       help='Whether input file contains pkd labels')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    logger.info(f"Using device: {args.device}")
    
    predictor = MS_APPT_Predictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    results = predictor.predict_file(
        input_path=args.input,
        output_path=args.output,
        has_labels=args.has_labels
    )
    
    logger.info(f"Predictions saved to: {args.output}")
    logger.info(f"Processed {results['num_samples']} samples")
    
    if 'performance' in results:
        logger.info("\nPerformance Summary:")
        overall_metrics = results['performance']['overall_metrics']
        logger.info(f"  RMSE: {overall_metrics['rmse']:.4f}")
        logger.info(f"  MAE: {overall_metrics['mae']:.4f}")
        logger.info(f"  Pearson r: {overall_metrics['pearson_r']:.4f}")
        logger.info(f"  Spearman ρ: {overall_metrics['spearman_r']:.4f}")


if __name__ == '__main__':
    main()