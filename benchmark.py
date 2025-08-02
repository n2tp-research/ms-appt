#!/usr/bin/env python3
"""
Benchmark MS-APPT model performance
"""

import argparse
import torch
import time
import numpy as np
from pathlib import Path
import yaml
import logging
import pandas as pd
from tqdm import tqdm

from src.models import MS_APPT, ESM2EmbeddingExtractor
from src.data import ProteinDataPreprocessor, create_dataloaders
from src.training import MS_APPT_Trainer


def benchmark_inference(model, embedding_extractor, dataloader, device='cuda', use_amp=True, warmup_steps=10):
    """Benchmark inference performance."""
    model.eval()
    embedding_extractor.set_model_eval()
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup_steps:
                break
            
            embeddings1 = embedding_extractor.get_batch_embeddings(batch['protein1_sequences'])
            embeddings2 = embedding_extractor.get_batch_embeddings(batch['protein2_sequences'])
            
            # Pad embeddings
            max_len1 = max(e.shape[0] for e in embeddings1)
            max_len2 = max(e.shape[0] for e in embeddings2)
            
            batch_size = len(embeddings1)
            padded_embeddings1 = torch.zeros(batch_size, max_len1, embeddings1[0].shape[1])
            padded_embeddings2 = torch.zeros(batch_size, max_len2, embeddings2[0].shape[1])
            
            for j, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
                padded_embeddings1[j, :emb1.shape[0]] = emb1
                padded_embeddings2[j, :emb2.shape[0]] = emb2
            
            padded_embeddings1 = padded_embeddings1.to(device)
            padded_embeddings2 = padded_embeddings2.to(device)
            
            if use_amp:
                from torch.amp import autocast
                with autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(padded_embeddings1, padded_embeddings2,
                             batch['protein1_sequences'], batch['protein2_sequences'])
            else:
                _ = model(padded_embeddings1, padded_embeddings2,
                         batch['protein1_sequences'], batch['protein2_sequences'])
    
    # Benchmark
    print("\nBenchmarking...")
    torch.cuda.synchronize()
    start_events = []
    end_events = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Benchmarking"):
            embeddings1 = embedding_extractor.get_batch_embeddings(batch['protein1_sequences'])
            embeddings2 = embedding_extractor.get_batch_embeddings(batch['protein2_sequences'])
            
            # Pad embeddings
            max_len1 = max(e.shape[0] for e in embeddings1)
            max_len2 = max(e.shape[0] for e in embeddings2)
            
            batch_size = len(embeddings1)
            padded_embeddings1 = torch.zeros(batch_size, max_len1, embeddings1[0].shape[1])
            padded_embeddings2 = torch.zeros(batch_size, max_len2, embeddings2[0].shape[1])
            
            for j, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
                padded_embeddings1[j, :emb1.shape[0]] = emb1
                padded_embeddings2[j, :emb2.shape[0]] = emb2
            
            padded_embeddings1 = padded_embeddings1.to(device)
            padded_embeddings2 = padded_embeddings2.to(device)
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            if use_amp:
                from torch.amp import autocast
                with autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(padded_embeddings1, padded_embeddings2,
                             batch['protein1_sequences'], batch['protein2_sequences'])
            else:
                _ = model(padded_embeddings1, padded_embeddings2,
                         batch['protein1_sequences'], batch['protein2_sequences'])
            end.record()
            
            start_events.append(start)
            end_events.append(end)
    
    torch.cuda.synchronize()
    
    # Calculate statistics
    times = []
    for start, end in zip(start_events, end_events):
        times.append(start.elapsed_time(end))
    
    times = np.array(times)
    total_samples = len(dataloader.dataset)
    total_time = np.sum(times) / 1000.0  # Convert to seconds
    
    return {
        'total_samples': total_samples,
        'total_time': total_time,
        'avg_time_per_batch': np.mean(times),
        'std_time_per_batch': np.std(times),
        'min_time_per_batch': np.min(times),
        'max_time_per_batch': np.max(times),
        'throughput': total_samples / total_time,
        'batch_size': dataloader.batch_sampler.batch_size if hasattr(dataloader, 'batch_sampler') else dataloader.batch_size
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark MS-APPT model')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file for benchmarking')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[16, 32, 64],
                       help='Batch sizes to test')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to use for benchmarking')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    # Limit samples
    if len(df) > args.num_samples:
        df = df.sample(n=args.num_samples, random_state=42)
    
    # Preprocess
    preprocessor = ProteinDataPreprocessor(config)
    checkpoint_dir = Path(args.checkpoint).parent
    norm_params_path = checkpoint_dir / 'normalization_params.json'
    if norm_params_path.exists():
        preprocessor.load_normalization_params(str(norm_params_path))
    
    # Initialize models
    logger.info("Loading models...")
    embedding_extractor = ESM2EmbeddingExtractor(
        model_name=config['model']['encoder']['model_name'],
        cache_dir=config['data']['cache_dir'],
        device=args.device
    )
    
    model = MS_APPT(config).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Benchmark different batch sizes
    results = {}
    use_amp = not args.no_amp and config['training']['mixed_precision']
    
    print(f"\nBenchmarking with {len(df)} samples")
    print(f"Device: {args.device}")
    print(f"Mixed Precision: {use_amp}")
    print("="*70)
    
    for batch_size in args.batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Update config
        config['inference']['batch_size'] = batch_size
        
        # Create dataloader
        train_loader, _ = create_dataloaders(df, df, config)
        
        # Benchmark
        stats = benchmark_inference(
            model, embedding_extractor, train_loader,
            device=args.device, use_amp=use_amp
        )
        
        results[batch_size] = stats
        
        print(f"  Total time: {stats['total_time']:.2f}s")
        print(f"  Throughput: {stats['throughput']:.1f} samples/s")
        print(f"  Avg batch time: {stats['avg_time_per_batch']:.2f}ms ± {stats['std_time_per_batch']:.2f}ms")
        print(f"  Min/Max batch time: {stats['min_time_per_batch']:.2f}ms / {stats['max_time_per_batch']:.2f}ms")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Batch Size':<12} {'Throughput (samples/s)':<25} {'Avg Time (ms/batch)':<20}")
    print("-"*70)
    
    for batch_size, stats in results.items():
        print(f"{batch_size:<12} {stats['throughput']:<25.1f} {stats['avg_time_per_batch']:<20.2f}")
    
    # Find optimal batch size
    best_batch_size = max(results.keys(), key=lambda k: results[k]['throughput'])
    print(f"\nOptimal batch size: {best_batch_size} ({results[best_batch_size]['throughput']:.1f} samples/s)")


if __name__ == '__main__':
    main()