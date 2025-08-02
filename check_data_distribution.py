#!/usr/bin/env python3
"""
Check data distribution between train and test sets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def analyze_dataset(df, name):
    """Analyze a dataset and return statistics."""
    stats = {
        'name': name,
        'num_samples': len(df),
        'pkd_mean': df['pkd'].mean(),
        'pkd_std': df['pkd'].std(),
        'pkd_min': df['pkd'].min(),
        'pkd_max': df['pkd'].max(),
        'pkd_median': df['pkd'].median(),
        'pkd_q1': df['pkd'].quantile(0.25),
        'pkd_q3': df['pkd'].quantile(0.75)
    }
    
    # Sequence length statistics
    seq_lengths = pd.concat([
        df['protein1_sequence'].str.len(),
        df['protein2_sequence'].str.len()
    ])
    
    stats['seq_mean_length'] = seq_lengths.mean()
    stats['seq_std_length'] = seq_lengths.std()
    stats['seq_min_length'] = seq_lengths.min()
    stats['seq_max_length'] = seq_lengths.max()
    
    return stats

def main():
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Load normalization parameters
    norm_params_path = Path('checkpoints/normalization_params.json')
    if norm_params_path.exists():
        with open(norm_params_path, 'r') as f:
            norm_params = json.load(f)
        print(f"\nNormalization parameters from training:")
        print(f"  Mean: {norm_params['mean']:.4f}")
        print(f"  Std:  {norm_params['std']:.4f}")
    
    # Analyze distributions
    train_stats = analyze_dataset(train_df, 'Train')
    test_stats = analyze_dataset(test_df, 'Test')
    
    # Print comparison
    print("\n" + "="*60)
    print("DATA DISTRIBUTION COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15} {'Difference':<15}")
    print("-"*60)
    
    metrics = ['num_samples', 'pkd_mean', 'pkd_std', 'pkd_min', 'pkd_max', 
               'pkd_median', 'seq_mean_length']
    
    for metric in metrics:
        train_val = train_stats[metric]
        test_val = test_stats[metric]
        
        if metric == 'num_samples':
            diff = f"{test_val/train_val:.1%} of train"
        elif 'length' in metric:
            diff = f"{test_val - train_val:.0f}"
        else:
            diff = f"{test_val - train_val:+.3f}"
            
        print(f"{metric:<20} {train_val:<15.3f} {test_val:<15.3f} {diff:<15}")
    
    # Check distribution shift
    print("\n" + "="*60)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("="*60)
    
    # Z-score of test mean using train distribution
    z_score = (test_stats['pkd_mean'] - train_stats['pkd_mean']) / (train_stats['pkd_std'] / np.sqrt(test_stats['num_samples']))
    print(f"\nZ-score of test mean: {z_score:.3f}")
    if abs(z_score) > 2:
        print("⚠️  WARNING: Significant distribution shift detected!")
    
    # Check if test values are in training range
    test_outside_range = ((test_df['pkd'] < train_stats['pkd_min']) | 
                         (test_df['pkd'] > train_stats['pkd_max'])).sum()
    print(f"\nTest samples outside training range: {test_outside_range} ({test_outside_range/len(test_df):.1%})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # pKd distributions
    ax = axes[0, 0]
    train_df['pkd'].hist(bins=30, alpha=0.6, label='Train', ax=ax, color='blue')
    test_df['pkd'].hist(bins=30, alpha=0.6, label='Test', ax=ax, color='red')
    ax.axvline(train_stats['pkd_mean'], color='blue', linestyle='--', label='Train mean')
    ax.axvline(test_stats['pkd_mean'], color='red', linestyle='--', label='Test mean')
    ax.set_xlabel('pKd')
    ax.set_ylabel('Count')
    ax.set_title('pKd Distribution')
    ax.legend()
    
    # Box plots
    ax = axes[0, 1]
    data_for_box = pd.DataFrame({
        'pKd': pd.concat([train_df['pkd'], test_df['pkd']]),
        'Dataset': ['Train']*len(train_df) + ['Test']*len(test_df)
    })
    sns.boxplot(data=data_for_box, x='Dataset', y='pKd', ax=ax)
    ax.set_title('pKd Box Plot Comparison')
    
    # Sequence length distributions
    ax = axes[1, 0]
    train_lengths = pd.concat([train_df['protein1_sequence'].str.len(), 
                              train_df['protein2_sequence'].str.len()])
    test_lengths = pd.concat([test_df['protein1_sequence'].str.len(),
                             test_df['protein2_sequence'].str.len()])
    
    train_lengths.hist(bins=30, alpha=0.6, label='Train', ax=ax, color='blue')
    test_lengths.hist(bins=30, alpha=0.6, label='Test', ax=ax, color='red')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Count')
    ax.set_title('Sequence Length Distribution')
    ax.legend()
    
    # Q-Q plot
    ax = axes[1, 1]
    train_quantiles = np.percentile(train_df['pkd'], np.arange(0, 101, 1))
    test_quantiles = np.percentile(test_df['pkd'], np.arange(0, 101, 1))
    ax.scatter(train_quantiles, test_quantiles, alpha=0.6)
    ax.plot([train_quantiles.min(), train_quantiles.max()], 
            [train_quantiles.min(), train_quantiles.max()], 'r--', lw=2)
    ax.set_xlabel('Train Quantiles')
    ax.set_ylabel('Test Quantiles')
    ax.set_title('Q-Q Plot: Test vs Train')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: data_distribution_comparison.png")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if abs(test_stats['pkd_mean'] - train_stats['pkd_mean']) > 0.5:
        print("• Consider retraining with a more representative training set")
        print("• The test set has a different pKd distribution than training")
    
    if test_outside_range > 0:
        print(f"• {test_outside_range} test samples have pKd values outside training range")
        print("• Model may extrapolate poorly for these samples")
    
    if abs(z_score) > 2:
        print("• Significant distribution shift detected")
        print("• Consider domain adaptation techniques")


if __name__ == '__main__':
    main()