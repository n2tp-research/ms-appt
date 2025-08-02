import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
import logging
from sklearn.model_selection import train_test_split
import hashlib

logger = logging.getLogger(__name__)


class ProteinDataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config['data']['preprocessing']
        self.valid_amino_acids = set(self.config['valid_amino_acids'])
        self.min_length = self.config['min_length']
        self.max_length = self.config['max_length']
        self.normalization_params = None
        
    def validate_sequence(self, sequence: str) -> bool:
        if not isinstance(sequence, str):
            return False
            
        sequence = sequence.upper().strip()
        
        if not sequence:
            return False
            
        if len(sequence) < self.min_length or len(sequence) > self.max_length:
            return False
            
        return all(aa in self.valid_amino_acids for aa in sequence)
    
    def clean_sequence(self, sequence: str) -> str:
        if not isinstance(sequence, str):
            return ""
        return sequence.upper().strip()
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        def get_pair_hash(row):
            seqs = sorted([row['protein1_sequence'], row['protein2_sequence']])
            return hashlib.md5(''.join(seqs).encode()).hexdigest()
        
        df['pair_hash'] = df.apply(get_pair_hash, axis=1)
        
        duplicates = df.groupby('pair_hash').agg({
            'pkd': ['count', 'median']
        })
        
        keep_hashes = duplicates[duplicates[('pkd', 'count')] == 1].index.tolist()
        median_hashes = duplicates[duplicates[('pkd', 'count')] > 1].index.tolist()
        
        result_df = df[df['pair_hash'].isin(keep_hashes)].copy()
        
        for hash_val in median_hashes:
            median_pkd = duplicates.loc[hash_val, ('pkd', 'median')]
            first_row = df[df['pair_hash'] == hash_val].iloc[0].copy()
            first_row['pkd'] = median_pkd
            result_df = pd.concat([result_df, pd.DataFrame([first_row])], ignore_index=True)
        
        result_df = result_df.drop('pair_hash', axis=1)
        
        return result_df
    
    def validate_pkd_range(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        
        df = df[(df['pkd'] >= 0) & (df['pkd'] <= 20)].copy()
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} entries with invalid pKd values")
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        Q1 = df['pkd'].quantile(0.25)
        Q3 = df['pkd'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['pkd'] < lower_bound) | (df['pkd'] > upper_bound)]
        
        if len(outliers) > 0:
            logger.warning(f"Found {len(outliers)} potential outliers (outside [{lower_bound:.2f}, {upper_bound:.2f}])")
            logger.info(f"pKd distribution - Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        
        return df
    
    def normalize_pkd(self, pkd_values: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit:
            self.normalization_params = {
                'mean': np.mean(pkd_values),
                'std': np.std(pkd_values)
            }
            logger.info(f"Normalization parameters - mean: {self.normalization_params['mean']:.4f}, "
                       f"std: {self.normalization_params['std']:.4f}")
        
        if self.normalization_params is None:
            raise ValueError("Normalization parameters not set. Run with fit=True first.")
        
        normalized = (pkd_values - self.normalization_params['mean']) / self.normalization_params['std']
        return normalized
    
    def denormalize_pkd(self, normalized_values: np.ndarray) -> np.ndarray:
        if self.normalization_params is None:
            raise ValueError("Normalization parameters not set.")
        
        return normalized_values * self.normalization_params['std'] + self.normalization_params['mean']
    
    def preprocess_dataset(self, df: pd.DataFrame, fit_normalization: bool = True) -> pd.DataFrame:
        logger.info(f"Starting preprocessing with {len(df)} samples")
        
        df = df.copy()
        
        # Drop rows with missing sequences
        initial_count = len(df)
        df = df.dropna(subset=['protein1_sequence', 'protein2_sequence'])
        if len(df) < initial_count:
            logger.warning(f"Dropped {initial_count - len(df)} rows with missing sequences")
        
        df['protein1_sequence'] = df['protein1_sequence'].apply(self.clean_sequence)
        df['protein2_sequence'] = df['protein2_sequence'].apply(self.clean_sequence)
        
        valid_mask = (
            df['protein1_sequence'].apply(self.validate_sequence) &
            df['protein2_sequence'].apply(self.validate_sequence)
        )
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count} entries with invalid sequences")
            df = df[valid_mask].copy()
        
        if self.config['remove_duplicates']:
            initial_count = len(df)
            df = self.remove_duplicates(df)
            logger.info(f"Removed {initial_count - len(df)} duplicate protein pairs")
        
        # Check if pkd column exists and has values
        if 'pkd' in df.columns:
            initial_count = len(df)
            df = df.dropna(subset=['pkd'])
            if len(df) < initial_count:
                logger.warning(f"Dropped {initial_count - len(df)} rows with missing pKd values")
            df = self.validate_pkd_range(df)
        else:
            logger.warning("No 'pkd' column found in dataset")
        
        # Only normalize if pkd exists
        if 'pkd' in df.columns:
            df = self.detect_outliers(df)
            df['pkd_normalized'] = self.normalize_pkd(df['pkd'].values, fit=fit_normalization)
        else:
            df['pkd_normalized'] = 0.0  # Default value for inference without labels
        
        sequence_lengths = pd.concat([
            df['protein1_sequence'].str.len(),
            df['protein2_sequence'].str.len()
        ])
        
        logger.info(f"Sequence length statistics:")
        logger.info(f"  Mean: {sequence_lengths.mean():.1f}")
        logger.info(f"  Median: {sequence_lengths.median():.1f}")
        logger.info(f"  Std: {sequence_lengths.std():.1f}")
        logger.info(f"  Min: {sequence_lengths.min()}")
        logger.info(f"  Max: {sequence_lengths.max()}")
        
        logger.info(f"Final dataset size: {len(df)} samples")
        
        return df
    
    def train_val_split(self, df: pd.DataFrame, val_split: float = 0.1, 
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        df['pkd_bin'] = pd.qcut(df['pkd'], q=5, labels=False)
        
        train_df, val_df = train_test_split(
            df, 
            test_size=val_split,
            stratify=df['pkd_bin'],
            random_state=random_state
        )
        
        train_df = train_df.drop('pkd_bin', axis=1)
        val_df = val_df.drop('pkd_bin', axis=1)
        
        logger.info(f"Train/validation split: {len(train_df)}/{len(val_df)} samples")
        
        return train_df, val_df
    
    def save_normalization_params(self, path: str):
        if self.normalization_params is None:
            raise ValueError("No normalization parameters to save")
        
        import json
        with open(path, 'w') as f:
            json.dump(self.normalization_params, f)
        
        logger.info(f"Saved normalization parameters to {path}")
    
    def load_normalization_params(self, path: str):
        import json
        with open(path, 'r') as f:
            self.normalization_params = json.load(f)
        
        logger.info(f"Loaded normalization parameters from {path}")