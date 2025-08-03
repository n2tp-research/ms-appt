import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ProteinPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_length: Optional[int] = None):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        item = {
            'protein1_sequence': row['protein1_sequence'],
            'protein2_sequence': row['protein2_sequence'],
            'index': idx
        }
        
        # Add pkd values only if they exist
        if 'pkd' in row:
            item['pkd'] = torch.tensor(row['pkd'], dtype=torch.float32)
        if 'pkd_normalized' in row:
            item['pkd_normalized'] = torch.tensor(row['pkd_normalized'], dtype=torch.float32)
            
        return item


def collate_protein_pairs(batch: list) -> Dict:
    protein1_seqs = [item['protein1_sequence'] for item in batch]
    protein2_seqs = [item['protein2_sequence'] for item in batch]
    indices = torch.tensor([item['index'] for item in batch])
    
    result = {
        'protein1_sequences': protein1_seqs,
        'protein2_sequences': protein2_seqs,
        'indices': indices
    }
    
    # Add pkd values only if they exist in the batch
    if 'pkd' in batch[0]:
        pkd_values = torch.stack([item['pkd'] for item in batch])
        result['pkd'] = pkd_values
        
    if 'pkd_normalized' in batch[0]:
        pkd_normalized = torch.stack([item['pkd_normalized'] for item in batch])
        result['pkd_normalized'] = pkd_normalized
    
    return result


class DynamicBatchSampler:
    def __init__(self, dataset: ProteinPairDataset, batch_size: int, 
                 max_tokens: int = 50000, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        
        self.sequence_lengths = []
        for i in range(len(dataset)):
            row = dataset.df.iloc[i]
            total_length = len(row['protein1_sequence']) + len(row['protein2_sequence'])
            self.sequence_lengths.append(total_length)
        
        self._create_batches()
    
    def _create_batches(self):
        indices = np.arange(len(self.dataset))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        sorted_indices = sorted(indices, key=lambda i: self.sequence_lengths[i])
        
        self.batches = []
        current_batch = []
        current_tokens = 0
        
        for idx in sorted_indices:
            seq_len = self.sequence_lengths[idx]
            
            if current_batch and (len(current_batch) >= self.batch_size or 
                                current_tokens + seq_len > self.max_tokens):
                self.batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
            current_tokens += seq_len
        
        if current_batch:
            self.batches.append(current_batch)
        
        if self.shuffle:
            np.random.shuffle(self.batches)
    
    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)


def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      config: Dict) -> Tuple[DataLoader, DataLoader]:
    
    train_dataset = ProteinPairDataset(train_df, max_length=config['data']['preprocessing']['max_length'])
    val_dataset = ProteinPairDataset(val_df, max_length=config['data']['preprocessing']['max_length'])
    
    train_batch_sampler = DynamicBatchSampler(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_batch_sampler = DynamicBatchSampler(
        val_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_protein_pairs,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=config['hardware']['num_workers'] > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=collate_protein_pairs,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=config['hardware']['num_workers'] > 0
    )
    
    logger.info(f"Created train loader with {len(train_loader)} batches")
    logger.info(f"Created validation loader with {len(val_loader)} batches")
    
    return train_loader, val_loader