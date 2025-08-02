import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import h5py
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import os
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


class ESM2EmbeddingExtractor:
    def __init__(self, model_name: str, cache_dir: str, device: str = 'cuda'):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.device = device
        
        self.cache_file_path = self.cache_dir / 'embeddings_cache.h5'
        self._cache_lock = threading.Lock()
        
        logger.info(f"Loading ESM-2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"ESM-2 embedding dimension: {self.embedding_dim}")
        
        self._init_cache()
    
    def _init_cache(self):
        if not self.cache_file_path.exists():
            with h5py.File(self.cache_file_path, 'w') as f:
                f.attrs['model_name'] = self.model_name
                f.attrs['embedding_dim'] = self.embedding_dim
            logger.info(f"Created new cache file: {self.cache_file_path}")
        else:
            with h5py.File(self.cache_file_path, 'r') as f:
                cached_model = f.attrs.get('model_name', '')
                if cached_model != self.model_name:
                    logger.warning(f"Cache was created with different model ({cached_model}). "
                                 f"Consider clearing cache.")
    
    def _get_sequence_hash(self, sequence: str) -> str:
        return hashlib.md5(sequence.encode()).hexdigest()
    
    @contextmanager
    def _get_cache_file(self, mode='r'):
        with self._cache_lock:
            cache_file = h5py.File(self.cache_file_path, mode)
            try:
                yield cache_file
            finally:
                cache_file.close()
    
    def _compute_embedding(self, sequence: str) -> torch.Tensor:
        inputs = self.tokenizer(
            sequence,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        
        embeddings = embeddings[:, 1:-1, :]
        
        return embeddings.squeeze(0).cpu()
    
    def get_embedding(self, sequence: str, use_cache: bool = True) -> torch.Tensor:
        sequence_hash = self._get_sequence_hash(sequence)
        
        if use_cache:
            with self._get_cache_file('r') as cache_file:
                if sequence_hash in cache_file:
                    embedding = torch.tensor(cache_file[sequence_hash][:])
                    return embedding
        
        embedding = self._compute_embedding(sequence)
        
        if use_cache:
            with self._get_cache_file('a') as cache_file:
                if sequence_hash not in cache_file:
                    cache_file.create_dataset(
                        sequence_hash,
                        data=embedding.numpy(),
                        compression='gzip',
                        compression_opts=4
                    )
        
        return embedding
    
    def get_batch_embeddings(self, sequences: List[str], use_cache: bool = True) -> List[torch.Tensor]:
        embeddings = []
        uncached_sequences = []
        uncached_indices = []
        
        for i, seq in enumerate(sequences):
            if use_cache:
                seq_hash = self._get_sequence_hash(seq)
                with self._get_cache_file('r') as cache_file:
                    if seq_hash in cache_file:
                        embedding = torch.tensor(cache_file[seq_hash][:])
                        embeddings.append(embedding)
                    else:
                        embeddings.append(None)
                        uncached_sequences.append(seq)
                        uncached_indices.append(i)
            else:
                uncached_sequences.append(seq)
                uncached_indices.append(i)
                embeddings.append(None)
        
        if uncached_sequences:
            batch_embeddings = self._compute_batch_embeddings(uncached_sequences)
            
            for idx, embedding in zip(uncached_indices, batch_embeddings):
                embeddings[idx] = embedding
                
                if use_cache:
                    seq_hash = self._get_sequence_hash(sequences[idx])
                    with self._get_cache_file('a') as cache_file:
                        if seq_hash not in cache_file:
                            cache_file.create_dataset(
                                seq_hash,
                                data=embedding.numpy(),
                                compression='gzip',
                                compression_opts=4
                            )
        
        return embeddings
    
    def _compute_batch_embeddings(self, sequences: List[str]) -> List[torch.Tensor]:
        sorted_data = sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True)
        sorted_indices = [x[0] for x in sorted_data]
        sorted_sequences = [x[1] for x in sorted_data]
        
        embeddings = []
        
        for seq in sorted_sequences:
            embedding = self._compute_embedding(seq)
            embeddings.append(embedding)
        
        unsorted_embeddings = [None] * len(sequences)
        for sorted_idx, orig_idx in enumerate(sorted_indices):
            unsorted_embeddings[orig_idx] = embeddings[sorted_idx]
        
        return unsorted_embeddings
    
    def clear_cache(self):
        if self.cache_file_path.exists():
            os.remove(self.cache_file_path)
            self._init_cache()
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        stats = {
            'cache_file': str(self.cache_file_path),
            'exists': self.cache_file_path.exists()
        }
        
        if stats['exists']:
            with self._get_cache_file('r') as cache_file:
                stats['num_sequences'] = len(cache_file.keys())
                stats['file_size_mb'] = self.cache_file_path.stat().st_size / (1024 * 1024)
                stats['model_name'] = cache_file.attrs.get('model_name', 'unknown')
        
        return stats
    
    def set_model_eval(self):
        self.model.eval()
    
    def set_model_train(self):
        self.model.train()


class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu'):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))