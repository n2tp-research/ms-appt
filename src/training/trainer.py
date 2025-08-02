import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    def __init__(self, checkpoint_dir: str, metric: str = 'validation_rmse',
                 mode: str = 'min', save_best_only: bool = True,
                 save_last_k: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last_k = save_last_k
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_history = []
        
    def save_checkpoint(self, state: Dict, score: float, epoch: int):
        is_best = False
        
        if self.mode == 'min':
            is_best = score < self.best_score
        else:
            is_best = score > self.best_score
        
        if is_best:
            self.best_score = score
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(state, checkpoint_path)
            logger.info(f"Saved best model with {self.metric}={score:.4f}")
        
        if not self.save_best_only or is_best:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(state, checkpoint_path)
            self.checkpoint_history.append(checkpoint_path)
            
            if len(self.checkpoint_history) > self.save_last_k:
                old_checkpoint = self.checkpoint_history.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
        
        with open(self.checkpoint_dir / 'checkpoint_info.json', 'w') as f:
            json.dump({
                'best_score': self.best_score,
                'best_epoch': epoch if is_best else None,
                'metric': self.metric,
                'mode': self.mode
            }, f, indent=2)
        
        return is_best


class EarlyStopping:
    def __init__(self, patience: int = 10, metric: str = 'validation_rmse',
                 mode: str = 'min'):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.mode == 'min':
            improved = score < self.best_score
        else:
            improved = score > self.best_score
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered. No improvement for {self.patience} epochs.")
        
        return self.should_stop


class MS_APPT_Trainer:
    def __init__(self, model: nn.Module, embedding_extractor, config: Dict,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.embedding_extractor = embedding_extractor
        self.config = config
        self.device = device
        
        self.setup_optimizer()
        self.setup_loss()
        self.setup_mixed_precision()
        self.setup_callbacks()
        
        self.current_epoch = 0
        self.global_step = 0
        
    def setup_optimizer(self):
        opt_config = self.config['training']['optimizer']
        
        # Note: ESM-2 is in the embedding_extractor, not in self.model
        # So we just optimize the MS-APPT model parameters
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(opt_config['learning_rate']),
            betas=opt_config['betas'],
            eps=float(opt_config['eps']),
            weight_decay=float(opt_config['weight_decay'])
        )
        
    def setup_loss(self):
        if self.config['loss']['type'] == 'mse':
            self.criterion = nn.MSELoss(reduction=self.config['loss']['reduction'])
        else:
            raise ValueError(f"Unknown loss type: {self.config['loss']['type']}")
    
    def setup_mixed_precision(self):
        self.use_amp = self.config['training']['mixed_precision']
        if self.use_amp:
            self.scaler = GradScaler()
    
    def setup_callbacks(self):
        early_stop_config = self.config['training']['early_stopping']
        self.early_stopping = EarlyStopping(
            patience=early_stop_config['patience'],
            metric=early_stop_config['metric'],
            mode=early_stop_config['mode']
        )
        
        self.checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=self.config['training']['checkpoint_dir'],
            metric=early_stop_config['metric'],
            mode=early_stop_config['mode'],
            save_best_only=self.config['training']['save_best_only'],
            save_last_k=self.config['training']['save_last_k']
        )
    
    def _get_embeddings_batch(self, protein1_sequences: List[str],
                            protein2_sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        embeddings1 = self.embedding_extractor.get_batch_embeddings(protein1_sequences)
        embeddings2 = self.embedding_extractor.get_batch_embeddings(protein2_sequences)
        
        max_len1 = max(e.shape[0] for e in embeddings1)
        max_len2 = max(e.shape[0] for e in embeddings2)
        
        batch_size = len(embeddings1)
        padded_embeddings1 = torch.zeros(batch_size, max_len1, embeddings1[0].shape[1])
        padded_embeddings2 = torch.zeros(batch_size, max_len2, embeddings2[0].shape[1])
        
        for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
            padded_embeddings1[i, :emb1.shape[0]] = emb1
            padded_embeddings2[i, :emb2.shape[0]] = emb2
        
        return padded_embeddings1.to(self.device), padded_embeddings2.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        # Keep ESM-2 in eval mode (frozen) for now
        # TODO: Implement ESM-2 fine-tuning if needed
        self.embedding_extractor.set_model_eval()
        
        epoch_metrics = defaultdict(float)
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        optimizer_step_count = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            protein1_embeddings, protein2_embeddings = self._get_embeddings_batch(
                batch['protein1_sequences'], batch['protein2_sequences']
            )
            
            targets = batch['pkd_normalized'].to(self.device)
            
            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    predictions = self.model(
                        protein1_embeddings, protein2_embeddings,
                        batch['protein1_sequences'], batch['protein2_sequences']
                    )
                    loss = self.criterion(predictions, targets)
                
                loss = loss / self.config['training']['gradient_accumulation_steps']
                self.scaler.scale(loss).backward()
            else:
                predictions = self.model(
                    protein1_embeddings, protein2_embeddings,
                    batch['protein1_sequences'], batch['protein2_sequences']
                )
                loss = self.criterion(predictions, targets)
                loss = loss / self.config['training']['gradient_accumulation_steps']
                loss.backward()
            
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clipping']['max_norm']
                )
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                optimizer_step_count += 1
                self.global_step += 1
            
            batch_loss = loss.item() * self.config['training']['gradient_accumulation_steps']
            epoch_metrics['loss'] += batch_loss
            epoch_metrics['grad_norm'] += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            
            progress_bar.set_postfix({
                'loss': batch_loss,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            })
        
        num_batches = len(train_loader)
        epoch_metrics['loss'] /= num_batches
        epoch_metrics['grad_norm'] /= optimizer_step_count
        
        return dict(epoch_metrics)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        self.embedding_extractor.set_model_eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                protein1_embeddings, protein2_embeddings = self._get_embeddings_batch(
                    batch['protein1_sequences'], batch['protein2_sequences']
                )
                
                targets = batch['pkd_normalized'].to(self.device)
                
                if self.use_amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        predictions = self.model(
                            protein1_embeddings, protein2_embeddings,
                            batch['protein1_sequences'], batch['protein2_sequences']
                        )
                        loss = self.criterion(predictions, targets)
                else:
                    predictions = self.model(
                        protein1_embeddings, protein2_embeddings,
                        batch['protein1_sequences'], batch['protein2_sequences']
                    )
                    loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'rmse': np.sqrt(np.mean((all_predictions - all_targets) ** 2)),
            'mae': np.mean(np.abs(all_predictions - all_targets))
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int) -> Dict[str, List[float]]:
        
        total_steps = len(train_loader) * num_epochs // self.config['training']['gradient_accumulation_steps']
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=float(self.config['training']['scheduler']['eta_min'])
        )
        
        history = defaultdict(list)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            for metric_name, value in train_metrics.items():
                history[f'train_{metric_name}'].append(value)
            
            val_metrics = self.validate(val_loader)
            for metric_name, value in val_metrics.items():
                history[f'val_{metric_name}'].append(value)
            
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val RMSE: {val_metrics['rmse']:.4f}, "
                       f"Val MAE: {val_metrics['mae']:.4f}")
            
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'metrics': {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            }
            
            val_rmse = val_metrics['rmse']
            is_best = self.checkpoint_manager.save_checkpoint(checkpoint_state, val_rmse, epoch)
            
            if self.early_stopping(val_rmse):
                logger.info("Early stopping triggered!")
                break
        
        return dict(history)
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")