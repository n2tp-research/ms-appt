import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False








class InterfaceGating(nn.Module):
    """Learns to identify important regions (potential binding interfaces) in protein sequences."""
    
    def __init__(self, hidden_dim: int, gating_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gating_net = nn.Sequential(
            nn.Linear(hidden_dim, gating_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden_dim, 1),
            nn.Sigmoid()  # Output importance scores [0, 1]
        )
        
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] binary mask for valid positions
        Returns:
            importance_scores: [batch, seq_len] importance scores for each position
        """
        scores = self.gating_net(features).squeeze(-1)  # [batch, seq_len]
        
        if mask is not None:
            scores = scores * mask  # Zero out padded positions
            
        return scores


class AdaptiveInterfacePooling(nn.Module):
    """Advanced pooling that focuses on binding interfaces and hotspots."""
    
    def __init__(self, hidden_dim: int, top_k_ratio: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k_ratio = top_k_ratio  # Fraction of positions to consider as hotspots
        
    def forward(self, features: torch.Tensor, importance_scores: torch.Tensor, 
                mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs multiple pooling strategies based on importance scores.
        
        Returns dict with:
            - interface_pool: Weighted average of high-importance regions
            - non_interface_pool: Weighted average of low-importance regions  
            - hotspot_pool: Average of top-k most important positions
            - global_pool: Standard mean pooling
            - interface_size: Normalized count of important positions
        """
        batch_size, seq_len = importance_scores.shape
        
        # Normalize importance scores per sequence
        masked_scores = importance_scores * mask
        score_sum = masked_scores.sum(dim=1, keepdim=True).clamp(min=1e-9)
        normalized_scores = masked_scores / score_sum
        
        # 1. Interface pool (weighted by importance)
        interface_weights = normalized_scores.unsqueeze(-1)  # [batch, seq_len, 1]
        interface_pool = (features * interface_weights).sum(dim=1)  # [batch, hidden_dim]
        
        # 2. Non-interface pool (weighted by 1 - importance)
        non_interface_scores = (1 - importance_scores) * mask
        non_interface_sum = non_interface_scores.sum(dim=1, keepdim=True).clamp(min=1e-9)
        non_interface_weights = (non_interface_scores / non_interface_sum).unsqueeze(-1)
        non_interface_pool = (features * non_interface_weights).sum(dim=1)
        
        # 3. Hotspot pool (top-k positions)
        k = max(1, int(seq_len * self.top_k_ratio))
        top_k_scores, top_k_indices = torch.topk(importance_scores, k, dim=1)
        
        # Gather top-k features
        top_k_features = torch.gather(features, 1, 
                                    top_k_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        hotspot_pool = top_k_features.mean(dim=1)
        
        # 4. Global pool (standard mean pooling with mask)
        mask_expanded = mask.unsqueeze(-1)
        global_pool = (features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        # 5. Interface characteristics
        interface_size = masked_scores.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        return {
            'interface_pool': interface_pool,
            'non_interface_pool': non_interface_pool,
            'hotspot_pool': hotspot_pool,
            'global_pool': global_pool,
            'interface_size': interface_size
        }


class MS_APPT(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        encoder_config = config['model']['encoder']
        self.embedding_dim = encoder_config['embedding_dim']
        
        # Projection layer
        proj_config = config['model']['projection']
        self.hidden_dim = proj_config['output_dim']
        
        # Single shared projection (like simple APPT)
        self.projector = nn.Linear(self.embedding_dim, self.hidden_dim)
        
        # Add transformer encoder layers (proven to work in simple APPT)
        transformer_config = config['model']['transformer']
        # Ensure num_heads divides hidden_dim evenly
        num_heads = transformer_config['num_heads']
        if self.hidden_dim % num_heads != 0:
            # Find the nearest valid number of heads
            valid_heads = [h for h in [8, 6, 4, 3, 2] if self.hidden_dim % h == 0]
            num_heads = valid_heads[0] if valid_heads else 1
            print(f"Warning: Adjusting num_heads from {transformer_config['num_heads']} to {num_heads} for hidden_dim={self.hidden_dim}")
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * transformer_config['dim_feedforward_multiplier'],
            dropout=transformer_config['dropout'],
            activation=transformer_config['activation'],
            batch_first=True,
            norm_first=transformer_config['norm_first']
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_config['num_layers'])
        
        # Interface gating mechanism
        gating_config = config['model']['interface_gating']
        self.interface_gating = InterfaceGating(
            self.hidden_dim,
            gating_config['hidden_dim'],
            dropout=gating_config['dropout']
        )
        
        # Adaptive pooling
        pooling_config = config['model']['adaptive_pooling']
        self.adaptive_pooling = AdaptiveInterfacePooling(
            self.hidden_dim,
            top_k_ratio=pooling_config['top_k_ratio']
        )
        
        # MLP head using config dimensions
        mlp_config = config['model']['mlp']
        mlp_dims = mlp_config['hidden_dims']
        # Input features:
        # - 2 * (interface + non_interface + hotspot + global) = 8 * hidden_dim
        # - complementarity, difference = 2 * hidden_dim  
        # - interface_sizes = 2
        input_dim = self.hidden_dim * 10 + 2
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, mlp_dims[0]),
            nn.GELU(),
            nn.Dropout(mlp_config['dropout']),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.GELU(),
            nn.Dropout(mlp_config['dropout']),
            nn.Linear(mlp_dims[1], mlp_dims[2]),
            nn.GELU(),
            nn.Dropout(mlp_config['dropout']),
            nn.Linear(mlp_dims[2], 1)
        )
        
    def _create_padding_mask(self, sequences: List[str], max_length: int, device: torch.device) -> torch.Tensor:
        batch_size = len(sequences)
        mask = torch.zeros(batch_size, max_length, device=device)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            mask[i, :seq_len] = 1
        
        return mask
    
    def forward(self, protein1_embeddings: torch.Tensor, protein2_embeddings: torch.Tensor,
                protein1_sequences: List[str], protein2_sequences: List[str]) -> torch.Tensor:
        
        device = protein1_embeddings.device
        batch_size = protein1_embeddings.shape[0]
        
        # Project embeddings
        features1 = self.projector(protein1_embeddings)
        features2 = self.projector(protein2_embeddings)
        
        # Create masks for original sequences
        mask1 = self._create_padding_mask(protein1_sequences, features1.shape[1], device)
        mask2 = self._create_padding_mask(protein2_sequences, features2.shape[1], device)
        
        # Stack proteins as separate sequences (like simple APPT)
        # Shape: [batch, 2, max_seq_len, hidden_dim]
        max_len = max(features1.shape[1], features2.shape[1])
        
        # Pad features and masks to same length
        if features1.shape[1] < max_len:
            pad_len = max_len - features1.shape[1]
            features1 = F.pad(features1, (0, 0, 0, pad_len))
            mask1 = F.pad(mask1, (0, pad_len))
        if features2.shape[1] < max_len:
            pad_len = max_len - features2.shape[1]
            features2 = F.pad(features2, (0, 0, 0, pad_len))
            mask2 = F.pad(mask2, (0, pad_len))
        
        # Stack: [batch, 2, seq_len, hidden_dim]
        stacked = torch.stack([features1, features2], dim=1)
        
        # Reshape for transformer: [batch * 2, seq_len, hidden_dim]
        stacked_flat = stacked.view(-1, max_len, self.hidden_dim)
        
        # Apply transformer encoder
        transformed_flat = self.transformer(stacked_flat)
        
        # Reshape back: [batch, 2, seq_len, hidden_dim]
        transformed = transformed_flat.view(batch_size, 2, max_len, self.hidden_dim)
        
        # Split transformed features
        trans_features1 = transformed[:, 0, :, :]  # [batch, seq_len, hidden_dim]
        trans_features2 = transformed[:, 1, :, :]  # [batch, seq_len, hidden_dim]
        
        # Learn interface importance scores for each protein
        importance1 = self.interface_gating(trans_features1, mask1)
        importance2 = self.interface_gating(trans_features2, mask2)
        
        # Apply adaptive pooling to get multiple representations
        pools1 = self.adaptive_pooling(trans_features1, importance1, mask1)
        pools2 = self.adaptive_pooling(trans_features2, importance2, mask2)
        
        # Biological feature engineering using pooled features
        # 1. Complementarity (using interface pools - what actually binds)
        complementarity = pools1['interface_pool'] * pools2['interface_pool']
        
        # 2. Difference features (important for specificity)
        difference = torch.abs(pools1['global_pool'] - pools2['global_pool'])
        
        # Concatenate all meaningful features
        combined_features = torch.cat([
            # Protein 1 pools
            pools1['interface_pool'],      # What binds
            pools1['non_interface_pool'],  # What doesn't bind
            pools1['hotspot_pool'],        # Key residues
            pools1['global_pool'],         # Overall features
            # Protein 2 pools  
            pools2['interface_pool'],
            pools2['non_interface_pool'],
            pools2['hotspot_pool'],
            pools2['global_pool'],
            # Interaction features
            complementarity,               # How well interfaces match
            difference,                    # Dissimilarity for specificity
            # Interface characteristics
            pools1['interface_size'],      # How much of protein 1 binds
            pools2['interface_size']       # How much of protein 2 binds
        ], dim=-1)
        
        # Final prediction
        output = self.mlp(combined_features)
        
        return output.squeeze(-1)
    
    def _mean_pool_with_mask(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Simple mean pooling with mask."""
        mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        masked_features = features * mask
        sum_features = masked_features.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_features / sum_mask