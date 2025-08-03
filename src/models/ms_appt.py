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


class MultiScaleConvolution(nn.Module):
    """Extract multi-scale local patterns from protein sequences."""
    def __init__(self, input_dim: int, kernel_sizes: List[int], num_filters: int, dropout: float = 0.0):
        super().__init__()
        self.convolutions = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.output_dim = num_filters * len(kernel_sizes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        
        conv_outputs = []
        for conv in self.convolutions:
            conv_out = F.relu(conv(x))
            if self.dropout is not None:
                conv_out = self.dropout(conv_out)
            conv_outputs.append(conv_out)
        
        combined = torch.cat(conv_outputs, dim=1)
        combined = combined.transpose(1, 2)  # [batch, seq_len, output_dim]
        
        return combined


class CrossAttention(nn.Module):
    """Model explicit protein-protein interactions."""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Separate projections for query and key-value
        self.q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv_linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.kv_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key_value.shape
        
        residual = query
        query = self.layer_norm(query)
        
        # Project query
        q = self.q_linear(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project key and value together
        kv = self.kv_linear(key_value).view(batch_size, key_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, H, S, D]
        k, v = kv[0], kv[1]  # Each is [B, H, S, D]
        
        if FLASH_ATTENTION_AVAILABLE and key_mask is None and query.is_cuda:
            # Use Flash Attention for cross-attention
            q = q.transpose(1, 2).contiguous()  # [B, S, H, D]
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            attention_output = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
            attention_output = attention_output.reshape(batch_size, query_len, self.hidden_dim)
        else:
            # Efficient standard attention with float32 for stability
            q = q.float()
            k = k.float()
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if key_mask is not None:
                key_mask = key_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(key_mask == 0, -65504.0 if scores.dtype == torch.float16 else -1e9)
            
            attention_weights = F.softmax(scores, dim=-1).to(v.dtype)
            attention_weights = self.dropout(attention_weights)
            
            attention_output = torch.matmul(attention_weights, v)
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                batch_size, query_len, self.hidden_dim
            )
        
        output = self.out_linear(attention_output)
        output = self.dropout(output)
        output = output + residual
        
        return output








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
        
        # Multi-scale convolution for local pattern extraction
        conv_config = config['model']['multi_scale_conv']
        self.multi_scale_conv = MultiScaleConvolution(
            self.hidden_dim,
            conv_config['kernel_sizes'],
            conv_config['num_filters'],
            dropout=conv_config['dropout']
        )
        self.conv_projection = nn.Linear(self.multi_scale_conv.output_dim, self.hidden_dim)
        
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
        
        # Cross-attention for explicit interaction modeling
        cross_att_config = config['model']['cross_attention']
        self.cross_attention = CrossAttention(
            cross_att_config['hidden_dim'],
            cross_att_config['num_heads'],
            dropout=cross_att_config['dropout']
        )
        
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
        
        # Apply multi-scale convolution first
        conv_features1 = self.multi_scale_conv(features1)
        conv_features2 = self.multi_scale_conv(features2)
        
        # Project conv features back to hidden_dim and combine with original
        conv_features1 = self.conv_projection(conv_features1)
        conv_features2 = self.conv_projection(conv_features2)
        
        # Residual connection - combine conv features with original
        features1 = features1 + conv_features1
        features2 = features2 + conv_features2
        
        # Update stacked features with conv-enhanced features
        stacked = torch.stack([features1, features2], dim=1)
        stacked_flat = stacked.view(-1, max_len, self.hidden_dim)
        
        # Apply transformer encoder
        transformed_flat = self.transformer(stacked_flat)
        
        # Reshape back: [batch, 2, seq_len, hidden_dim]
        transformed = transformed_flat.view(batch_size, 2, max_len, self.hidden_dim)
        
        # Split transformed features
        trans_features1 = transformed[:, 0, :, :]  # [batch, seq_len, hidden_dim]
        trans_features2 = transformed[:, 1, :, :]  # [batch, seq_len, hidden_dim]
        
        # Apply cross-attention for explicit protein-protein interaction
        cross_features1 = self.cross_attention(trans_features1, trans_features2, mask1, mask2)
        cross_features2 = self.cross_attention(trans_features2, trans_features1, mask2, mask1)
        
        # Learn interface importance scores for each protein
        importance1 = self.interface_gating(cross_features1, mask1)
        importance2 = self.interface_gating(cross_features2, mask2)
        
        # Apply adaptive pooling to get multiple representations
        pools1 = self.adaptive_pooling(cross_features1, importance1, mask1)
        pools2 = self.adaptive_pooling(cross_features2, importance2, mask2)
        
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