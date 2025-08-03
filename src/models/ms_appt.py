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
    def __init__(self, input_dim: int, kernel_sizes: List[int], num_filters: int, dropout: float = 0.0):
        super().__init__()
        self.convolutions = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.output_dim = num_filters * len(kernel_sizes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convolutions:
            conv_out = F.relu(conv(x))
            if self.dropout is not None:
                conv_out = self.dropout(conv_out)
            conv_outputs.append(conv_out)
        
        combined = torch.cat(conv_outputs, dim=1)
        combined = combined.transpose(1, 2)
        
        return combined


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Use a single linear projection for Q, K, V for efficiency
        self.qkv_linear = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.qkv_linear.weight)
        nn.init.xavier_uniform_(self.out_linear.weight)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        residual = x
        x = self.layer_norm(x)
        
        # Single projection for Q, K, V
        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, H, S, D]
        
        if FLASH_ATTENTION_AVAILABLE and mask is None and x.is_cuda:
            # Use Flash Attention for maximum speed (no mask support)
            # Reshape for flash attention: [B, S, H, D]
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            attention_output = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
            attention_output = attention_output.reshape(batch_size, seq_len, self.hidden_dim)
        else:
            # Efficient standard attention
            # Use float32 for attention computation to avoid overflow
            q = q.float()
            k = k.float()
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask == 0, -65504.0 if scores.dtype == torch.float16 else -1e9)
            
            attention_weights = F.softmax(scores, dim=-1).to(v.dtype)
            attention_weights = self.dropout(attention_weights)
            
            attention_output = torch.matmul(attention_weights, v)
            attention_output = attention_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.hidden_dim
            )
        
        output = self.out_linear(attention_output)
        output = self.dropout(output)
        output = output + residual
        
        return output


class CrossAttention(nn.Module):
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


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, attention_hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_scores = self.attention_weights(x).squeeze(-1)
        
        if mask is not None:
            # Use -65504 for float16 compatibility
            attention_scores = attention_scores.masked_fill(mask == 0, -65504.0 if attention_scores.dtype == torch.float16 else -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        weighted_sum = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        return weighted_sum


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2,
                 activation: str = 'gelu', use_layer_norm: bool = True,
                 use_residual: bool = True):
        super().__init__()
        
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = nn.ModuleDict({
                'linear': nn.Linear(prev_dim, hidden_dim),
                'dropout': nn.Dropout(dropout)
            })
            
            if use_layer_norm:
                layer['norm'] = nn.LayerNorm(hidden_dim)
            
            if use_residual and prev_dim == hidden_dim:
                layer['use_residual'] = nn.Parameter(torch.tensor(True), requires_grad=False)
            
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = x
            
            x = layer['linear'](x)
            
            if 'norm' in layer:
                x = layer['norm'](x)
            
            x = self.activation(x)
            x = layer['dropout'](x)
            
            if 'use_residual' in layer:
                x = x + residual
        
        output = self.output_layer(x)
        return output


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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Keep cross-attention for explicit interaction modeling
        cross_att_config = config['model']['cross_attention']
        self.cross_attention = CrossAttention(
            self.hidden_dim,
            cross_att_config['num_heads'],
            dropout=0.1  # Lower dropout
        )
        
        # MLP head - match simple APPT structure
        mlp_config = config['model']['mlp']
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 160),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(160, 80),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(80, 1)
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
        
        # Stack proteins as separate sequences (like simple APPT)
        # Shape: [batch, 2, max_seq_len, hidden_dim]
        max_len = max(features1.shape[1], features2.shape[1])
        
        # Pad to same length
        if features1.shape[1] < max_len:
            pad_len = max_len - features1.shape[1]
            features1 = F.pad(features1, (0, 0, 0, pad_len))
        if features2.shape[1] < max_len:
            pad_len = max_len - features2.shape[1]
            features2 = F.pad(features2, (0, 0, 0, pad_len))
        
        # Stack: [batch, 2, seq_len, hidden_dim]
        stacked = torch.stack([features1, features2], dim=1)
        
        # Reshape for transformer: [batch * 2, seq_len, hidden_dim]
        stacked_flat = stacked.view(-1, max_len, self.hidden_dim)
        
        # Apply transformer encoder
        transformed = self.transformer(stacked_flat)
        
        # Reshape back: [batch, 2, seq_len, hidden_dim]
        transformed = transformed.view(batch_size, 2, max_len, self.hidden_dim)
        
        # Mean pool over sequence dimension for each protein
        pooled = transformed.mean(dim=2)  # [batch, 2, hidden_dim]
        
        # Final pooling over the 2 proteins
        final_features = pooled.mean(dim=1)  # [batch, hidden_dim]
        
        # Final prediction
        output = self.mlp(final_features)
        
        return output.squeeze(-1)
    
    def _mean_pool_with_mask(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Simple mean pooling with mask."""
        mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        masked_features = features * mask
        sum_features = masked_features.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_features / sum_mask