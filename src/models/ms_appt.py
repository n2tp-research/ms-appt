import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


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
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        residual = x
        x = self.layer_norm(x)
        
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            # Use -65504 for float16 compatibility (max negative value for half precision)
            scores = scores.masked_fill(mask == 0, -65504.0 if scores.dtype == torch.float16 else -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        self._attention_weights = attention_weights.clone()
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
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
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key_value.shape[1]
        
        residual = query
        query = self.layer_norm(query)
        
        Q = self.q_linear(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key_value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(key_value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if key_mask is not None:
            key_mask = key_mask.unsqueeze(1).unsqueeze(2)
            # Use -65504 for float16 compatibility (max negative value for half precision)
            scores = scores.masked_fill(key_mask == 0, -65504.0 if scores.dtype == torch.float16 else -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        self._attention_weights = attention_weights.clone()
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
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
        
        proj_config = config['model']['projection']
        layers = [
            nn.Linear(self.embedding_dim, proj_config['output_dim']),
            nn.ReLU() if proj_config.get('activation', 'relu') == 'relu' else nn.GELU()
        ]
        if 'dropout' in proj_config:
            layers.append(nn.Dropout(proj_config['dropout']))
        self.projector = nn.Sequential(*layers)
        self.hidden_dim = proj_config['output_dim']
        
        conv_config = config['model']['conv_layers']
        self.multi_scale_conv = MultiScaleConvolution(
            self.hidden_dim,
            conv_config['kernel_sizes'],
            conv_config['num_filters'],
            conv_config.get('dropout', 0.0)
        )
        self.conv_output_dim = self.multi_scale_conv.output_dim
        
        self_att_config = config['model']['self_attention']
        num_self_att_layers = self_att_config.get('num_layers', 1)
        self.self_attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(
                self.conv_output_dim,
                self_att_config['num_heads'],
                self_att_config['dropout']
            ) for _ in range(num_self_att_layers)
        ])
        
        cross_att_config = config['model']['cross_attention']
        num_cross_att_layers = cross_att_config.get('num_layers', 1)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(
                self.conv_output_dim,
                cross_att_config['num_heads'],
                cross_att_config['dropout']
            ) for _ in range(num_cross_att_layers)
        ])
        
        pooling_config = config['model']['pooling']
        self.use_mean_pool = 'mean' in pooling_config['methods']
        self.use_max_pool = 'max' in pooling_config['methods']
        self.use_attention_pool = 'attention' in pooling_config['methods']
        
        if self.use_attention_pool:
            self.attention_pool = AttentionPooling(
                self.conv_output_dim,
                pooling_config['attention_hidden_dim']
            )
        
        num_pool_methods = len(pooling_config['methods'])
        pool_output_dim = self.conv_output_dim * num_pool_methods
        pair_feature_dim = pool_output_dim * 4
        
        mlp_config = config['model']['mlp']
        self.mlp = DeepMLP(
            pair_feature_dim,
            mlp_config['hidden_dims'],
            mlp_config['dropout'],
            mlp_config['activation'],
            mlp_config['use_layer_norm'],
            mlp_config['use_residual']
        )
        
    def _create_padding_mask(self, sequences: List[str], max_length: int, device: torch.device) -> torch.Tensor:
        batch_size = len(sequences)
        mask = torch.zeros(batch_size, max_length, device=device)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            mask[i, :seq_len] = 1
        
        return mask
    
    def _pool_features(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled_features = []
        
        if self.use_mean_pool:
            masked_features = features * mask.unsqueeze(-1)
            sum_features = masked_features.sum(dim=1)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_pooled = sum_features / count
            pooled_features.append(mean_pooled)
        
        if self.use_max_pool:
            # Use appropriate negative value for dtype
            fill_value = -65504.0 if features.dtype == torch.float16 else -1e9
            masked_features = features.masked_fill(~mask.unsqueeze(-1).bool(), fill_value)
            max_pooled, _ = masked_features.max(dim=1)
            pooled_features.append(max_pooled)
        
        if self.use_attention_pool:
            attention_pooled = self.attention_pool(features, mask)
            pooled_features.append(attention_pooled)
        
        return torch.cat(pooled_features, dim=-1)
    
    def forward(self, protein1_embeddings: torch.Tensor, protein2_embeddings: torch.Tensor,
                protein1_sequences: List[str], protein2_sequences: List[str]) -> torch.Tensor:
        
        device = protein1_embeddings.device
        
        proj1 = self.projector(protein1_embeddings)
        proj2 = self.projector(protein2_embeddings)
        
        conv1 = self.multi_scale_conv(proj1)
        conv2 = self.multi_scale_conv(proj2)
        
        mask1 = self._create_padding_mask(protein1_sequences, conv1.shape[1], device)
        mask2 = self._create_padding_mask(protein2_sequences, conv2.shape[1], device)
        
        # Apply multiple self-attention layers
        self_att1 = conv1
        for self_att_layer in self.self_attention_layers:
            self_att1 = self_att_layer(self_att1, mask1)
        
        self_att2 = conv2
        for self_att_layer in self.self_attention_layers:
            self_att2 = self_att_layer(self_att2, mask2)
        
        # Apply multiple cross-attention layers
        cross_att1 = self_att1
        cross_att2 = self_att2
        for cross_att_layer in self.cross_attention_layers:
            cross_att1_new = cross_att_layer(cross_att1, cross_att2, mask1, mask2)
            cross_att2_new = cross_att_layer(cross_att2, cross_att1, mask2, mask1)
            cross_att1 = cross_att1_new
            cross_att2 = cross_att2_new
        
        pooled1 = self._pool_features(cross_att1, mask1)
        pooled2 = self._pool_features(cross_att2, mask2)
        
        element_product = pooled1 * pooled2
        absolute_diff = torch.abs(pooled1 - pooled2)
        
        pair_features = torch.cat([pooled1, pooled2, element_product, absolute_diff], dim=-1)
        
        output = self.mlp(pair_features)
        
        return output.squeeze(-1)