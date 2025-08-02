#!/usr/bin/env python3
"""
Calculate model parameters for scaled-down MS-APPT
"""

# Configuration
embedding_dim = 1280  # ESM-2 650M
projection_dim = 256
conv_filters = 32
conv_kernels = 3  # Number of kernel sizes
attention_heads = 4
attention_layers_self = 2
attention_layers_cross = 2
mlp_dims = [256, 128, 64]
num_pooling_methods = 3

print("MS-APPT Model Parameter Calculation")
print("=" * 50)

total_params = 0

# 1. Projection layer
proj_params = embedding_dim * projection_dim + projection_dim  # weights + bias
total_params += proj_params
print(f"Projection (1280→256): {proj_params:,}")

# 2. Multi-scale convolution
conv_output_dim = conv_filters * conv_kernels  # 32 * 3 = 96
conv_params = 0
for kernel_size in [3, 5, 7]:
    # Conv1d params = in_channels * out_channels * kernel_size + out_channels
    conv_params += projection_dim * conv_filters * kernel_size + conv_filters
total_params += conv_params
print(f"Multi-scale Conv: {conv_params:,}")

# 3. Self-attention layers
# Each attention layer has: QKV projection + output projection + layer norm
attention_dim = conv_output_dim  # 96
self_att_params = 0
for _ in range(attention_layers_self):
    # QKV projection (single matrix)
    self_att_params += attention_dim * (3 * attention_dim)  # no bias in our implementation
    # Output projection
    self_att_params += attention_dim * attention_dim + attention_dim
    # Layer norm
    self_att_params += 2 * attention_dim  # scale + bias
total_params += self_att_params
print(f"Self-attention ({attention_layers_self} layers): {self_att_params:,}")

# 4. Cross-attention layers
cross_att_params = 0
for _ in range(attention_layers_cross):
    # Q projection
    cross_att_params += attention_dim * attention_dim  # no bias
    # KV projection (single matrix)
    cross_att_params += attention_dim * (2 * attention_dim)  # no bias
    # Output projection
    cross_att_params += attention_dim * attention_dim + attention_dim
    # Layer norm
    cross_att_params += 2 * attention_dim
total_params += cross_att_params
print(f"Cross-attention ({attention_layers_cross} layers): {cross_att_params:,}")

# 5. Attention pooling
att_pool_params = attention_dim * 128 + 128  # to hidden
att_pool_params += 128 * 1 + 1  # to scalar
total_params += att_pool_params
print(f"Attention pooling: {att_pool_params:,}")

# 6. MLP head
pool_output_dim = attention_dim * num_pooling_methods  # 96 * 3 = 288
pair_feature_dim = pool_output_dim * 4  # concat, product, diff = 288 * 4 = 1152

mlp_params = 0
prev_dim = pair_feature_dim
for hidden_dim in mlp_dims:
    mlp_params += prev_dim * hidden_dim + hidden_dim  # linear
    mlp_params += 2 * hidden_dim  # layer norm
    prev_dim = hidden_dim
mlp_params += prev_dim * 1 + 1  # final output
total_params += mlp_params
print(f"MLP head: {mlp_params:,}")

print("-" * 50)
print(f"Total MS-APPT parameters: {total_params:,}")
print(f"Total in millions: {total_params/1e6:.2f}M")

# Compare with original
original_8M = 8_000_000
print(f"\nReduction from original: {(1 - total_params/original_8M)*100:.1f}%")

# Memory estimate per sample (rough)
seq_len = 500  # average
batch_size = 32
bytes_per_param = 4  # float32

# Activation memory (very rough estimate)
activation_memory = seq_len * batch_size * attention_dim * bytes_per_param * 10  # rough multiplier
print(f"\nEstimated activation memory per batch: {activation_memory/1e9:.2f} GB")
print(f"Model memory: {total_params * bytes_per_param / 1e9:.3f} GB")

print("\n✓ This is a reasonable size for your dataset (~8K samples)")
print("  - Faster training with reduced attention layers")
print("  - Lower memory usage allowing larger batches")
print("  - Still sufficient capacity for the task")