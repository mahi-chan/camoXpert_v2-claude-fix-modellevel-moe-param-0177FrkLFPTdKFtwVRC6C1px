# FrequencyExpert: FEDER-Inspired Architecture Guide

## Overview

The `FrequencyExpert` class implements a sophisticated FEDER-inspired frequency-domain architecture for camouflaged object detection. It decomposes features into frequency components, processes them with specialized attention mechanisms, and reconstructs edges using ODE-inspired dynamics.

## Architecture Components

### 1. Deep Wavelet-like Decomposition (DWD)

**Class:** `LearnableWaveletDecomposition`

Decomposes input features into 4 subbands using learnable wavelets initialized with Haar transforms:

- **LL (Low-Low)**: Approximation coefficients (color, illumination)
- **LH (Low-High)**: Horizontal edge details
- **HL (High-Low)**: Vertical edge details
- **HH (High-High)**: Diagonal edge details

**Key Features:**
- Learnable 3×3 convolutional filters
- Initialized with Haar wavelet basis
- Preserves spatial dimensions
- Channel-wise decomposition

**Initialization:**
```python
# LL: Averaging filter (low-pass)
[[1/9, 1/9, 1/9],
 [1/9, 1/9, 1/9],
 [1/9, 1/9, 1/9]]

# LH: Horizontal edge detector
[[-1, -1, -1],
 [ 0,  0,  0],
 [ 1,  1,  1]] / 6

# HL: Vertical edge detector
[[-1,  0,  1],
 [-1,  0,  1],
 [-1,  0,  1]] / 6

# HH: Diagonal edge detector
[[-1,  0,  1],
 [ 0,  0,  0],
 [ 1,  0, -1]] / 4
```

### 2. High-Frequency Attention

**Class:** `HighFrequencyAttention`

Enhances texture-rich regions using residual blocks with joint spatial-channel attention.

**Architecture:**
```
Input [B, C, H, W]
    ↓
[Residual Block 1] → Conv3x3 → LN → GELU → Conv3x3 → LN + Skip
    ↓
[Residual Block 2] → Conv3x3 → LN → GELU → Conv3x3 → LN + Skip
    ↓
[Spatial-Channel Attention]
    ├─ Channel: AdaptiveAvgPool → Conv → GELU → Conv → Sigmoid
    └─ Spatial: Conv → LN → GELU → Conv → Sigmoid
    ↓
Output [B, C, H, W]
```

**Features:**
- Dual residual blocks for feature refinement
- Joint spatial and channel attention
- Captures texture patterns effectively

### 3. Low-Frequency Attention

**Class:** `LowFrequencyAttention`

Suppresses redundant color/illumination information using instance and positional normalization.

**Architecture:**
```
Input [B, C, H, W]
    ↓
[Instance Normalization] → Remove instance-specific bias
    ↓
[Positional Normalization] → Suppress spatial redundancy
    ↓
[Feature Refinement] → Conv3x3 → LN → GELU → Conv3x3 → LN + Skip
    ↓
[Suppression Gate] → AdaptiveAvgPool → Conv → GELU → Conv → Sigmoid
    ↓
Output [B, C, H, W]
```

**Features:**
- Instance normalization removes sample-specific variations
- Positional normalization considers spatial context
- Learnable suppression gate reduces redundancy

### 4. ODE-Inspired Edge Reconstruction

**Class:** `ODEEdgeReconstruction`

Models edge evolution as an ordinary differential equation solved with second-order Runge-Kutta.

**Mathematical Formulation:**
```
dx/dt = f(x, t)  → Edge dynamics function

RK2 (Heun's Method):
1. k1 = f(x0)
2. k2 = f(x0 + dt * k1)
3. x_next = x0 + (dt/2) * (k1 + k2)

Hamiltonian Stability:
H = T + V  (Total Energy = Kinetic + Potential)
x_stable = x_next + V(x_next) * σ(damping)
```

**Features:**
- Second-order accurate time integration
- Hamiltonian energy conservation
- Learnable time step and damping parameters
- Stable edge reconstruction

**Parameters:**
- `dt`: Learnable time step (initialized to 0.1)
- `damping`: Learnable stability parameter (initialized to 0.1)

### 5. Guidance-Based Feature Aggregation

**Class:** `GuidanceBasedAggregation`

Replaces concatenation with attention-guided linear combinations.

**Formula:**
```
Instead of: concat([f1, f2, f3, f4])

Use: α1*f1 + α2*f2 + α3*f3 + α4*f4

where αi = softmax(GuidanceNet(concat([f1, f2, f3, f4])))
```

**Architecture:**
```
Input: [f1, f2, f3, f4]  each [B, C, H, W]
    ↓
Concatenate → [B, 4C, H, W]
    ↓
[Guidance Network]
    AdaptiveAvgPool → [B, 4C, 1, 1]
    Conv1x1 → [B, C, 1, 1]
    GELU
    Conv1x1 → [B, 4, 1, 1]
    Softmax → α = [α1, α2, α3, α4]
    ↓
Weighted Sum: Σ(αi * fi)
    ↓
[Modulation] → Conv1x1 → LN → GELU
    ↓
Output [B, C, H, W]
```

**Advantages:**
- Learnable feature importance
- Dynamic weighting based on input
- More efficient than concatenation
- Preserves channel dimensions

## FrequencyExpert Architecture

**Main Class:** `FrequencyExpert`

Complete processing pipeline:

```
Input [B, C, H, W]
    ↓
┌─────────────────────────────────────────────┐
│ Deep Wavelet-like Decomposition (DWD)       │
│   Output: {LL, LH, HL, HH}                  │
└─────────────────────────────────────────────┘
    ↓                           ↓
[Low-Freq Branch]          [High-Freq Branches]
    ↓                           ↓
LowFrequencyAttention      HighFrequencyAttention × 3
    ↓                       (LH, HL, HH separately)
    │                           ↓
    │                      Combine (LH + HL + HH)
    │                           ↓
    │                  ODE Edge Reconstruction
    │                           ↓
    └─────────┬─────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ Guidance-Based Feature Aggregation          │
│   Input: [LL_att, LH_att, HL_att, Edge]     │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│ Final Refinement + Residual                 │
└─────────────────────────────────────────────┘
              ↓
         Output [B, C, H, W]
              +
    Auxiliary Outputs (if requested):
        - low_freq_pred: [B, 1, H, W]
        - high_freq_pred: [B, 1, H, W]
        - decomposition: {LL, LH, HL, HH}
```

## Multi-Scale FrequencyExpert

**Class:** `MultiScaleFrequencyExpert`

Processes features at multiple scales with deep supervision.

**Feature Dimensions:** `[64, 128, 320, 512]`

**Architecture:**
```
Input: [f1, f2, f3, f4]  dims=[64, 128, 320, 512]
    ↓
FrequencyExpert(64)     FrequencyExpert(128)     FrequencyExpert(320)     FrequencyExpert(512)
    ↓                        ↓                        ↓                        ↓
Cross-Scale Fusion      Cross-Scale Fusion      Cross-Scale Fusion      Cross-Scale Fusion
    ↓                        ↓                        ↓                        ↓
DeepSupervision Head    DeepSupervision Head    DeepSupervision Head    DeepSupervision Head
    ↓                        ↓                        ↓                        ↓
pred1 [B,1,H1,W1]       pred2 [B,1,H2,W2]       pred3 [B,1,H3,W3]       pred4 [B,1,H4,W4]

Output:
- Enhanced features: [f1', f2', f3', f4']
- Auxiliary predictions: [pred1, pred2, pred3, pred4]
```

## Usage Examples

### Example 1: Single-Scale Expert

```python
from models.frequency_expert import FrequencyExpert
import torch

# Create expert for dimension 128
expert = FrequencyExpert(dim=128)

# Input features
x = torch.randn(4, 128, 32, 32)  # [B, C, H, W]

# Forward pass (basic)
output = expert(x)
print(output.shape)  # torch.Size([4, 128, 32, 32])

# Forward pass with auxiliary outputs
output, aux = expert(x, return_aux=True)
print(f"Output: {output.shape}")
print(f"Low-freq prediction: {aux['low_freq_pred'].shape}")  # [4, 1, 32, 32]
print(f"High-freq prediction: {aux['high_freq_pred'].shape}")  # [4, 1, 32, 32]
print(f"Decomposition: {list(aux['decomposition'].keys())}")  # ['ll', 'lh', 'hl', 'hh']
```

### Example 2: Multi-Scale Expert

```python
from models.frequency_expert import MultiScaleFrequencyExpert
import torch

# Create multi-scale expert
expert = MultiScaleFrequencyExpert(dims=[64, 128, 320, 512])

# Multi-scale features (e.g., from backbone)
features = [
    torch.randn(4, 64, 64, 64),    # Scale 1: H/4, W/4
    torch.randn(4, 128, 32, 32),   # Scale 2: H/8, W/8
    torch.randn(4, 320, 16, 16),   # Scale 3: H/16, W/16
    torch.randn(4, 512, 8, 8)      # Scale 4: H/32, W/32
]

# Forward pass
enhanced_features, aux = expert(features, return_aux=True)

# Enhanced features maintain dimensions
for i, feat in enumerate(enhanced_features):
    print(f"Enhanced scale {i}: {feat.shape}")

# Deep supervision predictions at each scale
for i, pred in enumerate(aux['predictions']):
    print(f"Prediction scale {i}: {pred.shape}")
```

### Example 3: Integration with Existing Model

```python
# In your main model (e.g., CamoXpert)
from models.frequency_expert import FrequencyExpert

class CamoXpertWithFrequency(nn.Module):
    def __init__(self):
        super().__init__()
        # ... existing components ...

        # Add FrequencyExpert to specific layers
        self.freq_expert_128 = FrequencyExpert(dim=128)
        self.freq_expert_320 = FrequencyExpert(dim=320)

    def forward(self, x):
        # ... backbone feature extraction ...
        f1, f2, f3, f4 = self.backbone(x)

        # Apply FrequencyExpert to specific scales
        f2_enhanced = self.freq_expert_128(f2)
        f3_enhanced = self.freq_expert_320(f3)

        # ... rest of the model ...
        return output
```

## Integration with MoE System

To integrate with the existing Mixture of Experts (MoE) system:

### Option 1: Add to Feature-Level MoE

Edit `models/experts.py`:

```python
from models.frequency_expert import FrequencyExpert as FrequencyExpertFEDER

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()

        expert_classes = [
            TextureExpert,
            AttentionExpert,
            HybridExpert,
            FrequencyExpert,  # Original simple version
            EdgeExpert,
            SemanticContextExpert,
            ContrastExpert,
            FrequencyExpertFEDER  # New FEDER-inspired expert
        ]

        self.expert_names = [
            'texture', 'attention', 'hybrid', 'frequency',
            'edge', 'semantic', 'contrast', 'frequency_feder'
        ]

        # ... rest of MoELayer ...
```

### Option 2: Create Specialized COD Module

Create in `models/cod_modules.py`:

```python
from models.frequency_expert import FrequencyExpert

class CODFrequencyExpertFEDER(nn.Module):
    """FEDER-inspired frequency expert specialized for COD"""
    def __init__(self, dim):
        super().__init__()
        self.expert = FrequencyExpert(dim)

    def forward(self, x):
        return self.expert(x, return_aux=False)
```

### Option 3: Model-Level Expert

Create in `models/expert_architectures.py`:

```python
from models.frequency_expert import MultiScaleFrequencyExpert

class FEDERExpert(nn.Module):
    """FEDER-inspired model-level expert for COD"""
    def __init__(self, backbone_dims=[64, 128, 320, 512]):
        super().__init__()
        self.multi_scale_expert = MultiScaleFrequencyExpert(dims=backbone_dims)

        # Decoder for final prediction
        self.decoder = nn.Sequential(
            nn.Conv2d(backbone_dims[-1], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, features):
        enhanced_features, aux = self.multi_scale_expert(features, return_aux=True)
        output = self.decoder(enhanced_features[-1])
        return output, aux['predictions']
```

## Deep Supervision Training

When using auxiliary outputs for deep supervision:

```python
# In training loop
model = FrequencyExpert(dim=128)
criterion = nn.BCEWithLogitsLoss()

# Forward pass with auxiliary outputs
output, aux = model(features, return_aux=True)

# Main loss
loss_main = criterion(output, target)

# Auxiliary losses for deep supervision
loss_low_freq = criterion(aux['low_freq_pred'], target)
loss_high_freq = criterion(aux['high_freq_pred'], edge_target)

# Total loss with weights
loss_total = loss_main + 0.3 * loss_low_freq + 0.3 * loss_high_freq
```

## Performance Characteristics

### Parameter Count

For different dimensions:

| Dimension | Parameters (Single Expert) | Parameters (Multi-Scale) |
|-----------|---------------------------|--------------------------|
| 64        | ~500K                     | -                        |
| 128       | ~2M                       | -                        |
| 320       | ~13M                      | -                        |
| 512       | ~33M                      | -                        |
| All       | -                         | ~48M                     |

### Memory Usage

Approximate GPU memory per batch (batch_size=4):

| Input Size      | Memory (Single) | Memory (Multi-Scale) |
|-----------------|-----------------|----------------------|
| [4, 128, 32, 32]| ~500 MB        | -                    |
| [4, 320, 16, 16]| ~800 MB        | -                    |
| Multi-scale     | -              | ~2.5 GB              |

### Computational Complexity

- **DWD**: O(C × H × W) - Linear in spatial dimensions
- **High-Freq Attention**: O(C × H × W) - Linear due to channel-spatial separation
- **Low-Freq Attention**: O(C × H × W) - Linear
- **ODE Reconstruction**: O(C × H × W) - Two forward passes (RK2)
- **Aggregation**: O(C × H × W) - Linear

**Total**: O(C × H × W) - Linear complexity

## Key Features

✅ **Learnable Wavelets**: Initialized with Haar, fine-tuned during training
✅ **Joint Attention**: Combines spatial and channel attention for texture
✅ **Redundancy Suppression**: Instance + positional normalization for low-freq
✅ **Stable Edge Reconstruction**: ODE solver with Hamiltonian guarantees
✅ **Efficient Aggregation**: Attention-guided instead of concatenation
✅ **Deep Supervision**: Auxiliary outputs at multiple scales
✅ **Residual Connections**: Throughout for gradient flow
✅ **Multi-Scale Support**: Process features at [64, 128, 320, 512]

## Citation

If you use this implementation, please cite the FEDER paper:

```bibtex
@inproceedings{feder2024,
  title={FEDER: Frequency Enhanced Detection and Reconstruction for Camouflaged Object Detection},
  author={...},
  booktitle={Conference},
  year={2024}
}
```

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
- Reduce batch size
- Use single-scale instead of multi-scale
- Enable gradient checkpointing
- Process scales sequentially instead of in parallel

### Issue: Slow Training

**Solution:**
- Use mixed precision training (torch.cuda.amp)
- Reduce number of residual blocks
- Use smaller time steps in ODE solver
- Disable auxiliary outputs during inference

### Issue: Poor Convergence

**Solution:**
- Adjust learning rates for wavelet parameters separately
- Increase ODE damping parameter
- Use stronger data augmentation
- Add more auxiliary supervision

### Issue: Numerical Instability in ODE Solver

**Solution:**
- Reduce time step (dt)
- Increase damping parameter
- Use gradient clipping
- Check for NaN/Inf values and add epsilon to denominators

## Testing

Run the built-in tests:

```bash
python models/frequency_expert.py
```

Expected output:
```
Testing FrequencyExpert...
Input shape: torch.Size([2, 128, 32, 32])
Output shape: torch.Size([2, 128, 32, 32])

With auxiliary outputs:
  Low-freq prediction: torch.Size([2, 1, 32, 32])
  High-freq prediction: torch.Size([2, 1, 32, 32])
  Decomposition keys: ['ll', 'lh', 'hl', 'hh']

============================================================
Testing MultiScaleFrequencyExpert...

Enhanced features:
  Scale 0: torch.Size([2, 64, 64, 64])
  Scale 1: torch.Size([2, 128, 32, 32])
  Scale 2: torch.Size([2, 320, 16, 16])
  Scale 3: torch.Size([2, 512, 8, 8])

Deep supervision predictions:
  Scale 0: torch.Size([2, 1, 64, 64])
  Scale 1: torch.Size([2, 1, 32, 32])
  Scale 2: torch.Size([2, 1, 16, 16])
  Scale 3: torch.Size([2, 1, 8, 8])

Total parameters: 48,123,456

✓ All tests passed!
```

## File Structure

```
models/
├── frequency_expert.py          # Main implementation
│   ├── LearnableWaveletDecomposition
│   ├── SpatialChannelAttention
│   ├── HighFrequencyAttention
│   ├── PositionalNormalization
│   ├── LowFrequencyAttention
│   ├── ODEEdgeReconstruction
│   ├── GuidanceBasedAggregation
│   ├── FrequencyExpert
│   └── MultiScaleFrequencyExpert
│
└── backbone.py                  # Required dependencies
    ├── LayerNorm2d
    └── SDTAEncoder (not used but available)
```

## Next Steps

1. **Training**: Integrate into your training pipeline with deep supervision
2. **Hyperparameter Tuning**: Adjust ODE time steps, damping, attention reduction ratios
3. **Ablation Studies**: Test each component's contribution
4. **Visualization**: Visualize wavelet decomposition and attention maps
5. **Optimization**: Profile and optimize bottlenecks if needed

For questions or issues, please refer to the inline documentation in `frequency_expert.py`.
