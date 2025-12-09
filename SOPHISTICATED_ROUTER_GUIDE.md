# Sophisticated Router - Upgrade Guide

## Overview

The upgraded **SophisticatedRouter** is an advanced routing mechanism for Mixture of Experts (MoE) systems, featuring comprehensive feature analysis and dual routing modes: traditional **Token Choice** and the novel **Expert Choice** routing.

## Architecture Overview

```
Input Features [f1, f2, f3, f4]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Analysis Branches                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. Texture Complexity  → Multi-scale Dilations [1, 2, 4]        │
│ 2. Edge Density        → Sobel-initialized Kernels (H, V, D1, D2)│
│ 3. Frequency Content   → FFT Analysis (Low vs High Freq)        │
│ 4. Context Scale       → Pyramid Pooling [1x1, 2x2, 3x3, 6x6]  │
│ 5. Uncertainty         → Monte Carlo Dropout + Confidence       │
└─────────────────────────────────────────────────────────────────┘
    ↓
[Feature Fusion: 1280 dims] → [Decision Network] → [Router Logits]
    ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Routing Mechanism                          │
├─────────────────────────────────────────────────────────────────┤
│ Token Choice:  Tokens select top-k experts                     │
│ Expert Choice: Experts select top-capacity tokens              │
└─────────────────────────────────────────────────────────────────┘
    ↓
[Expert Assignments + Load Balancing Loss]
```

## Key Components

### 1. Multi-Scale Texture Analyzer

**Purpose**: Capture texture patterns at different scales using dilated convolutions.

**Architecture**:
```python
Input [B, 512, H, W]
    ↓
Parallel Branches:
    ├─ Dilation 1 (fine texture)   → [B, 85, H, W]
    ├─ Dilation 2 (medium texture) → [B, 85, H, W]
    └─ Dilation 4 (coarse texture) → [B, 86, H, W]
    ↓
Concatenate → [B, 256, H, W]
    ↓
Fusion Conv → AdaptiveAvgPool → [B, 256]
```

**Key Features**:
- **Dilation rate 1**: Captures fine-grained texture (adjacent pixels)
- **Dilation rate 2**: Captures medium-scale patterns (2-pixel spacing)
- **Dilation rate 4**: Captures coarse patterns (4-pixel spacing)
- Excellent for detecting camouflage texture complexity

**Applications**:
- Distinguish fine vs coarse camouflage patterns
- Identify texture-rich regions requiring specialized experts
- Route to texture-focused experts (e.g., TextureExpert, CODTextureExpert)

---

### 2. Sobel Edge Analyzer

**Purpose**: Detect edge density and direction using Sobel-initialized kernels.

**Architecture**:
```python
Input [B, 512, H, W]
    ↓
Parallel Edge Detection (Sobel-initialized):
    ├─ Horizontal Sobel (vertical edges)   → [B, 64, H, W]
    ├─ Vertical Sobel (horizontal edges)   → [B, 64, H, W]
    ├─ Diagonal Sobel 1 (↘ edges)          → [B, 64, H, W]
    └─ Diagonal Sobel 2 (↙ edges)          → [B, 64, H, W]
    ↓
Concatenate → [B, 256, H, W]
    ↓
Post-process → AdaptiveAvgPool → [B, 256]
```

**Sobel Kernels**:
```python
# Horizontal (detects vertical edges)
[[-1,  0,  1],
 [-2,  0,  2],
 [-1,  0,  1]] / 8

# Vertical (detects horizontal edges)
[[-1, -2, -1],
 [ 0,  0,  0],
 [ 1,  2,  1]] / 8

# Diagonal 1 (↘)
[[-2, -1,  0],
 [-1,  0,  1],
 [ 0,  1,  2]] / 8

# Diagonal 2 (↙)
[[ 0, -1, -2],
 [ 1,  0, -1],
 [ 2,  1,  0]] / 8
```

**Key Features**:
- Kernels initialized with classical Sobel operators
- Remain learnable during training (can adapt to camouflage-specific edges)
- Multi-directional edge detection (4 directions)
- Captures edge density and boundary characteristics

**Applications**:
- Detect sharp vs smooth boundaries
- Route to edge-focused experts (e.g., EdgeExpert, BoundaryExpert)
- Identify boundary-ambiguous regions requiring specialized processing

---

### 3. FFT Frequency Analyzer

**Purpose**: Analyze frequency spectrum to distinguish high vs low frequency dominance.

**Architecture**:
```python
Input [B, 512, H, W]
    ↓
Pre-process → [B, 128, H, W]
    ↓
2D FFT (rfft2) → [B, 128, H, W//2+1] (complex)
    ↓
Magnitude Spectrum → [B, 128, H, W//2+1]
    ↓
Frequency Separation:
    ├─ Low-freq mask (center region)  → [B, 128, H, W//2+1]
    └─ High-freq mask (outer region)  → [B, 128, H, W//2+1]
    ↓
Separate Processing:
    ├─ Low-freq network  → [B, 128, H, W]
    └─ High-freq network → [B, 128, H, W]
    ↓
Concatenate → [B, 256, H, W]
    ↓
Post-process → [B, 256]
```

**Frequency Decomposition**:
```
FFT Spectrum (frequency domain):
┌─────────────────────────────┐
│ Low Freq  │                 │
│  (DC +    │                 │
│   smooth) │   High Freq     │
│           │   (edges +      │
│           │    texture)     │
└─────────────────────────────┘
```

**Key Features**:
- Real FFT (rfft2) for efficiency
- Separates low-frequency (smooth regions, color) from high-frequency (edges, texture)
- Magnitude spectrum analysis (phase-invariant)
- Learnable processing of frequency components

**Applications**:
- Detect high-frequency texture vs low-frequency smooth regions
- Route to frequency-domain experts (FrequencyExpert, FEDER)
- Identify frequency-dominant characteristics

---

### 4. Pyramid Pooling Analyzer

**Purpose**: Capture context at multiple spatial scales.

**Architecture**:
```python
Input [B, 512, H, W]
    ↓
Parallel Pyramid Levels:
    ├─ Pool to 1x1  → Conv → Upsample to [B, 64, H, W]
    ├─ Pool to 2x2  → Conv → Upsample to [B, 64, H, W]
    ├─ Pool to 3x3  → Conv → Upsample to [B, 64, H, W]
    └─ Pool to 6x6  → Conv → Upsample to [B, 64, H, W]
    ↓
Concatenate → [B, 256, H, W]
    ↓
Fusion → AdaptiveAvgPool → [B, 256]
```

**Pooling Scales**:
- **1x1**: Global context (entire image)
- **2x2**: Quarter-image context
- **3x3**: Ninth-image context
- **6x6**: Fine-grained spatial context

**Key Features**:
- Multi-scale spatial aggregation
- Captures both global and local context
- Inspired by PSPNet (Pyramid Scene Parsing)
- Bilinear upsampling to preserve spatial information

**Applications**:
- Detect object scale (small vs large camouflaged objects)
- Route to context-aware experts (SemanticContextExpert)
- Identify scale-dependent characteristics

---

### 5. Uncertainty Estimator

**Purpose**: Estimate routing confidence using Monte Carlo Dropout.

**Architecture**:
```python
Input [B, 512, H, W]
    ↓
Feature Network (with Dropout):
    Conv → BN → ReLU → Dropout2d(0.3)
    ↓
    Conv → BN → ReLU → Dropout2d(0.3)
    ↓
    AdaptiveAvgPool → [B, 128]
    ↓
Monte Carlo Sampling (N=5 forward passes):
    ├─ Sample 1: [B, 128]
    ├─ Sample 2: [B, 128]
    ├─ ...
    └─ Sample N: [B, 128]
    ↓
Compute Statistics:
    ├─ Mean: [B, 128]
    └─ Variance: [B, 128]
    ↓
Uncertainty = Mean(Variance) → [B, 1]
Confidence = exp(-Uncertainty) → [B, 1]
    ↓
Uncertainty Head (MLP):
    Linear → ReLU → Linear → Sigmoid → [B, 1]
    ↓
Final Confidence = (MC Confidence + Predicted Confidence) / 2
```

**Monte Carlo Dropout**:
```
Multiple Forward Passes with Dropout:

Pass 1: [x] → [Dropout] → [y1]
Pass 2: [x] → [Dropout] → [y2]  (different dropout mask)
Pass 3: [x] → [Dropout] → [y3]  (different dropout mask)
...

Variance across passes = Epistemic Uncertainty
```

**Key Features**:
- **Epistemic uncertainty**: Model uncertainty (what the model doesn't know)
- **Aleatoric uncertainty**: Data uncertainty (inherent noise)
- Spatial Dropout2d for better feature uncertainty estimation
- Dual confidence estimation (MC + learned predictor)

**Applications**:
- Weight routing decisions by confidence
- Uncertain samples → smoother routing (distribute across experts)
- Confident samples → sharper routing (focus on best expert)
- Monitor routing quality

---

### 6. Expert Choice Routing

**Purpose**: Novel routing paradigm where experts choose tokens instead of tokens choosing experts.

**Comparison**:

| Aspect | Token Choice | Expert Choice |
|--------|--------------|---------------|
| Selection | Each token selects top-k experts | Each expert selects top-capacity tokens |
| Load Balancing | Uneven (popular experts overloaded) | Even (each expert processes exact capacity) |
| Expert Collapse | Possible (unused experts) | Prevented (all experts used) |
| Efficiency | Lower (variable expert load) | Higher (fixed expert capacity) |
| Batch Scaling | Worse (imbalance grows) | Better (capacity scales with batch) |

**Token Choice Flow**:
```
Token 1 → [Router] → Select Experts [0, 2]     (45%, 55%)
Token 2 → [Router] → Select Experts [1, 2]     (60%, 40%)
Token 3 → [Router] → Select Experts [0, 3]     (70%, 30%)
...

Result: Expert 2 may be overloaded, Expert 1 underused
```

**Expert Choice Flow**:
```
Expert 0 → [Router Scores] → Select top-capacity tokens [1, 3, 5, ...]
Expert 1 → [Router Scores] → Select top-capacity tokens [2, 4, 6, ...]
Expert 2 → [Router Scores] → Select top-capacity tokens [7, 8, 9, ...]
Expert 3 → [Router Scores] → Select top-capacity tokens [10, 11, 12, ...]

Result: All experts process exactly 'capacity' tokens (balanced)
```

**Algorithm**:
```python
1. Compute router scores: [num_tokens, num_experts]
2. Transpose: [num_experts, num_tokens]
3. For each expert:
   - Select top-capacity tokens by score
   - Normalize weights for selected tokens
4. Result: [num_experts, capacity] assignments
```

**Capacity Calculation**:
```python
capacity = (num_tokens * capacity_factor) / num_experts

# Example:
# - num_tokens = 1000
# - num_experts = 4
# - capacity_factor = 1.25
# → capacity = (1000 * 1.25) / 4 = 312 tokens per expert
```

**Key Features**:
- **Perfect load balancing**: Each expert processes exactly `capacity` tokens
- **No expert collapse**: All experts guaranteed usage
- **Scalable**: Capacity grows proportionally with batch size
- **Efficient**: Fixed expert load enables better parallelization

**Applications**:
- Large-batch training (better load distribution)
- Preventing expert collapse in long training runs
- Systems where expert balance is critical
- High-throughput inference

---

### 7. Global-Batch Load Balancing

**Purpose**: Prevent expert collapse by encouraging uniform expert usage.

**Loss Formulation**:
```python
# Average probability for each expert across batch
mean_probs = expert_probs.mean(dim=0)  # [num_experts]

# Ideal uniform distribution
ideal_prob = 1.0 / num_experts

# L2 distance from uniform
L2_loss = sum((mean_probs - ideal_prob)^2)

# Coefficient of variation (relative std)
CV = std(mean_probs) / mean(mean_probs)

# Total load balance loss
load_balance_loss = coef * (L2_loss + CV)
```

**Expert Collapse Detection**:
```
Good Distribution (balanced):
Expert 0: 25%  ████████
Expert 1: 24%  ████████
Expert 2: 26%  ████████
Expert 3: 25%  ████████

Bad Distribution (collapsed):
Expert 0: 60%  ████████████████████
Expert 1:  5%  ██
Expert 2: 30%  ██████████
Expert 3:  5%  ██
```

**Metrics**:
- **L2 distance**: Penalizes deviation from uniform (25%, 25%, 25%, 25%)
- **Coefficient of Variation**: Penalizes relative spread
- **Entropy**: Measures routing diversity (higher = more diverse)

**Key Features**:
- Global batch statistics (not per-sample)
- Dual regularization (L2 + CV)
- Learnable coefficient (default: 0.01)
- Computed for both Token Choice and Expert Choice modes

**Applications**:
- Prevent expert specialization on few samples
- Ensure all experts contribute to learning
- Maintain diverse expert behaviors
- Monitor training health

---

## Usage Examples

### Example 1: Basic Token Choice Routing

```python
from models.sophisticated_router import SophisticatedRouter
import torch

# Create router
router = SophisticatedRouter(
    backbone_dims=[64, 128, 320, 512],
    num_experts=4,
    top_k=2,
    routing_mode='token_choice'
)

# Backbone features
features = [
    torch.randn(8, 64, 112, 112),   # 1/4 resolution
    torch.randn(8, 128, 56, 56),    # 1/8 resolution
    torch.randn(8, 320, 28, 28),    # 1/16 resolution
    torch.randn(8, 512, 14, 14)     # 1/32 resolution
]

# Route
expert_probs, top_k_indices, top_k_weights, aux = router(features)

# Use routing
for i in range(8):
    selected_experts = top_k_indices[i]  # [2] expert indices
    weights = top_k_weights[i]           # [2] weights (sum to 1)

    print(f"Sample {i}:")
    print(f"  Expert {selected_experts[0]}: {weights[0]:.2%}")
    print(f"  Expert {selected_experts[1]}: {weights[1]:.2%}")
    print(f"  Confidence: {aux['confidence'][i].item():.3f}")
```

### Example 2: Expert Choice Routing

```python
router = SophisticatedRouter(
    backbone_dims=[64, 128, 320, 512],
    num_experts=4,
    routing_mode='expert_choice',
    expert_capacity_factor=1.25
)

logits, expert_assignments, expert_weights, aux = router(features)

# Expert assignments shape: [num_experts, capacity]
# Each row contains token indices assigned to that expert

for expert_id in range(4):
    token_indices = expert_assignments[expert_id]  # [capacity]
    weights = expert_weights[expert_id]            # [capacity]
    mask = aux['expert_mask'][expert_id]           # [capacity]

    # Get actual tokens (excluding padding)
    actual_tokens = token_indices[mask == 1]

    print(f"Expert {expert_id} processes {len(actual_tokens)} tokens")
```

### Example 3: Dynamic Mode Switching

```python
# Start with Token Choice
router = SophisticatedRouter(routing_mode='token_choice')

# Training with Token Choice
for epoch in range(10):
    output = router(features)
    # ... training ...

# Switch to Expert Choice for better load balancing
router.set_routing_mode('expert_choice')

# Continue training with Expert Choice
for epoch in range(10, 20):
    output = router(features)
    # ... training ...
```

### Example 4: Monitoring Routing Health

```python
expert_probs, _, _, aux = router(features)

# Get usage statistics
stats = router.get_expert_usage_stats(expert_probs)

print("Routing Health Metrics:")
print(f"  Average expert probs: {stats['avg_expert_probs']}")
print(f"  Entropy: {stats['entropy']:.3f}")  # Higher = more diverse
print(f"  CV: {stats['coefficient_of_variation']:.3f}")  # Lower = more balanced
print(f"  Max prob: {stats['max_prob']:.2%}")
print(f"  Min prob: {stats['min_prob']:.2%}")

# Check for expert collapse
if stats['min_prob'] < 0.05:  # Less than 5%
    print("⚠ Warning: Potential expert collapse detected!")
```

### Example 5: Training with Load Balancing Loss

```python
# Forward pass
expert_probs, top_k_indices, top_k_weights, aux = router(features)

# Main task loss (e.g., segmentation)
output = model(features, top_k_indices, top_k_weights)
task_loss = criterion(output, target)

# Load balancing loss (from router)
load_balance_loss = aux['load_balance_loss']

# Total loss
total_loss = task_loss + load_balance_loss

# Backward
total_loss.backward()
optimizer.step()

# Log
print(f"Task loss: {task_loss.item():.4f}")
print(f"Load balance loss: {load_balance_loss.item():.6f}")
```

## Integration with Model-Level MoE

### Current Integration

In `models/model_level_moe.py`:

```python
from models.sophisticated_router import SophisticatedRouter

class ModelLevelMoE(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone
        self.backbone = create_backbone()

        # Router (upgraded)
        self.router = SophisticatedRouter(
            backbone_dims=[64, 128, 320, 512],
            num_experts=4,
            top_k=2,
            routing_mode='token_choice'  # or 'expert_choice'
        )

        # Expert models
        self.experts = nn.ModuleList([
            SINetExpert(),
            PraNetExpert(),
            ZoomNetExpert(),
            UJSCExpert()
        ])

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Route
        expert_probs, top_k_indices, top_k_weights, aux = self.router(features)

        # Ensemble expert outputs
        batch_outputs = []
        for b in range(x.shape[0]):
            # Get selected experts for this sample
            selected = top_k_indices[b]  # [top_k]
            weights = top_k_weights[b]   # [top_k]

            # Run selected experts
            sample_output = 0
            for idx, weight in zip(selected, weights):
                expert_out = self.experts[idx](features)
                sample_output += weight * expert_out[b:b+1]

            batch_outputs.append(sample_output)

        output = torch.cat(batch_outputs, dim=0)

        return output, aux['load_balance_loss']
```

### With Expert Choice

```python
class ModelLevelMoEExpertChoice(nn.Module):
    def forward(self, x):
        features = self.backbone(x)

        # Expert Choice routing
        logits, expert_assignments, expert_weights, aux = self.router(features)

        # Process tokens by expert
        # Reshape features to token format
        B, C, H, W = features[-1].shape
        tokens = features[-1].flatten(2).transpose(1, 2)  # [B, H*W, C]
        tokens = tokens.reshape(-1, C)  # [B*H*W, C]

        # Process each expert
        expert_outputs = []
        for expert_id, expert in enumerate(self.experts):
            # Get assigned tokens
            token_indices = expert_assignments[expert_id]
            weights = expert_weights[expert_id]
            mask = aux['expert_mask'][expert_id]

            # Select tokens
            selected_tokens = tokens[token_indices[mask == 1]]

            # Process with expert
            expert_out = expert(selected_tokens)

            # Weight outputs
            expert_out = expert_out * weights[mask == 1].unsqueeze(-1)

            expert_outputs.append(expert_out)

        # Combine outputs (scatter back to original positions)
        # ... implementation depends on your expert architecture ...

        return output, aux['load_balance_loss']
```

## Performance Characteristics

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| Texture Analyzer | ~1.5M |
| Edge Analyzer | ~1.8M |
| Frequency Analyzer | ~1.2M |
| Context Analyzer | ~1.5M |
| Uncertainty Estimator | ~0.4M |
| Multi-scale Integration | ~0.5M |
| Decision Network | ~2.5M |
| **Total** | **~9.4M** |

### Memory Usage

Approximate GPU memory per batch (batch_size=8):

| Feature Resolution | Memory |
|-------------------|--------|
| 512 @ 14×14 | ~150 MB |
| Intermediate (all branches) | ~400 MB |
| Router overhead | ~50 MB |
| **Total** | **~600 MB** |

### Computational Complexity

| Branch | Complexity | Notes |
|--------|-----------|-------|
| Texture | O(C×H×W) | 3 dilated convs |
| Edge | O(C×H×W) | 4 Sobel convs |
| Frequency | O(C×H×W×log(HW)) | FFT overhead |
| Context | O(C×H×W) | 4 pyramid levels |
| Uncertainty | O(C×H×W×N) | N MC samples |
| **Total** | **O(C×H×W×log(HW))** | FFT dominates |

## Advantages Over Previous Router

| Feature | Old Router | New Router |
|---------|-----------|------------|
| Texture Analysis | Basic dilations | Multi-scale [1,2,4] |
| Edge Detection | Generic conv | Sobel-initialized |
| Frequency | Conv approximation | True FFT analysis |
| Context | Single-scale | Pyramid [1,2,3,6] |
| Uncertainty | None | MC Dropout |
| Routing | Token Choice only | Both modes |
| Load Balancing | Basic | Global-batch + CV |
| Parameters | ~8M | ~9.4M |

## Troubleshooting

### Issue: High load balance loss

**Symptoms**: Load balance loss > 0.1
**Causes**:
- Expert collapse (one expert dominates)
- Insufficient training
- Too high temperature parameter

**Solutions**:
```python
# Increase load balance coefficient
router.load_balance_coef = 0.05  # default: 0.01

# Check temperature
print(router.temperature)  # Should be in [0.1, 5.0]

# Monitor expert usage
stats = router.get_expert_usage_stats(expert_probs)
print(stats['min_prob'])  # Should be > 0.05
```

### Issue: Out of Memory with FFT

**Symptoms**: CUDA OOM during frequency analysis
**Causes**:
- Large feature maps (high resolution)
- FFT allocates temporary buffers

**Solutions**:
```python
# Reduce channels before FFT
router.frequency_analyzer.pre_process[0].out_channels = 64  # default: 128

# Use lower resolution features
features_for_routing = [f[::2, ::2] for f in features]  # Downsample 2x
```

### Issue: Uncertain routing (high uncertainty values)

**Symptoms**: Uncertainty > 0.5 for most samples
**Causes**:
- Model undertrained
- Too much dropout
- Difficult samples

**Solutions**:
```python
# Reduce dropout
router.uncertainty_estimator.feature_net[3].p = 0.1  # default: 0.3

# Fewer MC samples
confidence, uncertainty = router.uncertainty_estimator(x, num_samples=3)  # default: 5

# Check if samples are genuinely ambiguous
# (high uncertainty might be correct!)
```

### Issue: Expert Choice imbalance

**Symptoms**: Some experts get 0 tokens
**Causes**:
- Capacity too low
- Router hasn't learned proper scoring

**Solutions**:
```python
# Increase capacity factor
router.expert_capacity_factor = 1.5  # default: 1.25

# Check capacity calculation
num_tokens = B * H * W
capacity = num_tokens * 1.5 / num_experts
print(f"Capacity per expert: {capacity}")

# Ensure capacity > 0 for all experts
```

## Best Practices

1. **Start with Token Choice**: Easier to debug and understand
2. **Monitor routing entropy**: Should be > 1.0 for balanced routing
3. **Use load balancing loss**: Essential for preventing collapse
4. **Tune temperature**: Lower for specialized routing, higher for exploration
5. **Check uncertainty**: High uncertainty = hard samples (might need ensemble)
6. **Visualize expert usage**: Plot avg_expert_probs over time
7. **Switch to Expert Choice**: After initial training, for better balance
8. **Validate both modes**: Ensure performance similar in both modes

## Advanced Features

### Custom Capacity Strategy

```python
class AdaptiveCapacityRouter(SophisticatedRouter):
    def forward(self, features):
        # Compute base routing
        output = super().forward(features)

        # Adjust capacity based on uncertainty
        if self.routing_mode == 'expert_choice':
            _, _, _, aux = output
            uncertainty = aux['uncertainty'].mean()

            # High uncertainty → increase capacity (more ensemble)
            if uncertainty > 0.3:
                self.expert_choice_router.expert_capacity_factor = 1.5
            else:
                self.expert_choice_router.expert_capacity_factor = 1.0

        return output
```

### Confidence-Weighted Ensemble

```python
def ensemble_with_confidence(expert_outputs, weights, confidence):
    """
    Weight expert outputs by both routing weights and confidence.

    Args:
        expert_outputs: List of [B, C, H, W] tensors
        weights: [B, top_k] routing weights
        confidence: [B, 1] confidence scores
    """
    # Multiply routing weights by confidence
    adjusted_weights = weights * confidence  # [B, top_k]

    # Renormalize
    adjusted_weights = adjusted_weights / adjusted_weights.sum(dim=1, keepdim=True)

    # Weighted sum
    output = sum(w.unsqueeze(-1).unsqueeze(-1) * out
                 for w, out in zip(adjusted_weights.t(), expert_outputs))

    return output
```

### Feature Analysis Visualization

```python
def visualize_router_features(router, features):
    """Visualize what each branch is detecting"""
    with torch.no_grad():
        # Extract branch features
        texture_feat = router.texture_analyzer(features[-1])
        edge_feat = router.edge_analyzer(features[-1])
        freq_feat = router.frequency_analyzer(features[-1])
        context_feat = router.context_analyzer(features[-1])

        # Plot feature distributions
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].hist(texture_feat.cpu().numpy().flatten(), bins=50)
        axes[0, 0].set_title('Texture Features')

        axes[0, 1].hist(edge_feat.cpu().numpy().flatten(), bins=50)
        axes[0, 1].set_title('Edge Features')

        axes[1, 0].hist(freq_feat.cpu().numpy().flatten(), bins=50)
        axes[1, 0].set_title('Frequency Features')

        axes[1, 1].hist(context_feat.cpu().numpy().flatten(), bins=50)
        axes[1, 1].set_title('Context Features')

        plt.tight_layout()
        plt.savefig('router_features.png')
```

## Citation

If you use this router in your research, please cite:

```bibtex
@software{sophisticated_router_2024,
  title={Sophisticated Router: Advanced Feature Analysis and Expert Choice Routing for MoE Systems},
  author={CamoXpert Team},
  year={2024},
  url={https://github.com/mahi-chan/camoXpert_v2}
}
```

## References

- **Expert Choice**: [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)
- **Pyramid Pooling**: [Pyramid Scene Parsing Network (PSPNet)](https://arxiv.org/abs/1612.01105)
- **Sobel Edge Detection**: Classical Computer Vision
- **FFT Analysis**: [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/abs/2006.10739)
- **MC Dropout**: [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142)

---

**Last Updated**: 2024
**Version**: 2.0 (Upgraded)
