# OptimizedTrainer: Advanced Training Orchestrator for Camouflaged Object Detection

## Overview

The **OptimizedTrainer** is a comprehensive training framework specifically designed for camouflaged object detection (COD) tasks. It incorporates state-of-the-art training techniques including adaptive learning rate scheduling, Mixture-of-Experts (MoE) optimization, mixed precision training, and COD-specific data augmentations.

## Key Features

### 1. Cosine Annealing with Warmup
- **Warmup Phase**: Linearly increases learning rate from `1e-6` to `1e-4` over 5 epochs
- **Cosine Decay**: Smoothly decays from max LR to min LR following cosine curve
- **Prevents Early Instability**: Warmup prevents large gradient updates at initialization

### 2. Expert Collapse Detection
- **Monitors MoE Routing**: Tracks expert utilization in Mixture-of-Experts models
- **Detects Underutilization**: Identifies when experts receive too few tokens
- **Load Imbalance Detection**: Uses Coefficient of Variation to measure imbalance
- **Confidence Tracking**: Monitors routing confidence scores

### 3. Global-Batch Load Balancing
- **Cross-Batch Statistics**: Tracks expert usage across entire training
- **L2 + CV Loss**: Combines L2 deviation and Coefficient of Variation
- **Prevents Expert Collapse**: Encourages uniform expert utilization
- **Capacity Overflow Tracking**: Monitors when experts exceed capacity

### 4. Gradient Accumulation
- **Effective Batch Size**: Simulates larger batch sizes on limited hardware
- **Memory Efficient**: Processes smaller batches with same gradient quality
- **Configurable Steps**: Set accumulation steps based on available memory

### 5. Mixed Precision Training (AMP)
- **FP16/BF16 Computation**: Reduces memory usage by 50%
- **Automatic Loss Scaling**: Prevents gradient underflow
- **2-3× Speedup**: Faster training on modern GPUs
- **Gradient Clipping**: Maintains training stability

### 6. Progressive Augmentation
- **Adaptive Strength**: Increases augmentation after epoch 20
- **Gradual Ramp-Up**: Smoothly transitions over 10 epochs
- **Prevents Overfitting**: Stronger augmentation in later epochs

### 7. COD-Specific Augmentations

#### Fourier-Based Mixing
- **Frequency Domain Blending**: Mixes images using FFT
- **Preserves High Frequencies**: Maintains texture/edge details
- **Adaptive Mixing**: Higher frequencies preserved more than low
- **Phase Mixing**: Preserves structural information

#### Contrastive Learning
- **Positive Pairs**: Same image with different augmentations
- **Color Jittering**: Brightness and contrast variations
- **Gaussian Blur**: Simulates focus variations
- **Random Flips**: Horizontal mirroring

#### Mirror Disruption
- **Breaks Symmetry**: Random horizontal/vertical/diagonal flips
- **Asymmetric Crops**: Non-centered crop-and-resize
- **Reduces Bias**: Prevents symmetry assumption in camouflage

---

## Architecture Components

### CosineAnnealingWithWarmup

```python
class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Learning rate scheduler with warmup and cosine decay.

    Schedule:
    - Epochs 0-4: Linear warmup from 1e-6 to 1e-4
    - Epochs 5+: Cosine decay from 1e-4 to 1e-6
    """
```

**Mathematical Formulation:**

Warmup phase (epochs 0 to `warmup_epochs`):
```
α = epoch / warmup_epochs
lr = min_lr + α * (max_lr - min_lr)
```

Cosine decay phase (epochs `warmup_epochs` to `total_epochs`):
```
progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
cosine_decay = 0.5 * (1 + cos(π * progress))
lr = min_lr + (max_lr - min_lr) * cosine_decay
```

**Visualization:**

```
LR
│
│     ┌──────────────────────────────────────
│    ╱                                        ╲
│   ╱                                          ╲
│  ╱                                            ╲
│ ╱                                              ╲
│╱                                                ╲___
└────────────────────────────────────────────────────→ Epoch
    5                                            100
    ↑
  Warmup                  Cosine Decay
```

### ExpertCollapseDetector

```python
class ExpertCollapseDetector:
    """
    Monitors routing statistics to detect expert collapse in MoE models.

    Collapse Indicators:
    1. Expert usage < 5% threshold
    2. Coefficient of Variation > 2.0
    3. Average routing confidence < 0.3
    """
```

**Tracked Statistics:**

1. **Expert Usage Distribution**
   ```
   usage[i] = tokens_to_expert[i] / total_tokens
   ```

2. **Load Imbalance (Coefficient of Variation)**
   ```
   CV = std(usage) / mean(usage)
   ```

3. **Routing Confidence**
   ```
   confidence = mean(max(routing_probs, dim=1))
   ```

**Example Output:**
```
Expert Collapse Detected:
  - 3/6 experts underutilized (min usage: 0.02 < 0.05)
  - High load imbalance (CV: 2.45 > 2.0)
```

### GlobalBatchLoadBalancer

```python
class GlobalBatchLoadBalancer:
    """
    Tracks expert usage across batches for MoE load balancing.

    Loss Components:
    1. L2 Loss: MSE from uniform distribution
    2. CV Loss: Squared coefficient of variation
    """
```

**Load Balance Loss:**

```python
# Target: Uniform distribution (1/num_experts for each expert)
uniform_target = torch.ones(num_experts) / num_experts

# L2 component: Squared deviation from uniform
l2_loss = MSE(expert_fraction, uniform_target)

# CV component: Penalize high variance
cv = std(expert_fraction) / mean(expert_fraction)
cv_loss = cv^2

# Combined loss
load_balance_loss = α_l2 * l2_loss + α_cv * cv_loss
```

**Default Hyperparameters:**
- `α_l2 = 0.01`
- `α_cv = 0.01`

### CODProgressiveAugmentation

```python
class CODProgressiveAugmentation:
    """
    Progressive augmentation with COD-specific techniques.

    Strength Schedule:
    - Epochs 0-19: strength = 0.3 (initial)
    - Epochs 20-29: strength = 0.3 → 0.8 (linear ramp)
    - Epochs 30+: strength = 0.8 (max)
    """
```

#### Fourier-Based Mixing

**Algorithm:**
1. Convert images to frequency domain using FFT
2. Separate amplitude and phase components
3. Create distance-based mixing mask
4. Mix amplitudes adaptively (preserve high frequencies)
5. Mix phases to maintain structure
6. Convert back to spatial domain

**Mathematical Details:**

```python
# FFT transform
FFT₁ = FFT2D(image₁)
FFT₂ = FFT2D(image₂)

# Separate components
A₁, φ₁ = |FFT₁|, ∠FFT₁  # Amplitude and phase
A₂, φ₂ = |FFT₂|, ∠FFT₂

# Adaptive mixing weight based on frequency
freq_distance = √((y - center_y)² + (x - center_x)²)
α_adaptive = α * exp(-2 * freq_distance)

# Mix components
A_mixed = α_adaptive * A₁ + (1 - α_adaptive) * A₂
φ_mixed = α_adaptive * φ₁ + (1 - α_adaptive) * φ₂

# Reconstruct
FFT_mixed = A_mixed * exp(i * φ_mixed)
image_mixed = IFFT2D(FFT_mixed)
```

**Why Fourier Mixing for COD?**
- Camouflaged objects have subtle high-frequency texture differences
- Low-frequency color blending without losing edge details
- Preserves critical camouflage patterns during augmentation

#### Contrastive Augmentation

**Generates Positive Pairs:**

1. **Anchor**: Original image (mild augmentation)
2. **Positive**: Same image with strong augmentation
   - Brightness: ±0.5 * strength
   - Contrast: ±0.5 * strength
   - Gaussian blur: σ = strength * 2.0
   - Random horizontal flip

**Use Case:**
- Self-supervised pretraining
- Metric learning
- Consistency regularization

#### Mirror Disruption

**Breaking Symmetry Assumptions:**

1. **Random Mirror Modes**:
   - Horizontal flip
   - Vertical flip
   - Both (180° rotation)
   - Diagonal (transpose if square)

2. **Asymmetric Cropping**:
   - Crop 70-95% of image (depends on strength)
   - Random non-centered position
   - Resize back to original size

**Why for COD?**
- Natural camouflage may have bilateral symmetry
- Models shouldn't rely on symmetry cues
- Improves robustness to orientation

---

## Usage Examples

### Basic Training

```python
from trainers import OptimizedTrainer
import torch
import torch.nn as nn
import torch.optim as optim

# Create model, optimizer, criterion
model = YourCODModel().cuda()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

# Create trainer
trainer = OptimizedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=torch.device('cuda'),
    accumulation_steps=4,           # 4x effective batch size
    use_amp=True,                   # Mixed precision
    total_epochs=100,
    warmup_epochs=5,
    min_lr=1e-6,
    max_lr=1e-4,
    enable_progressive_aug=True,
    aug_transition_epoch=20
)

# Training loop
for epoch in range(100):
    # Train
    train_metrics = trainer.train_epoch(
        train_loader,
        epoch=epoch,
        log_interval=10
    )

    # Validate
    val_metrics = trainer.validate(val_loader)

    # Save checkpoint
    if val_metrics['val_loss'] < best_loss:
        trainer.save_checkpoint(
            f'checkpoints/best_model.pth',
            epoch=epoch,
            metrics=val_metrics
        )
```

### Training with MoE Models

```python
# For models with Mixture-of-Experts
trainer = OptimizedTrainer(
    model=moe_model,
    optimizer=optimizer,
    criterion=criterion,
    device=torch.device('cuda'),
    num_experts=6,                    # Number of experts
    enable_load_balancing=True,       # Enable load balance loss
    enable_collapse_detection=True,   # Monitor expert collapse
    accumulation_steps=2,
    use_amp=True,
    total_epochs=100
)

# Train
for epoch in range(100):
    metrics = trainer.train_epoch(train_loader, epoch)

    # Check MoE statistics
    if metrics.get('collapse_collapsed', False):
        print(f"⚠ Expert collapse detected!")
        print(f"Reasons: {metrics['collapse_collapse_reasons']}")

    # Monitor load balance
    global_stats = trainer.load_balancer.get_global_statistics()
    print(f"Expert usage: {global_stats['global_expert_usage']}")
```

### Using Individual Augmentations

```python
from trainers.optimized_trainer import CODProgressiveAugmentation

# Create augmentation module
aug = CODProgressiveAugmentation(
    initial_strength=0.3,
    max_strength=0.8,
    transition_epoch=20,
    transition_duration=10
)

# Update for current epoch
aug.update_epoch(epoch=25)  # strength ≈ 0.55

# Apply specific augmentation
images, masks = next(iter(dataloader))

# Fourier mixing
if batch_size > 1:
    mixed_imgs, mixed_masks = aug.fourier_based_mixing(
        images[:2], images[2:4],
        masks[:2], masks[2:4],
        alpha=0.5
    )

# Contrastive pairs
anchor, positive, anchor_mask, pos_mask = aug.contrastive_augmentation(
    images, masks
)

# Mirror disruption
aug_imgs, aug_masks = aug.mirror_disruption(images, masks)

# Random augmentation
aug_imgs, aug_masks = aug.apply(images, masks, augmentation_type='random')
```

### Custom Metrics Function

```python
def compute_cod_metrics(predictions, targets):
    """Custom metrics for COD validation."""
    from metrics.cod_metrics import CODMetrics

    metrics = CODMetrics()

    # Binarize predictions
    preds_binary = (torch.sigmoid(predictions) > 0.5).float()

    # Compute metrics
    mae = torch.abs(preds_binary - targets).mean()
    iou = metrics.compute_iou(preds_binary, targets)
    f_measure = metrics.compute_f_measure(preds_binary, targets)

    return {
        'val_mae': mae.item(),
        'val_iou': iou.item(),
        'val_f_measure': f_measure.item()
    }

# Use in validation
val_metrics = trainer.validate(val_loader, metrics_fn=compute_cod_metrics)
print(f"Validation MAE: {val_metrics['val_mae']:.4f}")
print(f"Validation IoU: {val_metrics['val_iou']:.4f}")
```

### Resume Training from Checkpoint

```python
# Create trainer
trainer = OptimizedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device
)

# Load checkpoint
start_epoch = trainer.load_checkpoint('checkpoints/epoch_50.pth')

# Resume training
for epoch in range(start_epoch + 1, total_epochs):
    metrics = trainer.train_epoch(train_loader, epoch)
```

---

## Model Output Formats

The trainer supports multiple output formats:

### Format 1: Simple Tensor
```python
def forward(self, x):
    predictions = self.decode(self.encode(x))
    return predictions  # [B, 1, H, W]
```

### Format 2: Dictionary
```python
def forward(self, x):
    return {
        'predictions': main_pred,        # [B, 1, H, W]
        'aux_outputs': [aux1, aux2],     # List of auxiliary predictions
        'routing_info': {                # For MoE models
            'routing_probs': probs,      # [B, num_experts]
            'expert_assignments': assign # [B, top_k]
        }
    }
```

### Format 3: Tuple
```python
def forward(self, x):
    return (
        predictions,     # [B, 1, H, W]
        aux_outputs,     # List or None
        routing_info     # Dict or None
    )
```

---

## Hyperparameter Recommendations

### Learning Rate Schedule
| Dataset Size | Warmup LR | Max LR | Min LR | Warmup Epochs |
|-------------|-----------|--------|---------|---------------|
| Small (<1k) | 1e-6 | 5e-5 | 1e-6 | 3 |
| Medium (1-10k) | 1e-6 | 1e-4 | 1e-6 | 5 |
| Large (>10k) | 1e-6 | 2e-4 | 1e-6 | 10 |

### Gradient Accumulation
| GPU Memory | Batch Size | Accum Steps | Effective BS |
|-----------|-----------|-------------|--------------|
| 8 GB | 2 | 8 | 16 |
| 16 GB | 4 | 4 | 16 |
| 24 GB | 8 | 2 | 16 |
| 40 GB | 16 | 1 | 16 |

### Progressive Augmentation
| Training Length | Transition Epoch | Initial Strength | Max Strength |
|----------------|------------------|------------------|--------------|
| 50 epochs | 15 | 0.3 | 0.7 |
| 100 epochs | 20 | 0.3 | 0.8 |
| 200 epochs | 40 | 0.3 | 0.9 |

### MoE Load Balancing
| Model Type | α_l2 | α_cv | Collapse Threshold |
|-----------|------|------|-------------------|
| Small MoE (4 experts) | 0.01 | 0.01 | 0.10 |
| Medium MoE (6 experts) | 0.01 | 0.01 | 0.05 |
| Large MoE (8+ experts) | 0.02 | 0.02 | 0.03 |

---

## Performance Optimizations

### Mixed Precision Training
- **Memory Reduction**: ~50% lower VRAM usage
- **Speed Improvement**: 2-3× faster on Ampere GPUs (RTX 30xx, A100)
- **Minimal Accuracy Loss**: <0.1% difference with proper scaling

### Gradient Accumulation Benefits
- **Larger Effective Batch Size**: Better gradient estimates
- **Reduced Memory**: Process smaller chunks
- **Same Convergence**: Identical to large-batch training

### Progressive Augmentation Benefits
- **Early Training**: Faster convergence with weaker augmentation
- **Late Training**: Better generalization with stronger augmentation
- **Adaptive**: Automatically adjusts based on epoch

---

## Troubleshooting

### Issue: Expert Collapse Detected

**Symptoms:**
```
Expert Collapse Detected:
  - 4/6 experts underutilized (min usage: 0.01 < 0.05)
```

**Solutions:**
1. Increase load balance loss weights:
   ```python
   balancer = GlobalBatchLoadBalancer(
       num_experts=6,
       alpha_l2=0.05,  # Increased from 0.01
       alpha_cv=0.05   # Increased from 0.01
   )
   ```

2. Reduce expert capacity factor (force more sharing)

3. Add noise to routing logits:
   ```python
   routing_logits = routing_logits + 0.1 * torch.randn_like(routing_logits)
   ```

### Issue: Training Unstable with AMP

**Symptoms:**
- Loss becomes NaN
- Gradients explode
- Predictions collapse to all 0 or all 1

**Solutions:**
1. Disable AMP temporarily:
   ```python
   trainer = OptimizedTrainer(..., use_amp=False)
   ```

2. Increase gradient clipping:
   ```python
   # In OptimizedTrainer.train_epoch()
   torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
   ```

3. Use larger warmup:
   ```python
   trainer = OptimizedTrainer(..., warmup_epochs=10)
   ```

### Issue: Out of Memory

**Solutions:**
1. Increase gradient accumulation:
   ```python
   trainer = OptimizedTrainer(..., accumulation_steps=8)
   ```

2. Reduce batch size proportionally

3. Enable gradient checkpointing in model:
   ```python
   model.enable_gradient_checkpointing()
   ```

### Issue: Augmentation Too Strong/Weak

**Symptoms:**
- Model not learning (too strong)
- Overfitting (too weak)

**Solutions:**
1. Adjust strength range:
   ```python
   aug = CODProgressiveAugmentation(
       initial_strength=0.2,  # Reduced
       max_strength=0.6       # Reduced
   )
   ```

2. Change transition timing:
   ```python
   aug = CODProgressiveAugmentation(
       transition_epoch=30,    # Later
       transition_duration=20  # Slower ramp
   )
   ```

---

## Comparison with Standard Training

| Feature | Standard Training | OptimizedTrainer |
|---------|------------------|------------------|
| LR Schedule | Fixed or step decay | Warmup + cosine |
| Precision | FP32 | FP16/BF16 |
| Batch Size | Physical limit | Gradient accumulation |
| Augmentation | Fixed strength | Progressive |
| MoE Support | None | Collapse detection + balancing |
| COD Augmentations | Basic | Fourier + contrastive + mirror |
| Memory Usage | Baseline | 50-60% of baseline |
| Training Speed | Baseline | 2-3× faster |

---

## Integration with Existing Code

### Replacing Standard Training Loop

**Before:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

for epoch in range(100):
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

**After:**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = OptimizedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    use_amp=True,
    accumulation_steps=4,
    enable_progressive_aug=True
)

for epoch in range(100):
    metrics = trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.validate(val_loader)
```

**Benefits:**
- Automatic AMP management
- Gradient accumulation handled
- Progressive augmentation applied
- Learning rate warmup + cosine decay
- Checkpoint management
- MoE optimization (if applicable)

---

## Advanced Features

### Custom Augmentation Pipeline

```python
class CustomCODAugmentation(CODProgressiveAugmentation):
    def custom_camouflage_aug(self, images, masks):
        """Add custom camouflage-specific augmentation."""
        # Your custom logic here
        return aug_images, aug_masks

    def apply(self, images, masks, augmentation_type='random'):
        # Apply base augmentations
        aug_imgs, aug_masks = super().apply(images, masks, augmentation_type)

        # Apply custom augmentation
        if torch.rand(1).item() < self.current_strength * 0.3:
            aug_imgs, aug_masks = self.custom_camouflage_aug(aug_imgs, aug_masks)

        return aug_imgs, aug_masks

# Use custom augmentation
trainer = OptimizedTrainer(...)
trainer.augmentation = CustomCODAugmentation()
```

### Multi-Stage Training

```python
# Stage 1: Warmup with weak augmentation
trainer_stage1 = OptimizedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    total_epochs=30,
    enable_progressive_aug=False  # Disabled
)

for epoch in range(30):
    trainer_stage1.train_epoch(train_loader, epoch)

# Stage 2: Fine-tuning with progressive augmentation
optimizer_stage2 = torch.optim.AdamW(model.parameters(), lr=5e-5)
trainer_stage2 = OptimizedTrainer(
    model=model,
    optimizer=optimizer_stage2,
    criterion=criterion,
    device=device,
    total_epochs=70,
    enable_progressive_aug=True,
    aug_transition_epoch=10  # Relative to stage 2
)

for epoch in range(70):
    trainer_stage2.train_epoch(train_loader, epoch)
```

---

## References

### Learning Rate Schedules
- [Loshchilov & Hutter, 2016] SGDR: Stochastic Gradient Descent with Warm Restarts
- [He et al., 2019] Bag of Tricks for Image Classification with CNNs

### Mixed Precision Training
- [Micikevicius et al., 2018] Mixed Precision Training
- [NVIDIA AMP Documentation] Automatic Mixed Precision

### MoE Optimization
- [Fedus et al., 2021] Switch Transformers
- [Lepikhin et al., 2021] GShard: Scaling Giant Models

### Data Augmentation
- [Cubuk et al., 2019] AutoAugment
- [DeVries & Taylor, 2017] Improved Regularization via Cutout

### COD-Specific
- [Sun et al., 2021] Context-aware Cross-level Fusion Network (C2FNet)
- [Lv et al., 2021] Simultaneously Localize, Segment and Rank (SINet-v2)

---

## Citation

If you use OptimizedTrainer in your research, please cite:

```bibtex
@software{optimized_trainer_2024,
  title = {OptimizedTrainer: Advanced Training Framework for Camouflaged Object Detection},
  author = {CamoXpert Team},
  year = {2024},
  url = {https://github.com/your-repo/camoXpert_v2}
}
```

---

## License

This trainer is part of the CamoXpert project and is released under the same license.

---

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review test suite for usage examples

---

**Last Updated**: 2024-11-23
**Version**: 1.0.0
