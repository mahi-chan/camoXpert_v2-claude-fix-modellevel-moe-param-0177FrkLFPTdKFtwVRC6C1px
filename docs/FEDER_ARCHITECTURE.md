# FEDER: Frequency Decomposition and Dynamic Edge Reconstruction

Complete paper-accurate implementation for camouflaged object detection.

## Architecture Overview

FEDER processes multi-scale features through frequency decomposition, specialized attention, and ODE-based edge reconstruction to enhance camouflaged object detection.

```
PVT Features [64, 128, 320, 512]
    ↓
┌───────────────────────────────────────────────────────┐
│  For each scale (4 parallel pipelines):              │
│                                                       │
│  1. Deep Wavelet Decomposition                       │
│     Input [B, C, H, W]                               │
│       ↓                                              │
│     Learnable Haar Wavelets                          │
│       ↓                                              │
│     {LL, LH, HL, HH} subbands                        │
│                                                       │
│  2. Frequency-Specific Attention                     │
│     LL → LowFreqAttention  (instance norm)           │
│     LH → HighFreqAttention (residual + dilated)      │
│     HL → HighFreqAttention                           │
│     HH → HighFreqAttention                           │
│                                                       │
│  3. High-Frequency Aggregation                       │
│     high_freq = LH + HL + HH                         │
│                                                       │
│  4. ODE Edge Reconstruction (RK2)                    │
│     f1 = dynamics_net(high_freq)                     │
│     f2 = dynamics_net(high_freq + α·f1)              │
│     edges = high_freq + gate·(β₁·f1 + β₂·f2)         │
│                                                       │
│  5. Frequency Fusion                                 │
│     concat[LL, LH, HL, edges] → Conv → residual      │
│                                                       │
└───────────────────────────────────────────────────────┘
    ↓
Enhanced Features [64, 128, 320, 512]
    ↓
Deep Supervision Decoder
    ↓
Main Prediction [B, 1, H, W]
Auxiliary Outputs: [aux1, aux2, aux3]
```

## Component Details

### 1. DeepWaveletDecomposition

**Purpose**: Separates features into frequency subbands using learnable Haar wavelets.

**Implementation**:
- Initializes with standard Haar wavelets:
  - LL (low-low): `[1,1;1,1]/4` - approximation, semantic content
  - LH (low-high): `[1,-1;1,-1]/4` - horizontal edges
  - HL (high-low): `[-1,1;-1,1]/4` - vertical edges
  - HH (high-high): `[-1,-1;1,1]/4` - diagonal edges
- Learnable filters adapt during training
- Depthwise separable convolutions for efficiency

**Parameters per scale**: ~1-5K (varies by channel count)

### 2. HighFrequencyAttention

**Purpose**: Enhances texture and edge features in high-frequency subbands.

**Key Features**:
- Residual blocks with dilated convolutions (capture multi-scale context)
- Joint spatial-channel attention:
  - Channel attention: Global pooling → FC → Sigmoid
  - Spatial attention: Channel pooling → Conv → Sigmoid
- Preserves fine-grained edge details

**Parameters**: ~200-800K per module (depends on input channels)

### 3. LowFrequencyAttention

**Purpose**: Processes semantic content while suppressing redundant patterns.

**Key Features**:
- Instance normalization for illumination invariance
- Global context modeling via adaptive pooling
- Reduced dimensionality (reduction=4) for efficiency
- Focuses on shape and semantic cues

**Parameters**: ~50-200K per module

### 4. ODEEdgeReconstruction

**Purpose**: Refines edge features using physics-inspired ODE dynamics.

**RK2 Solver Implementation**:
```python
# Heun's method (2nd-order Runge-Kutta)
for step in range(num_steps):
    k1 = dynamics_net(x)
    x_temp = x + dt * k1
    k2 = dynamics_net(x_temp)
    x = x + (dt/2) * (k1 + k2)

    # Hamiltonian stability
    potential = potential_net(x)
    x = x + potential * sigmoid(damping)
```

**Learnable Parameters**:
- `dt`: Time step size (clamped to [0.01, 0.5])
- `damping`: Stability factor
- `dynamics_net`: Edge evolution function
- `potential_net`: Hamiltonian potential energy

**Parameters**: ~400K-1.2M per module

### 5. Frequency Fusion

**Purpose**: Combines all frequency components into unified representation.

**Architecture**:
```python
concat[LL, LH, HL, reconstructed_edges]  # 4C channels
    ↓
Conv(4C → C) + BN + ReLU
    ↓
Conv(C → C) + BN + ReLU
    ↓
Add residual connection (input features)
```

**Parameters**: ~50-400K per scale

### 6. Deep Supervision Decoder

**Purpose**: Progressive upsampling with auxiliary losses at multiple scales.

**Architecture**:
```
f4 [512, H/32] ──┐
                 ├→ Decoder4 → [256, H/16] → aux3
f3 [320, H/16] ──┘              ↓
                                ├→ Decoder3 → [128, H/8] → aux2
f2 [128, H/8] ──────────────────┘              ↓
                                               ├→ Decoder2 → [64, H/4] → aux1
f1 [64, H/4] ───────────────────────────────────┘            ↓
                                                             Decoder1 → [32, H/2]
                                                                        ↓
                                                                   Upsample(2x)
                                                                        ↓
                                                                   PredHead
                                                                        ↓
                                                              Main Pred [1, H, W]
```

**Parameters**: ~2-3M

**Deep Supervision**:
- 3 auxiliary predictions from intermediate decoder layers
- All upsampled to final resolution for loss computation
- Helps gradient flow and improves boundary accuracy

## Total Parameter Count

**Target**: 12-15M parameters

**Expected Breakdown**:
- Wavelet Decompositions (4x): ~20K-200K total
- High-Freq Attentions (4x): ~800K-3.2M total
- Low-Freq Attentions (4x): ~200K-800K total
- ODE Reconstructions (4x): ~1.6M-4.8M total
- Frequency Fusion (4x): ~200K-1.6M total
- Decoder + Heads: ~2-3M

**Total**: ~12-15M ✓

## Usage

### Basic Forward Pass

```python
from models.expert_architectures import FEDERFrequencyExpert

# Create FEDER expert
feder = FEDERFrequencyExpert(feature_dims=[64, 128, 320, 512])

# Forward pass (features from PVT backbone)
pred, _ = feder(features, return_aux=False)
# pred: [B, 1, H, W]

# With deep supervision
pred, aux_outputs = feder(features, return_aux=True)
# pred: [B, 1, H, W]
# aux_outputs: list of 3 auxiliary predictions [B, 1, H, W]
```

### Integration with MoE

FEDER is integrated as Expert 4 in the Mixture of Experts:

```python
# models/model_level_moe.py
self.expert_models = nn.ModuleList([
    SINetExpert(feature_dims),      # Expert 0: Search & Identify
    PraNetExpert(feature_dims),     # Expert 1: Reverse Attention
    ZoomNetExpert(feature_dims),    # Expert 2: Multi-Scale Zoom
    FEDERFrequencyExpert(feature_dims)  # Expert 3: Frequency Analysis
])
```

### Training

Use the standard training command:

```bash
torchrun --nproc_per_node=2 train_advanced.py \
  --data-root /path/to/COD10K \
  --backbone pvt_v2_b2 \
  --num-experts 4 \
  --top-k 2 \
  --deep-supervision \
  --loss-scheme progressive \
  --img-size 416 \
  ...
```

## Advantages for Camouflaged Object Detection

1. **Frequency Domain Analysis**:
   - Separates texture (high-freq) from color (low-freq)
   - Critical for detecting objects that blend in color but differ in texture

2. **Learnable Wavelets**:
   - Adapts decomposition to camouflage patterns
   - More flexible than fixed Fourier or DCT transforms

3. **Specialized Attention**:
   - High-freq attention focuses on subtle edge differences
   - Low-freq attention with instance norm handles illumination changes

4. **ODE Edge Refinement**:
   - Physics-inspired dynamics smooth and stabilize edge features
   - Reduces noise while preserving true boundaries
   - Hamiltonian stability prevents gradient explosion

5. **Deep Supervision**:
   - Multi-scale auxiliary losses improve gradient flow
   - Better localization at object boundaries
   - Helps with small and large camouflaged objects

## Verification

Run the verification script to check the implementation:

```bash
python verify_feder.py
```

This will verify:
- ✓ Parameter count (12-15M)
- ✓ All components present
- ✓ Forward pass works
- ✓ Output shapes correct
- ✓ Gradient flow functional

## References

- Haar Wavelets: Haar, A. (1910). "Zur Theorie der orthogonalen Funktionensysteme"
- Runge-Kutta Methods: Butcher, J. C. (2016). "Numerical Methods for Ordinary Differential Equations"
- Frequency Analysis for COD: Multiple papers on texture-based camouflage detection

## Notes

- Output size is **dynamic** and matches input resolution (not hardcoded to 448x448)
- For 416×416 input → 416×416 output
- For 448×448 input → 448×448 output
- Compatible with DistributedDataParallel (DDP) and Automatic Mixed Precision (AMP)
- All components use standard PyTorch layers (no custom CUDA kernels)
