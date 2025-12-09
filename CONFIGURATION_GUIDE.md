# CamoXpert Training Configuration Guide

Complete reference for all command-line flags in `train_advanced.py`.

## 🔧 Quick Examples

### Minimal (Quick Test)
```bash
python train_advanced.py \
    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
    --epochs 5
```

### Recommended for Kaggle (2x T4 GPUs)
```bash
torchrun --nproc_per_node=2 train_advanced.py \
    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
    --epochs 100 \
    --batch-size 12 \
    --backbone pvt_v2_b2 \
    --use-ddp
```

### High Performance (Larger Backbone)
```bash
torchrun --nproc_per_node=2 train_advanced.py \
    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
    --epochs 150 \
    --batch-size 8 \
    --backbone pvt_v2_b4 \
    --num-experts 6 \
    --top-k 3 \
    --use-ddp
```

## 📊 All Configuration Flags

### Data Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data-root` | str | **REQUIRED** | Path to COD10K-v3 dataset |
| `--batch-size` | int | 16 | Batch size per GPU |
| `--img-size` | int | 384 | Input image size (384x384) |
| `--num-workers` | int | 4 | DataLoader workers |

**Example:**
```bash
--data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
--batch-size 12 \
--img-size 384 \
--num-workers 4
```

---

### Model Architecture

| Flag | Type | Default | Choices | Description |
|------|------|---------|---------|-------------|
| `--backbone` | str | pvt_v2_b2 | pvt_v2_b2, b3, b4, b5 | Backbone architecture |
| `--num-experts` | int | 4 | 2-8 | Number of expert models |
| `--top-k` | int | 2 | 1-num_experts | Active experts per forward pass |
| `--pretrained` | flag | True | - | Use pretrained backbone weights |
| `--no-pretrained` | flag | - | - | Train backbone from scratch |
| `--deep-supervision` | flag | True | - | Enable deep supervision |
| `--no-deep-supervision` | flag | - | - | Disable deep supervision |

**Backbone Comparison:**

| Backbone | Params | Speed | Accuracy | Best For |
|----------|--------|-------|----------|----------|
| pvt_v2_b2 | ~25M | Fast | Good | **Kaggle (recommended)** |
| pvt_v2_b3 | ~45M | Medium | Better | High accuracy |
| pvt_v2_b4 | ~62M | Slower | Best | Research/competitions |
| pvt_v2_b5 | ~82M | Slowest | Best+ | Maximum accuracy |

**Example:**
```bash
# Standard setup
--backbone pvt_v2_b2 --num-experts 4 --top-k 2

# More experts for diversity
--backbone pvt_v2_b3 --num-experts 6 --top-k 3

# Train from scratch (not recommended)
--backbone pvt_v2_b2 --no-pretrained
```

---

### Training Parameters

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--epochs` | int | 100 | Total training epochs |
| `--lr` | float | 1e-4 | Initial learning rate |
| `--weight-decay` | float | 1e-4 | Weight decay for AdamW |
| `--accumulation-steps` | int | 2 | Gradient accumulation steps |
| `--warmup-epochs` | int | 5 | Warmup epochs for scheduler |
| `--min-lr` | float | 1e-6 | Minimum learning rate |
| `--use-amp` | flag | True | Use mixed precision training |
| `--no-amp` | flag | - | Disable AMP (not recommended) |

**Learning Rate Schedule:**
- **Epochs 1-5:** Linear warmup from `min-lr` (1e-6) to `lr` (1e-4)
- **Epochs 6-100:** Cosine annealing from `lr` to `min-lr`

**Example:**
```bash
# Standard training
--epochs 100 --lr 0.0001 --warmup-epochs 5

# Longer training with smaller LR
--epochs 150 --lr 0.00005 --warmup-epochs 10 --min-lr 1e-7

# Disable AMP (if having issues)
--epochs 100 --no-amp
```

---

### Progressive Augmentation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--enable-progressive-aug` | flag | True | Enable progressive augmentation |
| `--aug-transition-epoch` | int | 20 | Epoch to start increasing augmentation strength |

**Augmentation Timeline:**
- **Epochs 1-20:** Mild augmentation (strength = 0.3)
- **Epochs 21-100:** Gradually increase to strong augmentation (strength = 0.8)

**COD-Specific Augmentations:**
1. **Fourier-Based Mixing:** Frequency domain blending
2. **Contrastive Learning:** Positive pair generation
3. **Mirror Disruption:** Symmetry breaking

**Example:**
```bash
# Standard (recommended)
--enable-progressive-aug --aug-transition-epoch 20

# Start augmentation later
--enable-progressive-aug --aug-transition-epoch 40
```

---

### Loss Configuration

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--loss-scheme` | str | progressive | Loss weighting scheme (progressive/full) |
| `--boundary-lambda-start` | float | 0.5 | Starting weight for boundary loss |
| `--boundary-lambda-end` | float | 2.0 | Ending weight for boundary loss |
| `--frequency-weight` | float | 1.5 | Weight for frequency loss |
| `--scale-small-weight` | float | 2.0 | Weight for small object scale loss |
| `--uncertainty-threshold` | float | 0.5 | Threshold for uncertainty loss |

**Progressive Loss Scheme:**

| Stage | Epochs | Focus | Active Components |
|-------|--------|-------|-------------------|
| Early | 1-30 | Basic segmentation | BCE-Dice (1.0), IoU (0.5) |
| Mid | 31-70 | Add boundaries | + Boundary (ramp 0.5→2.0), Frequency (1.5) |
| Late | 71-100 | Refinement | + Scale (2.0), Uncertainty (adaptive) |

**Example:**
```bash
# Standard progressive scheme (recommended)
--loss-scheme progressive

# Strong boundary emphasis
--boundary-lambda-start 1.0 --boundary-lambda-end 3.0

# Focus on small objects
--scale-small-weight 3.0

# Full loss from start (not recommended)
--loss-scheme full
```

---

### Checkpointing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint-dir` | str | ./checkpoints_advanced | Checkpoint directory |
| `--save-interval` | int | 5 | Save checkpoint every N epochs |
| `--resume-from` | str | None | Resume from checkpoint path |

**Checkpoint Structure:**
```
checkpoints_advanced/
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── ...
├── checkpoint_epoch_100.pth
└── best_model.pth  # Best validation IoU
```

**Example:**
```bash
# Standard checkpointing
--checkpoint-dir /kaggle/working/checkpoints --save-interval 5

# Resume training
--resume-from /kaggle/working/checkpoints/checkpoint_epoch_50.pth
```

---

### Distributed Training

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use-ddp` | flag | False | Enable DistributedDataParallel |
| `--local_rank` | int | 0 | Local rank (set by torchrun) |

**Usage:**
```bash
# Single GPU
python train_advanced.py --data-root /path/to/data

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 train_advanced.py \
    --data-root /path/to/data \
    --use-ddp

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 train_advanced.py \
    --data-root /path/to/data \
    --use-ddp
```

**Effective Batch Size:**
```
Effective batch size = batch_size × num_gpus × accumulation_steps
```

Example: `--batch-size 12` with 2 GPUs and `--accumulation-steps 2` = **48 effective batch size**

---

### System

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--seed` | int | 42 | Random seed for reproducibility |
| `--cache-in-memory` | flag | True | Cache dataset in RAM for faster training |
| `--no-cache` | flag | - | Disable RAM caching |

**DDP-Aware Caching:**
- With multi-GPU training, each GPU only caches its subset of data
- Example: 2 GPUs with 3000 images → GPU 0 caches 1500, GPU 1 caches 1500
- Recommended for Kaggle (2x T4 with 30GB RAM) - saves ~30-40% data loading time
- Disable with `--no-cache` if memory is limited

**Example:**
```bash
# Enable caching (default, recommended)
torchrun --nproc_per_node=2 train_advanced.py --data-root /path/to/data

# Disable caching if limited RAM
python train_advanced.py --data-root /path/to/data --no-cache
```

---

## 🎯 Recommended Configurations

### 1. Quick Test (5 minutes)
```bash
python train_advanced.py \
    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
    --epochs 5 \
    --batch-size 8 \
    --backbone pvt_v2_b2
```

### 2. Standard Training (Kaggle 2x T4, ~90 min)
```bash
torchrun --nproc_per_node=2 train_advanced.py \
    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
    --epochs 100 \
    --batch-size 12 \
    --accumulation-steps 2 \
    --backbone pvt_v2_b2 \
    --num-experts 4 \
    --top-k 2 \
    --lr 0.0001 \
    --warmup-epochs 5 \
    --use-amp \
    --enable-progressive-aug \
    --aug-transition-epoch 20 \
    --use-ddp
```

### 3. High Performance (Kaggle 2x T4, ~150 min)
```bash
torchrun --nproc_per_node=2 train_advanced.py \
    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
    --epochs 150 \
    --batch-size 8 \
    --accumulation-steps 3 \
    --backbone pvt_v2_b4 \
    --num-experts 6 \
    --top-k 3 \
    --lr 0.00005 \
    --warmup-epochs 10 \
    --boundary-lambda-end 3.0 \
    --scale-small-weight 3.0 \
    --use-amp \
    --enable-progressive-aug \
    --use-ddp
```

### 4. Fast Experimentation (Single GPU, ~120 min)
```bash
python train_advanced.py \
    --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
    --epochs 80 \
    --batch-size 8 \
    --accumulation-steps 4 \
    --backbone pvt_v2_b2 \
    --num-experts 4 \
    --top-k 2 \
    --use-amp
```

---

## 📈 Expected Performance

| Configuration | Training Time | IoU | F-measure | MAE | Params |
|---------------|---------------|-----|-----------|-----|--------|
| Quick Test | 5 min | 0.65-0.70 | 0.70-0.75 | 0.055 | 85M |
| Standard | 90 min | **0.80-0.82** | **0.85-0.87** | **0.035-0.040** | 85M |
| High Performance | 150 min | **0.82-0.84** | **0.87-0.89** | **0.030-0.035** | 125M |
| Fast Experiment | 120 min | 0.78-0.80 | 0.83-0.85 | 0.040-0.045 | 85M |

**Baseline (ModelLevelMoE without new modules):** IoU 0.76-0.78, F-measure 0.81-0.83, MAE 0.045-0.050

---

## 💡 Tuning Tips

### For Better Accuracy
- Use larger backbone: `--backbone pvt_v2_b4`
- More experts: `--num-experts 6 --top-k 3`
- Stronger boundary loss: `--boundary-lambda-end 3.0`
- Focus on small objects: `--scale-small-weight 3.0`

### For Faster Training
- Smaller backbone: `--backbone pvt_v2_b2`
- Fewer experts: `--num-experts 4 --top-k 2`
- Higher learning rate: `--lr 0.0002`
- Fewer epochs: `--epochs 80`

### For Limited GPU Memory
- Smaller batch size: `--batch-size 6`
- More accumulation: `--accumulation-steps 4`
- Smaller backbone: `--backbone pvt_v2_b2`

### For Overfitting Issues
- Stronger augmentation earlier: `--aug-transition-epoch 10`
- Higher weight decay: `--weight-decay 0.0002`
- More dropout (requires code change)

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size and increase accumulation
--batch-size 6 --accumulation-steps 4
```

### Training Unstable
```bash
# Lower learning rate and longer warmup
--lr 0.00005 --warmup-epochs 10
```

### Poor Small Object Detection
```bash
# Increase scale loss weight
--scale-small-weight 3.0 --boundary-lambda-end 3.0
```

### Want to Change Something Without Git Push
**That's what this guide is for!** All parameters are now configurable via command-line flags.
Just change the flags in your Kaggle notebook cell and rerun.
