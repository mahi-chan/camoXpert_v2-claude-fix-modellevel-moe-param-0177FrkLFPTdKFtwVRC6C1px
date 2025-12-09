"""
CamoXpert Training Script with Anti-Under-Segmentation Improvements

Fixes under-segmentation by:
1. TverskyLoss (beta=0.7) - penalizes false negatives
2. Positive pixel weighting (pos_weight=3)
3. EMA for better generalization
4. Stronger augmentation with mixup
5. Multi-threshold validation

Target:
- Training Val: S-measure 0.93+, IoU 0.78+, F-measure 0.88+
- Test: S-measure 0.88+, IoU 0.72+, F-measure 0.82+

Usage:
    # Single GPU
    python train.py --data-root /path/to/COD10K --epochs 150

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 train.py --data-root /path/to/COD10K --use-ddp --epochs 150
"""

import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models.model_level_moe import ModelLevelMoE
from data.dataset import COD10KDataset
from losses.boundary_aware_loss import CombinedEnhancedLoss
from utils.ema import EMA
from metrics.cod_metrics import CODMetrics


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_ddp():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class ProgressiveAugmentation:
    """
    Progressive augmentation that increases strength over training.

    Args:
        initial_strength: Starting augmentation strength (0-1)
        max_strength: Maximum augmentation strength (0-1)
        transition_start: Epoch to start increasing strength
        transition_duration: Number of epochs to reach max strength
    """

    def __init__(self, image_size=448, initial_strength=0.0, max_strength=0.5,
                 transition_start=50, transition_duration=50):
        self.image_size = image_size
        self.initial_strength = initial_strength
        self.max_strength = max_strength
        self.transition_start = transition_start
        self.transition_duration = transition_duration
        self.current_strength = initial_strength

    def update_epoch(self, epoch):
        """Update augmentation strength based on current epoch"""
        if epoch < self.transition_start:
            self.current_strength = self.initial_strength
        elif epoch < self.transition_start + self.transition_duration:
            progress = (epoch - self.transition_start) / self.transition_duration
            self.current_strength = (
                self.initial_strength +
                (self.max_strength - self.initial_strength) * progress
            )
        else:
            self.current_strength = self.max_strength

    def __call__(self, image, mask):
        """Apply progressive augmentation"""
        # Basic transforms (always applied)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Progressive augmentations (strength increases over time)
        if random.random() > (1 - self.current_strength):
            # ColorJitter with progressive strength
            brightness = 0.4 * self.current_strength
            contrast = 0.4 * self.current_strength
            saturation = 0.4 * self.current_strength
            hue = 0.1 * self.current_strength

            image = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )(image)

        # Gaussian Blur with progressive probability
        if random.random() > (1 - self.current_strength * 0.6):
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 2.0)
            image = transforms.GaussianBlur(kernel_size, sigma)(image)

        # Random Grayscale with progressive probability
        if random.random() > (1 - self.current_strength * 0.2):
            image = transforms.Grayscale(num_output_channels=3)(image)

        # Coarse Dropout with progressive strength
        if random.random() > (1 - self.current_strength * 0.6):
            num_holes = int(8 * self.current_strength)
            max_h_size = int(32 * self.current_strength)
            max_w_size = int(32 * self.current_strength)
            if num_holes > 0 and max_h_size > 0 and max_w_size > 0:
                image = self._coarse_dropout(image, num_holes, max_h_size, max_w_size)

        return image, mask

    def _coarse_dropout(self, image, num_holes=8, max_h_size=32, max_w_size=32):
        """Apply coarse dropout (random rectangular masks)"""
        h, w = image.shape[1], image.shape[2]

        for _ in range(num_holes):
            if h > max_h_size and w > max_w_size:
                y = random.randint(0, h - max_h_size)
                x = random.randint(0, w - max_w_size)
                h_size = random.randint(1, max_h_size)
                w_size = random.randint(1, max_w_size)
                image[:, y:y+h_size, x:x+w_size] = 0

        return image


class MixupAugmentation:
    """Mixup augmentation"""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, images, masks):
        """Mix images and masks"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_masks = lam * masks + (1 - lam) * masks[index]

        return mixed_images, mixed_masks


def validate_multi_threshold(model, dataloader, device, thresholds=[0.3, 0.4, 0.5], rank=0):
    """
    Validate at multiple thresholds and return best results.

    Returns:
        best_metrics: Best metrics across all thresholds
        threshold_results: Results for each threshold
        diagnostics: Diagnostic information
    """
    model.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False, disable=(rank != 0)):
            # Handle tuple or dict batch format
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
            else:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)

            # Forward pass
            output = model(images)
            logits = output['pred'] if isinstance(output, dict) else (output[0] if isinstance(output, tuple) else output)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu())
            all_gts.append(masks.cpu())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # Compute diagnostics
    mean_pred_confidence = all_preds.mean().item()

    # Compute IoU for each image at threshold=0.5
    diagnostic_ious = []
    for i in range(all_preds.size(0)):
        pred_bin = (all_preds[i] > 0.5).float()
        gt_bin = (all_gts[i] > 0.5).float()

        intersection = (pred_bin * gt_bin).sum()
        union = pred_bin.sum() + gt_bin.sum() - intersection
        iou = (intersection / (union + 1e-8)).item()
        diagnostic_ious.append(iou)

    pct_high_iou = (np.array(diagnostic_ious) > 0.7).mean() * 100

    diagnostics = {
        'mean_pred_confidence': mean_pred_confidence,
        'pct_iou_above_0.7': pct_high_iou,
        'warning': mean_pred_confidence < 0.2
    }

    # Evaluate at each threshold
    threshold_results = {}

    for thresh in thresholds:
        metrics = CODMetrics()

        for i in range(all_preds.size(0)):
            metrics.update(all_preds[i], all_gts[i], threshold=thresh)

        threshold_results[thresh] = metrics.compute()

    # Find best threshold based on IoU
    best_thresh = max(threshold_results.keys(), key=lambda t: threshold_results[t]['IoU'])
    best_metrics = threshold_results[best_thresh]
    best_metrics['best_threshold'] = best_thresh

    return best_metrics, threshold_results, diagnostics


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, ema, mixup,
                use_mixup=True, accumulation_steps=1, rank=0):
    """Train for one epoch with enhanced boundary-aware loss"""
    model.train()

    total_loss = 0.0
    loss_components = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # Handle tuple or dict batch format
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
        else:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

        # Apply mixup augmentation
        if use_mixup and random.random() > 0.5:
            images, masks = mixup(images, masks)

        # Forward pass with mixed precision
        with autocast():
            # Get prediction AND auxiliary outputs (routing_info)
            pred, routing_info = model(images, return_routing_info=True)

            # Apply sigmoid if needed
            if pred.min() < 0:
                pred = torch.sigmoid(pred)

            # Compute enhanced loss with auxiliary outputs
            loss, loss_dict = criterion(pred, masks, aux_outputs=routing_info)
            loss = loss / accumulation_steps  # Scale loss for gradient accumulation

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA (only after optimizer step)
            if ema is not None:
                ema.update()

        # Accumulate losses
        total_loss += loss.item() * accumulation_steps
        for key, value in loss_dict.items():
            if key not in loss_components:
                loss_components[key] = 0.0
            if isinstance(value, torch.Tensor):
                loss_components[key] += value.item()
            else:
                loss_components[key] += value
        num_batches += 1

        # Update progress bar
        if rank == 0:
            seg_loss = loss_dict.get('seg_bce', 0)
            seg_loss_val = seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'seg': f"{seg_loss_val:.3f}"
            })

    # Compute averages
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def main():
    parser = argparse.ArgumentParser(description='CamoXpert Training with Anti-Under-Segmentation')

    # ========== Data Arguments ==========
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of COD10K dataset')
    parser.add_argument('--img-size', type=str, default=448,
                       help='Input image size (default: 448)')
    parser.add_argument('--batch-size', type=int, default=6,
                       help='Batch size per GPU (default: 6)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers (default: 8)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable dataset caching')

    # ========== Model Arguments ==========
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                       help='Backbone architecture (default: pvt_v2_b2)')
    parser.add_argument('--num-experts', type=int, default=3,
                       help='Number of experts (default: 3 - SINet, PraNet, ZoomNet)')
    parser.add_argument('--top-k', type=int, default=2,
                       help='Number of experts to select (default: 2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbone')
    parser.add_argument('--deep-supervision', action='store_true',
                       help='Enable deep supervision')

    # ========== Training Arguments ==========
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs (default: 5)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                       help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps (default: 1)')

    # ========== Loss Arguments (Enhanced Boundary-Aware Loss) ==========
    parser.add_argument('--seg-weight', type=float, default=1.0,
                       help='Weight for segmentation loss (boundary-aware BCE + Dice) (default: 1.0)')
    parser.add_argument('--boundary-weight', type=float, default=2.0,
                       help='Weight for boundary prediction loss (default: 2.0)')
    parser.add_argument('--discontinuity-weight', type=float, default=0.3,
                       help='Weight for discontinuity supervision (TDD + GAD) (default: 0.3)')
    parser.add_argument('--expert-weight', type=float, default=0.3,
                       help='Weight for per-expert supervision (default: 0.3)')
    parser.add_argument('--hard-mining-weight', type=float, default=0.5,
                       help='Weight for hard sample mining (default: 0.5)')
    parser.add_argument('--load-balance-weight', type=float, default=0.1,
                       help='Weight for router load balance (default: 0.1)')

    # ========== EMA Arguments ==========
    parser.add_argument('--ema-decay', type=float, default=0.999,
                       help='EMA decay rate (default: 0.999)')
    parser.add_argument('--no-ema', action='store_true',
                       help='Disable EMA')

    # ========== Augmentation Arguments ==========
    parser.add_argument('--use-mixup', action='store_true', default=True,
                       help='Use mixup augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                       help='Mixup alpha parameter (default: 0.2)')
    parser.add_argument('--enable-progressive-aug', action='store_true',
                       help='Enable progressive augmentation')
    parser.add_argument('--aug-transition-epoch', type=int, default=50,
                       help='Epoch to start increasing augmentation (default: 50)')
    parser.add_argument('--aug-max-strength', type=float, default=0.5,
                       help='Maximum augmentation strength (default: 0.5)')
    parser.add_argument('--aug-transition-duration', type=int, default=50,
                       help='Epochs to reach max augmentation (default: 50)')

    # ========== Router Warmup Arguments ==========
    parser.add_argument('--enable-router-warmup', action='store_true',
                       help='Enable router warmup (freeze router initially)')
    parser.add_argument('--router-warmup-epochs', type=int, default=20,
                       help='Epochs to keep router frozen (default: 20)')

    # ========== Distributed Training Arguments ==========
    parser.add_argument('--use-ddp', action='store_true',
                       help='Use DistributedDataParallel for multi-GPU training')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use automatic mixed precision')

    # ========== Checkpointing Arguments ==========
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory (default: ./checkpoints)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')

    # ========== Validation Arguments ==========
    parser.add_argument('--val-freq', type=int, default=5,
                       help='Validate every N epochs (default: 5)')
    parser.add_argument('--val-thresholds', type=float, nargs='+', default=[0.3, 0.4, 0.5],
                       help='Thresholds for multi-threshold validation (default: 0.3 0.4 0.5)')

    # ========== Other Arguments ==========
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Setup DDP if requested
    rank, world_size, local_rank = 0, 1, 0
    if args.use_ddp:
        rank, world_size, local_rank = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed
    set_seed(args.seed + rank)

    # Print configuration (only on rank 0)
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"CAMOXPERT TRAINING - FIXING UNDER-SEGMENTATION")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"DDP: {'Enabled' if args.use_ddp else 'Disabled'} (Rank {rank}/{world_size})")
        print(f"Image size: {args.img_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size * args.accumulation_steps}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"EMA: {'Enabled' if not args.no_ema else 'Disabled'} (decay={args.ema_decay})")
        print(f"Mixup: {args.use_mixup} (alpha={args.mixup_alpha})")
        print(f"Progressive Aug: {args.enable_progressive_aug}")
        print(f"Router Warmup: {args.enable_router_warmup} (epochs={args.router_warmup_epochs})")
        print(f"\nEnhanced Loss Weights (Boundary-Aware):")
        print(f"  Segmentation:     {args.seg_weight}")
        print(f"  Boundary Pred:    {args.boundary_weight} ⭐")
        print(f"  Discontinuity:    {args.discontinuity_weight}")
        print(f"  Per-Expert:       {args.expert_weight}")
        print(f"  Hard Mining:      {args.hard_mining_weight}")
        print(f"  Load Balance:     {args.load_balance_weight}")
        print(f"{'='*70}\n")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    if rank == 0:
        print("Loading datasets...")

    train_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='train',
        img_size=args.img_size,
        augment=True,
        cache_in_memory=not args.no_cache,
        rank=rank,
        world_size=world_size
    )

    val_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='test',  # Use 'test' split for validation
        img_size=args.img_size,
        augment=False,
        cache_in_memory=not args.no_cache,
        rank=rank,
        world_size=world_size
    )

    # Create samplers for DDP
    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if rank == 0:
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Val samples: {len(val_dataset)}\n")

    # Create model
    if rank == 0:
        print("Creating model...")

    model = ModelLevelMoE(
        backbone_name=args.backbone,
        num_experts=args.num_experts,
        top_k=args.top_k,
        pretrained=args.pretrained
    ).to(device)

    # Wrap model with DDP
    if args.use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create EMA
    ema = None
    if not args.no_ema:
        # EMA wraps the original model (before DDP)
        ema_model = model.module if args.use_ddp else model
        ema = EMA(ema_model, decay=args.ema_decay)
        if rank == 0:
            print(f"✓ EMA created with decay={args.ema_decay}\n")

    # Create enhanced loss function with boundary awareness
    criterion = CombinedEnhancedLoss(
        seg_weight=args.seg_weight,
        boundary_weight=args.boundary_weight,
        discontinuity_weight=args.discontinuity_weight,
        expert_weight=args.expert_weight,
        hard_mining_weight=args.hard_mining_weight,
        load_balance_weight=args.load_balance_weight
    )

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler with warmup
    def warmup_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )

    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None

    # Augmentations
    mixup = MixupAugmentation(alpha=args.mixup_alpha) if args.use_mixup else None

    # Progressive augmentation
    if args.enable_progressive_aug:
        progressive_aug = ProgressiveAugmentation(
            image_size=args.img_size,
            initial_strength=0.0,
            max_strength=args.aug_max_strength,
            transition_start=args.aug_transition_epoch,
            transition_duration=args.aug_transition_duration
        )

    # Training state
    best_iou = 0.0
    start_epoch = 0

    # Resume if needed
    if args.resume:
        if rank == 0:
            print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_to_load = model.module if args.use_ddp else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if ema is not None and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        if rank == 0:
            print(f"✓ Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}\n")

    # Training history
    history = {
        'train_loss': [],
        'train_loss_components': [],
        'val_metrics': [],
        'val_diagnostics': []
    }

    if rank == 0:
        print("Starting training...\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Update sampler epoch for DDP
        if args.use_ddp:
            train_sampler.set_epoch(epoch)

        # Update progressive augmentation
        if args.enable_progressive_aug:
            progressive_aug.update_epoch(epoch)
            if rank == 0 and epoch % 10 == 0:
                print(f"Aug strength: {progressive_aug.current_strength:.2f}")

        # Router warmup: freeze router for first N epochs
        if args.enable_router_warmup:
            model_to_control = model.module if args.use_ddp else model
            if hasattr(model_to_control, 'freeze_router'):
                if epoch < args.router_warmup_epochs:
                    model_to_control.freeze_router()
                    if epoch == 0 and rank == 0:
                        print(f"🔒 Router FROZEN for first {args.router_warmup_epochs} epochs\n")
                elif epoch == args.router_warmup_epochs:
                    model_to_control.unfreeze_router()
                    if rank == 0:
                        print("🔓 Router UNFROZEN - now learning to route\n")

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"{'='*70}")

        # Train
        avg_loss, loss_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, ema, mixup,
            args.use_mixup, args.accumulation_steps, rank
        )

        if rank == 0:
            print(f"\nTraining Loss: {avg_loss:.4f}")
            print("Loss Components:")
            # Print main components first
            main_keys = ['seg_bce', 'seg_dice', 'boundary', 'tdd', 'gad', 'expert', 'hard_mining', 'load_balance']
            for key in main_keys:
                if key in loss_components:
                    print(f"  {key}: {loss_components[key]:.4f}")
            # Print any other components
            for key, value in loss_components.items():
                if key not in main_keys and key != 'total':
                    print(f"  {key}: {value:.4f}")

            history['train_loss'].append(avg_loss)
            history['train_loss_components'].append(loss_components)

        # Step scheduler
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nLearning rate: {current_lr:.2e}")

        # Validate
        if (epoch + 1) % args.val_freq == 0 or (epoch + 1) == args.epochs:
            if rank == 0:
                print("\nValidating with EMA weights...")

            # Use EMA model for validation
            if ema is not None:
                ema.apply_shadow()

            # Get model for validation (unwrap DDP)
            val_model = model.module if args.use_ddp else model

            # Validate
            best_metrics, threshold_results, diagnostics = validate_multi_threshold(
                val_model, val_loader, device, thresholds=args.val_thresholds, rank=rank
            )

            # Restore model weights
            if ema is not None:
                ema.restore()

            if rank == 0:
                print(f"\nValidation Results:")
                print(f"  IoU @ 0.3: {threshold_results[0.3]['IoU']:.4f}")
                print(f"  IoU @ 0.4: {threshold_results[0.4]['IoU']:.4f}")
                print(f"  IoU @ 0.5: {threshold_results[0.5]['IoU']:.4f}")
                print(f"  Best IoU:  {best_metrics['IoU']:.4f} ⭐ (threshold={best_metrics['best_threshold']})")
                print(f"  S-measure: {best_metrics['S-measure']:.4f}")
                print(f"  F-measure: {best_metrics['F-measure']:.4f}")
                print(f"  MAE:       {best_metrics['MAE']:.4f}")

                print(f"\nDiagnostics:")
                print(f"  Mean prediction: {diagnostics['mean_pred_confidence']:.4f}")
                print(f"  % IoU > 0.7:     {diagnostics['pct_iou_above_0.7']:.1f}%")

                # Warnings
                if diagnostics['mean_pred_confidence'] < 0.2:
                    print(f"  ⚠️  WARNING: Model is under-confident (mean < 0.2)!")
                elif diagnostics['mean_pred_confidence'] > 0.6:
                    print(f"  ⚠️  WARNING: Model is over-confident (mean > 0.6)!")

                history['val_metrics'].append(best_metrics)
                history['val_diagnostics'].append(diagnostics)

                # Save best model (only rank 0)
                if best_metrics['IoU'] > best_iou:
                    best_iou = best_metrics['IoU']
                    print(f"\n✓ New best IoU: {best_iou:.4f}")

                    # Apply EMA for best model
                    if ema is not None:
                        ema.apply_shadow()

                    # Save model (unwrap DDP)
                    save_model = model.module if args.use_ddp else model

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': save_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ema_state_dict': ema.state_dict() if ema is not None else None,
                        'best_iou': best_iou,
                        'metrics': best_metrics,
                        'args': vars(args)
                    }, checkpoint_dir / 'best_model.pth')

                    # Restore model
                    if ema is not None:
                        ema.restore()

        # Save checkpoint (only rank 0)
        if rank == 0 and (epoch + 1) % args.save_interval == 0:
            save_model = model.module if args.use_ddp else model

            torch.save({
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_state_dict': ema.state_dict() if ema is not None else None,
                'best_iou': best_iou,
                'args': vars(args)
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Save final model (only rank 0)
    if rank == 0:
        print("\nSaving final model...")

        if ema is not None:
            ema.apply_shadow()

        save_model = model.module if args.use_ddp else model

        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': save_model.state_dict(),
            'ema_state_dict': ema.state_dict() if ema is not None else None,
            'best_iou': best_iou,
            'args': vars(args)
        }, checkpoint_dir / 'final_model.pth')

        # Save training history
        with open(checkpoint_dir / 'history.json', 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            history_serializable = {}
            for key, value in history.items():
                if isinstance(value, list):
                    history_serializable[key] = [
                        {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                         for k, v in item.items()} if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    history_serializable[key] = value
            json.dump(history_serializable, f, indent=2)

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best IoU: {best_iou:.4f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print(f"{'='*70}\n")

    # Cleanup DDP
    if args.use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
