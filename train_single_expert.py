"""
Train Single SINet Expert - Diagnostic Script

Purpose: Verify the loss function works before using full MoE
Target: IoU > 0.65 on validation

This script trains a simple PVT-v2-b2 + SINet decoder to verify:
1. TverskyLoss (beta=0.7) correctly penalizes false negatives
2. Loss function achieves good IoU
3. Model predictions are confident (mean > 0.2)

If this works, we can confidently use the loss in full MoE training.
"""

import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

from models.expert_architectures import SINetExpert
from data.dataset import COD10KDataset
from losses.aggressive_loss import AggressiveCombinedLoss
from metrics.cod_metrics import CODMetrics
import timm


class SimpleSINet(nn.Module):
    """
    Simple wrapper: PVT-v2-b2 backbone + SINet decoder

    This is equivalent to using just one expert from the MoE,
    but without the router overhead.
    """

    def __init__(self, backbone_name='pvt_v2_b2', pretrained=True):
        super().__init__()

        print(f"\n{'='*70}")
        print("SIMPLE SINET MODEL")
        print(f"{'='*70}")
        print(f"  Backbone: {backbone_name}")
        print(f"  Decoder: SINet-style")
        print(f"{'='*70}\n")

        # Create backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        # Get feature dimensions
        self.feature_dims = [64, 128, 320, 512]  # PVT-v2-b2

        # Create SINet decoder
        self.decoder = SINetExpert(self.feature_dims)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"✓ Total parameters: {total_params/1e6:.1f}M")
        print(f"✓ Trainable parameters: {trainable_params/1e6:.1f}M\n")

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            pred: [B, 1, H, W] - main prediction
            aux_preds: list of auxiliary predictions for deep supervision
        """
        # Extract features
        features = self.backbone(x)  # [f1, f2, f3, f4]

        # Decode
        pred, aux_preds = self.decoder(features)

        return pred, aux_preds


def validate_multi_threshold(model, dataloader, device, thresholds=[0.3, 0.4, 0.5], is_main_process=True):
    """
    Validate at multiple thresholds and return best results.

    Returns:
        best_metrics: Best metrics across all thresholds
        diagnostics: Prediction statistics
    """
    model.eval()

    all_preds = []
    all_gts = []

    # Only show progress bar on main process
    if is_main_process:
        iterator = tqdm(dataloader, desc="Validating", leave=False)
    else:
        iterator = dataloader

    with torch.no_grad():
        for batch in iterator:
            # Handle both tuple and dict formats
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
            else:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)

            # Forward
            pred, _ = model(images)
            pred = torch.sigmoid(pred)

            all_preds.append(pred.cpu())
            all_gts.append(masks.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # Diagnostics
    mean_pred = all_preds.mean().item()
    min_pred = all_preds.min().item()
    max_pred = all_preds.max().item()

    diagnostics = {
        'mean_pred': mean_pred,
        'min_pred': min_pred,
        'max_pred': max_pred,
        'warning': mean_pred < 0.25
    }

    # Evaluate at each threshold
    threshold_results = {}

    for thresh in thresholds:
        metrics = CODMetrics()

        for i in range(all_preds.size(0)):
            # Add batch dimension back (metrics expect [B, C, H, W])
            pred_sample = all_preds[i].unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
            gt_sample = all_gts[i].unsqueeze(0)      # [1, H, W] -> [1, 1, H, W]
            metrics.update(pred_sample, gt_sample, threshold=thresh)

        threshold_results[thresh] = metrics.compute()

    # Find best threshold based on IoU
    best_thresh = max(threshold_results.keys(), key=lambda t: threshold_results[t]['IoU'])
    best_metrics = threshold_results[best_thresh]
    best_metrics['best_threshold'] = best_thresh

    return best_metrics, threshold_results, diagnostics


def setup_ddp(args):
    """Setup distributed training."""
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        args.local_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()
    else:
        args.world_size = 1
        args.local_rank = 0

    return args.local_rank == 0  # is_main_process


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, is_main_process=True):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    loss_components = {
        'focal_tversky': 0.0,
        'asym_bce': 0.0,
        'confidence': 0.0
    }
    num_batches = 0

    # Only show progress bar on main process
    if is_main_process:
        pbar = tqdm(dataloader, desc="Training")
    else:
        pbar = dataloader

    for batch in pbar:
        # Handle both tuple and dict formats
        if isinstance(batch, dict):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
        else:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

        # Forward with AMP
        with autocast():
            pred, aux_preds = model(images)
            loss, loss_dict = criterion(pred, masks, aux_preds)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Accumulate
        total_loss += loss.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key]
        num_batches += 1

        # Update progress bar (only on main process)
        if is_main_process:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ftv': f"{loss_dict.get('focal_tversky', 0):.4f}"
            })

    # Compute averages
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return avg_loss, loss_components


def main():
    parser = argparse.ArgumentParser(description='Train Single SINet Expert')

    # Data
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of COD10K dataset')
    parser.add_argument('--image-size', type=int, default=448,
                       help='Input image size (default: 448)')
    parser.add_argument('--batch-size', type=int, default=6,
                       help='Batch size (default: 6)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of workers (default: 8)')

    # Model
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                       help='Backbone (default: pvt_v2_b2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbone')

    # Training
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs (default: 150)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs (default: 5)')

    # Aggressive Loss Arguments
    parser.add_argument('--focal-tversky-weight', type=float, default=3.0,
                       help='Focal Tversky loss weight (default: 3.0)')
    parser.add_argument('--asym-bce-weight', type=float, default=1.0,
                       help='Asymmetric BCE loss weight (default: 1.0)')
    parser.add_argument('--confidence-weight', type=float, default=0.5,
                       help='Confidence penalty weight (default: 0.5)')
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Tversky alpha (FP weight) (default: 0.2)')
    parser.add_argument('--beta', type=float, default=0.8,
                       help='Tversky beta (FN weight) (default: 0.8)')
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Focal gamma parameter (default: 2.0)')
    parser.add_argument('--pos-weight', type=float, default=10.0,
                       help='BCE positive pixel weight (default: 10.0)')
    parser.add_argument('--neg-weight', type=float, default=0.5,
                       help='BCE negative pixel weight (default: 0.5)')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_single',
                       help='Checkpoint directory')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='Save every N epochs (default: 10)')
    parser.add_argument('--val-freq', type=int, default=5,
                       help='Validate every N epochs (default: 5)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # DDP
    parser.add_argument('--use-ddp', action='store_true',
                       help='Use Distributed Data Parallel training')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for DDP (set by torchrun)')

    args = parser.parse_args()

    # Setup DDP
    is_main_process = setup_ddp(args)

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.use_ddp:
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if is_main_process:
        print(f"\n{'='*70}")
        print("SINGLE SINET EXPERT TRAINING - AGGRESSIVE MODE")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"DDP: {args.use_ddp} (world_size={args.world_size})")
        print(f"Image size: {args.image_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"\nAggressive Loss Configuration:")
        print(f"  Focal Tversky: {args.focal_tversky_weight} (α={args.alpha}, β={args.beta}, γ={args.gamma})")
        print(f"  Asymmetric BCE: {args.asym_bce_weight} (pos={args.pos_weight}, neg={args.neg_weight})")
        print(f"  Confidence:    {args.confidence_weight}")
        print(f"  Total weight:  {args.focal_tversky_weight + args.asym_bce_weight + args.confidence_weight}")
        print(f"{'='*70}\n")

    # Create checkpoint dir
    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    if is_main_process:
        print("Loading datasets...")

    train_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='train',
        img_size=args.image_size,
        augment=True,
        cache_in_memory=False
    )

    val_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='val',
        img_size=args.image_size,
        augment=False,
        cache_in_memory=False
    )

    # Samplers for DDP
    if args.use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=False
        )
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

    if is_main_process:
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Val samples: {len(val_dataset)}\n")

    # Create model
    if is_main_process:
        print("Creating model...")
    model = SimpleSINet(
        backbone_name=args.backbone,
        pretrained=args.pretrained
    ).to(device)

    # Wrap with DDP
    if args.use_ddp:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )

    # Create aggressive loss
    if is_main_process:
        criterion = AggressiveCombinedLoss(
            focal_tversky_weight=args.focal_tversky_weight,
            asym_bce_weight=args.asym_bce_weight,
            confidence_weight=args.confidence_weight,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            pos_weight=args.pos_weight,
            neg_weight=args.neg_weight
        )
    else:
        # Create loss without printing (for non-main processes)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        criterion = AggressiveCombinedLoss(
            focal_tversky_weight=args.focal_tversky_weight,
            asym_bce_weight=args.asym_bce_weight,
            confidence_weight=args.confidence_weight,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            pos_weight=args.pos_weight,
            neg_weight=args.neg_weight
        )
        sys.stdout = old_stdout

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
    main_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=1,
        eta_min=args.lr * 0.01
    )

    # AMP scaler
    scaler = GradScaler()

    # Training history
    history = {
        'train_loss': [],
        'val_metrics': [],
        'diagnostics': []
    }

    best_iou = 0.0

    if is_main_process:
        print("Starting training...\n")

    # Training loop
    for epoch in range(args.epochs):
        import time
        epoch_start = time.time()

        # Set epoch for distributed sampler
        if args.use_ddp:
            train_sampler.set_epoch(epoch)

        if is_main_process:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"{'='*70}")

        # Train
        train_start = time.time()
        avg_loss, loss_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, is_main_process
        )
        train_time = time.time() - train_start

        if is_main_process:
            print(f"\nTraining Loss: {avg_loss:.4f} (time: {train_time:.1f}s)")
            print("Loss Components:")
            for key, value in loss_components.items():
                print(f"  {key}: {value:.4f}")

        history['train_loss'].append(avg_loss)

        # Step scheduler
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nLearning rate: {current_lr:.2e}")

        # Validate
        if (epoch + 1) % args.val_freq == 0:
            if is_main_process:
                print("\nValidating...")

            best_metrics, threshold_results, diagnostics = validate_multi_threshold(
                model, val_loader, device, thresholds=[0.3, 0.4, 0.5], is_main_process=is_main_process
            )

            if is_main_process:
                print(f"\nValidation Results (best threshold={best_metrics['best_threshold']}):")
                print(f"  S-measure: {best_metrics['S-measure']:.4f}")
                print(f"  F-measure: {best_metrics['F-measure']:.4f}")
                print(f"  IoU:       {best_metrics['IoU']:.4f} {'⭐' if best_metrics['IoU'] > 0.65 else ''}")
                print(f"  MAE:       {best_metrics['MAE']:.4f}")

                print(f"\nPrediction Statistics:")
                print(f"  Mean: {diagnostics['mean_pred']:.4f}")
                print(f"  Min:  {diagnostics['min_pred']:.4f}")
                print(f"  Max:  {diagnostics['max_pred']:.4f}")

                if diagnostics['warning']:
                    print(f"  ⚠️  WARNING: Mean prediction < 0.25 (under-confident model!)")

                history['val_metrics'].append(best_metrics)
                history['diagnostics'].append(diagnostics)

                # Save best model
                if best_metrics['IoU'] > best_iou:
                    best_iou = best_metrics['IoU']
                    print(f"\n✓ New best IoU: {best_iou:.4f}")

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_iou': best_iou,
                        'metrics': best_metrics,
                        'args': vars(args)
                    }, checkpoint_dir / 'best_model.pth')

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 and is_main_process:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'args': vars(args)
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

        # Print epoch time
        epoch_time = time.time() - epoch_start
        if is_main_process:
            print(f"\n⏱️  Epoch time: {epoch_time:.1f}s ({epoch_time/60:.1f}min)")

    # Save final model
    if is_main_process:
        print("\nSaving final model...")
        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': model.state_dict(),
            'best_iou': best_iou,
            'args': vars(args)
        }, checkpoint_dir / 'final_model.pth')

        # Save history
        with open(checkpoint_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best IoU: {best_iou:.4f}")
        print(f"Target:   0.6500 {'✓' if best_iou >= 0.65 else '✗'}")
        print(f"\nCheckpoints saved to: {checkpoint_dir}")
        print(f"{'='*70}\n")

        if best_iou >= 0.65:
            print("✅ SUCCESS! Loss function works correctly.")
            print("   You can now use it in full MoE training.\n")
        else:
            print("⚠️  IoU below target. Consider:")
            print("   - Increasing tversky_weight to 3.0")
            print("   - Increasing pos_weight to 5.0")
            print("   - Training for more epochs\n")

    # Cleanup DDP
    if args.use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
