"""
Train Single Expert Script

Trains ONE expert architecture (SINet, PraNet, or FSPNet) standalone
to reach SOTA performance, then saves checkpoint for MoE integration.

Uses ALL the same training features as train_advanced.py:
- Multi-scale processing
- Progressive augmentation
- Warmup epochs
- Cosine annealing LR
- Gradient clipping
- AMP training
- Deep supervision

Usage:
    python train_single_expert.py --expert sinet --epochs 50 --data-root ./combined_dataset --checkpoint-dir ./checkpoints_sinet --use-multi-scale
    python train_single_expert.py --expert pranet --epochs 50 --data-root ./combined_dataset --checkpoint-dir ./checkpoints_pranet --use-multi-scale
    python train_single_expert.py --expert fspnet --epochs 50 --data-root ./combined_dataset --checkpoint-dir ./checkpoints_fspnet --use-multi-scale
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import time
from tqdm import tqdm
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.expert_architectures import (
    SINetExpert, PraNetExpert, ZoomNetExpert,
    BASNetExpert, CPDExpert, GCPANetExpert
)
from models.fspnet_expert import FSPNetExpert
from models.zoomnext_expert import ZoomNeXtExpert
from models.multi_scale_processor import MultiScaleInputProcessor
from losses.sota_loss import SOTALoss
from dataset import COD10KDataset
from metrics.cod_metrics import CODMetrics
from trainers.optimized_trainer import CODProgressiveAugmentation


EXPERT_CLASSES = {
    'sinet': SINetExpert,
    'pranet': PraNetExpert,
    'zoomnet': ZoomNetExpert,
    'fspnet': FSPNetExpert,
    # New diverse experts
    'basnet': BASNetExpert,
    'cpd': CPDExpert,
    'gcpanet': GCPANetExpert,
    # SOTA TPAMI 2024
    'zoomnext': ZoomNeXtExpert,
}

BACKBONE_DIMS = {
    'pvt_v2_b0': [32, 64, 160, 256],
    'pvt_v2_b1': [64, 128, 320, 512],
    'pvt_v2_b2': [64, 128, 320, 512],
    'pvt_v2_b3': [64, 128, 320, 512],
    'pvt_v2_b4': [64, 128, 320, 512],
    'pvt_v2_b5': [64, 128, 320, 512],
    'resnet50': [256, 512, 1024, 2048],
}


class SingleExpertModel(nn.Module):
    """Single expert model with shared backbone for standalone training."""
    
    def __init__(self, backbone_name: str, expert_type: str, pretrained: bool = True,
                 use_multi_scale: bool = False, multi_scale_factors: list = None):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.expert_type = expert_type
        self.feature_dims = BACKBONE_DIMS.get(backbone_name, [64, 128, 320, 512])
        self.use_multi_scale = use_multi_scale
        
        # Create backbone using timm (same as ModelLevelMoE)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        # Enable gradient checkpointing if available
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        
        # Wrap with multi-scale processor if enabled
        if use_multi_scale:
            if multi_scale_factors is None:
                multi_scale_factors = [0.75, 1.0, 1.25]
            
            self.backbone = MultiScaleInputProcessor(
                backbone=self.backbone,
                channels_list=self.feature_dims,
                scales=multi_scale_factors,
                use_hierarchical=True
            )
            print(f"  ✓ Multi-scale processing enabled: {multi_scale_factors}")
        
        # Single expert
        expert_class = EXPERT_CLASSES[expert_type]
        self.expert = expert_class(feature_dims=self.feature_dims)
        
        # Count parameters
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        expert_params = sum(p.numel() for p in self.expert.parameters())
        
        print(f"\n{'='*60}")
        print(f"Single Expert Model: {expert_type.upper()}")
        print(f"Backbone: {backbone_name} ({backbone_params/1e6:.2f}M params)")
        print(f"Expert: {expert_type} ({expert_params/1e6:.2f}M params)")
        print(f"Multi-scale: {use_multi_scale}")
        print(f"Total: {(backbone_params + expert_params)/1e6:.2f}M params")
        print(f"{'='*60}\n")
    
    def forward(self, x, return_aux=True):
        """Forward pass through backbone + expert."""
        features = self.backbone(x)
        pred, aux = self.expert(features, return_aux=return_aux)
        return pred, aux


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, total_epochs, 
                use_amp=True, accumulation_steps=1, augmentation=None):
    """Train for one epoch with gradient accumulation and progressive augmentation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    # Update augmentation strength for this epoch
    if augmentation is not None:
        augmentation.update_epoch(epoch)
    
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{total_epochs}]", leave=True)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Apply progressive augmentation (GPU-side)
        if augmentation is not None and augmentation.current_strength > 0.1:
            if torch.rand(1).item() < augmentation.current_strength:
                images, masks = augmentation.apply(images, masks, augmentation_type='random')
        
        with torch.autocast(device_type='cuda', enabled=use_amp):
            pred, aux_outputs = model(images, return_aux=True)
            
            # Main loss
            loss = criterion(pred, masks)
            
            # Deep supervision
            if aux_outputs:
                for i, aux_pred in enumerate(aux_outputs[:2]):
                    if aux_pred is not None:
                        # Resize target if needed
                        if aux_pred.shape[2:] != masks.shape[2:]:
                            aux_target = nn.functional.interpolate(
                                masks, size=aux_pred.shape[2:], mode='nearest'
                            )
                        else:
                            aux_target = masks
                        weight = 0.4 * (0.7 ** i)
                        loss = loss + weight * criterion(aux_pred, aux_target)
            
            # Scale for accumulation
            loss = loss / accumulation_steps
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate and compute metrics."""
    model.eval()
    metrics = CODMetrics()
    
    for images, masks in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        pred, _ = model(images, return_aux=False)
        
        # Apply sigmoid if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        metrics.update(pred, masks)
    
    results = metrics.compute()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train Single Expert (SOTA Strategy)')
    
    # Required
    parser.add_argument('--expert', type=str, required=True, 
                       choices=['sinet', 'pranet', 'zoomnet', 'fspnet', 
                               'basnet', 'cpd', 'gcpanet', 'zoomnext'],
                       help='Expert type to train')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to dataset root')
    
    # Model
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                       help='Backbone architecture (default: pvt_v2_b2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained backbone')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                       help='Do not use pretrained backbone')
    
    # Data
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--img-size', type=int, default=448,
                       help='Image size (default: 448)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loader workers (default: 0 for Windows)')
    
    # Multi-scale (like train_advanced.py)
    parser.add_argument('--use-multi-scale', action='store_true', default=False,
                       help='Enable multi-scale processing')
    parser.add_argument('--multi-scale-factors', nargs='+', type=float,
                       default=[0.75, 1.0, 1.25],
                       help='Scale factors for multi-scale processing')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                       help='Warmup epochs (default: 10)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                       help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--accumulation-steps', type=int, default=2,
                       help='Gradient accumulation steps (default: 2)')
    
    # Loss weights (tune for better IoU/F-measure)
    parser.add_argument('--bce-weight', type=float, default=1.0,
                       help='BCE loss weight (default: 1.0)')
    parser.add_argument('--iou-weight', type=float, default=2.0,
                       help='IoU loss weight (default: 2.0 - higher for better IoU)')
    parser.add_argument('--dice-weight', type=float, default=2.0,
                       help='Dice loss weight (default: 2.0 - higher for better F-measure)')
    parser.add_argument('--structure-weight', type=float, default=0.5,
                       help='Structure loss weight (default: 0.5)')
    parser.add_argument('--pos-weight', type=float, default=5.0,
                       help='Foreground weight in BCE (default: 5.0 - handles class imbalance)')
    
    # AMP
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--no-amp', action='store_false', dest='use_amp',
                       help='Disable mixed precision')
    
    # Progressive augmentation (like train_advanced.py)
    parser.add_argument('--enable-progressive-aug', action='store_true', default=True,
                       help='Enable progressive augmentation')
    parser.add_argument('--aug-transition-epoch', type=int, default=25,
                       help='Epoch when progressive aug starts increasing (default: 25)')
    parser.add_argument('--aug-max-strength', type=float, default=1.0,
                       help='Maximum augmentation strength (default: 1.0)')
    
    # Checkpoint
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_expert',
                       help='Checkpoint directory')
    parser.add_argument('--val-freq', type=int, default=5,
                       help='Validation frequency (default: 5)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"SINGLE EXPERT TRAINING: {args.expert.upper()}")
    print(f"{'='*60}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.img_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    print(f"  Multi-scale: {args.use_multi_scale}")
    print(f"  AMP: {args.use_amp}")
    print(f"{'='*60}\n")
    
    # Create model with multi-scale if enabled
    model = SingleExpertModel(
        args.backbone, 
        args.expert, 
        pretrained=args.pretrained,
        use_multi_scale=args.use_multi_scale,
        multi_scale_factors=args.multi_scale_factors
    )
    model = model.to(device)
    
    # Dataset (no RAM caching - load from disk)
    train_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='train',
        img_size=args.img_size,
        augment=True,
        cache_in_memory=False
    )
    
    val_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='test',
        img_size=args.img_size,
        augment=False,
        cache_in_memory=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"Val: {len(val_dataset)} images")
    
    # Loss with tunable weights (higher IoU/Dice for COD10K)
    criterion = SOTALoss(
        bce_weight=args.bce_weight,
        iou_weight=args.iou_weight,
        dice_weight=args.dice_weight,
        structure_weight=args.structure_weight,
        pos_weight=args.pos_weight,
        aux_weight=0.0  # No MoE aux loss
    )
    
    # Optimizer with different LR for backbone
    backbone_params = list(model.backbone.parameters())
    expert_params = list(model.expert.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for pretrained backbone
        {'params': expert_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    # Cosine scheduler with warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr)
    scaler = torch.GradScaler() if args.use_amp else None
    
    # Progressive augmentation (for generalization)
    augmentation = None
    if args.enable_progressive_aug:
        augmentation = CODProgressiveAugmentation(
            initial_strength=0.3,
            max_strength=args.aug_max_strength,
            transition_epoch=args.aug_transition_epoch,
            transition_duration=args.epochs - args.aug_transition_epoch
        )
        print(f"\n✓ Progressive augmentation enabled:")
        print(f"  Transition epoch: {args.aug_transition_epoch}")
        print(f"  Max strength: {args.aug_max_strength}")
    
    # Resume
    start_epoch = 0
    best_sm = 0.0
    
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_sm = checkpoint.get('best_sm', 0.0)
        print(f"✓ Resumed from epoch {start_epoch}, best S-measure: {best_sm:.4f}")
    
    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Warmup LR
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0:  # backbone
                    param_group['lr'] = args.lr * 0.1 * warmup_factor
                else:  # expert
                    param_group['lr'] = args.lr * warmup_factor
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, 
            epoch, args.epochs, args.use_amp, args.accumulation_steps, augmentation
        )
        
        # Step scheduler after warmup
        if epoch > args.warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[1]['lr']  # Expert LR
        
        # Validate
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            results = validate(model, val_loader, device)
            sm = results.get('S-measure', 0.0)
            iou = results.get('IoU', 0.0)
            fm = results.get('F-measure', 0.0)
            mae = results.get('MAE', 0.0)
            
            print(f"\nEpoch [{epoch}/{args.epochs}] Results:")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  S-measure: {sm:.4f}")
            print(f"  IoU: {iou:.4f}")
            print(f"  F-measure: {fm:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best
            if sm > best_sm:
                best_sm = sm
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_sm': best_sm,
                    'best_iou': iou,
                    'expert_type': args.expert,
                    'backbone': args.backbone,
                    'use_multi_scale': args.use_multi_scale,
                    # Save expert weights separately for easy MoE loading
                    'expert_state_dict': model.expert.state_dict(),
                    'backbone_state_dict': model.backbone.state_dict()
                }, checkpoint_dir / 'best_model.pth')
                print(f"  ✓ New best! S-measure: {best_sm:.4f} ⭐")
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_sm': best_sm,
                'expert_type': args.expert,
                'backbone': args.backbone,
                'use_multi_scale': args.use_multi_scale,
                'expert_state_dict': model.expert.state_dict(),
                'backbone_state_dict': model.backbone.state_dict()
            }, checkpoint_dir / f'epoch_{epoch}.pth')
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_sm': best_sm,
            'expert_type': args.expert,
            'backbone': args.backbone,
            'use_multi_scale': args.use_multi_scale,
            'expert_state_dict': model.expert.state_dict(),
            'backbone_state_dict': model.backbone.state_dict()
        }, checkpoint_dir / 'latest.pth')
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Expert: {args.expert.upper()}")
    print(f"Best S-measure: {best_sm:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*60}")
    print(f"\nNext step: Use this checkpoint with load_pretrained_experts.py")


if __name__ == '__main__':
    main()
