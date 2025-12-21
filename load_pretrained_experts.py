"""
Load Pre-trained Experts into MoE

After training each expert individually with train_single_expert.py,
use this script to:
1. Load pre-trained expert weights into MoE model
2. Optionally train the router with frozen experts
3. Optionally fine-tune everything

Usage:
    # Load experts and train router only
    python load_pretrained_experts.py \
        --sinet-checkpoint ./checkpoints_sinet/best_model.pth \
        --pranet-checkpoint ./checkpoints_pranet/best_model.pth \
        --fspnet-checkpoint ./checkpoints_fspnet/best_model.pth \
        --mode router-only \
        --epochs 30 \
        --output-dir ./checkpoints_moe_pretrained
    
    # Fine-tune full model after loading
    python load_pretrained_experts.py \
        --sinet-checkpoint ./checkpoints_sinet/best_model.pth \
        --pranet-checkpoint ./checkpoints_pranet/best_model.pth \
        --fspnet-checkpoint ./checkpoints_fspnet/best_model.pth \
        --mode full \
        --epochs 50 \
        --output-dir ./checkpoints_moe_finetuned
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_level_moe import ModelLevelMoE
from losses.sota_loss import SOTALoss
from dataset import COD10KDataset
from metrics.cod_metrics import CODMetrics


def load_expert_weights(model: ModelLevelMoE, expert_checkpoints: dict, device: str):
    """
    Load pre-trained expert weights into MoE model.
    
    Args:
        model: ModelLevelMoE instance
        expert_checkpoints: Dict mapping expert_type -> checkpoint_path
        device: Device to load on
    """
    expert_order = model.expert_types  # ['sinet', 'pranet', 'fspnet']
    
    print("\n" + "="*60)
    print("Loading Pre-trained Expert Weights")
    print("="*60)
    
    for i, expert_type in enumerate(expert_order):
        if expert_type in expert_checkpoints and expert_checkpoints[expert_type]:
            ckpt_path = expert_checkpoints[expert_type]
            print(f"\n[{i}] {expert_type.upper()}:")
            
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                
                # Load expert weights (saved separately)
                if 'expert_state_dict' in checkpoint:
                    model.expert_models[i].load_state_dict(checkpoint['expert_state_dict'])
                    print(f"    ✓ Loaded expert weights from: {ckpt_path}")
                    print(f"    ✓ Best S-measure: {checkpoint.get('best_sm', 'N/A')}")
                else:
                    print(f"    ⚠ No expert_state_dict found, trying full model...")
                    # Try to extract from full model state dict
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    expert_state = {k.replace('expert.', ''): v 
                                   for k, v in state_dict.items() 
                                   if k.startswith('expert.')}
                    if expert_state:
                        model.expert_models[i].load_state_dict(expert_state)
                        print(f"    ✓ Loaded expert weights from model state dict")
            else:
                print(f"    ❌ Checkpoint not found: {ckpt_path}")
        else:
            print(f"\n[{i}] {expert_type.upper()}: No checkpoint provided (random init)")
    
    # Load backbone weights from first available checkpoint
    for expert_type in expert_order:
        if expert_type in expert_checkpoints and expert_checkpoints[expert_type]:
            ckpt_path = expert_checkpoints[expert_type]
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                if 'backbone_state_dict' in checkpoint:
                    backbone_state = checkpoint['backbone_state_dict']
                    # Handle key prefix mismatch - strip 'backbone.' if present
                    fixed_state = {}
                    for k, v in backbone_state.items():
                        if k.startswith('backbone.'):
                            fixed_state[k.replace('backbone.', '')] = v
                        else:
                            fixed_state[k] = v
                    model.backbone.load_state_dict(fixed_state)
                    print(f"\n✓ Loaded backbone weights from: {ckpt_path}")
                    break
                else:
                    # Try loading from model_state_dict
                    state_dict = checkpoint.get('model_state_dict', {})
                    backbone_state = {k.replace('backbone.', ''): v 
                                     for k, v in state_dict.items() 
                                     if k.startswith('backbone.')}
                    if backbone_state:
                        model.backbone.load_state_dict(backbone_state)
                        print(f"\n✓ Loaded backbone weights from model_state_dict: {ckpt_path}")
                        break
    
    print("="*60 + "\n")
    
    return model


def freeze_experts(model: ModelLevelMoE):
    """Freeze all expert parameters."""
    for expert in model.expert_models:
        for param in expert.parameters():
            param.requires_grad = False
    print("✓ All experts frozen")


def unfreeze_experts(model: ModelLevelMoE):
    """Unfreeze all expert parameters."""
    for expert in model.expert_models:
        for param in expert.parameters():
            param.requires_grad = True
    print("✓ All experts unfrozen")


def freeze_backbone(model: ModelLevelMoE):
    """Freeze backbone parameters."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("✓ Backbone frozen")


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{total_epochs}]", leave=True)
    
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type='cuda', enabled=True):
            outputs = model(images, return_routing_info=True)
            
            if isinstance(outputs, tuple):
                pred, routing_info = outputs
                aux_outputs = routing_info.get('aux_outputs', None)
                expert_preds = routing_info.get('individual_expert_preds', None)
                lb_loss = routing_info.get('load_balance_loss', None)
            else:
                pred = outputs
                aux_outputs = None
                expert_preds = None
                lb_loss = None
            
            # Main loss with per-expert gradients
            loss = criterion(pred, masks, expert_preds=expert_preds)
            
            # Add load balancing loss
            if lb_loss is not None:
                loss = loss + 0.5 * lb_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate and compute metrics."""
    model.eval()
    metrics = CODMetrics()
    
    for images, masks in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        pred = model(images, return_routing_info=False)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = torch.sigmoid(pred)
        
        metrics.update(pred, masks)
    
    results = metrics.compute()
    return results


def main():
    parser = argparse.ArgumentParser(description='Load Pre-trained Experts into MoE')
    parser.add_argument('--sinet-checkpoint', type=str, default=None)
    parser.add_argument('--pranet-checkpoint', type=str, default=None)
    parser.add_argument('--fspnet-checkpoint', type=str, default=None)
    parser.add_argument('--zoomnet-checkpoint', type=str, default=None)
    parser.add_argument('--expert-types', nargs='+', default=['sinet', 'pranet', 'fspnet'])
    parser.add_argument('--mode', type=str, choices=['router-only', 'full', 'save-only'], 
                       default='router-only',
                       help='router-only: freeze experts, train router only; '
                            'full: train everything; '
                            'save-only: just save combined model')
    parser.add_argument('--data-root', type=str, default='./combined_dataset')
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=448)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='./checkpoints_moe_pretrained')
    parser.add_argument('--val-freq', type=int, default=5)
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create MoE model
    model = ModelLevelMoE(
        backbone_name=args.backbone,
        num_experts=len(args.expert_types),
        top_k=2,
        pretrained=False,  # Will load from checkpoints
        expert_types=args.expert_types
    )
    
    # Build checkpoint dict
    expert_checkpoints = {
        'sinet': args.sinet_checkpoint,
        'pranet': args.pranet_checkpoint,
        'fspnet': args.fspnet_checkpoint,
        'zoomnet': args.zoomnet_checkpoint
    }
    
    # Load pre-trained weights
    model = load_expert_weights(model, expert_checkpoints, device)
    model = model.to(device)
    
    # Save-only mode: just save combined model and exit
    if args.mode == 'save-only':
        save_path = output_dir / 'moe_with_pretrained_experts.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'expert_types': args.expert_types,
            'backbone': args.backbone
        }, save_path)
        print(f"\n✓ Saved combined MoE model to: {save_path}")
        print("  Use --resume-from with train_advanced.py to continue training")
        return
    
    # Apply freezing based on mode
    if args.mode == 'router-only':
        freeze_experts(model)
        freeze_backbone(model)
        print("\nMode: Router-only (experts & backbone frozen)")
    else:
        print("\nMode: Full training (everything unfrozen)")
    
    # Dataset
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
        num_workers=0, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    
    # Loss and optimizer
    criterion = SOTALoss(aux_weight=0.5)
    
    # Only optimize unfrozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.GradScaler()
    
    # Training loop
    best_sm = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args.epochs
        )
        scheduler.step()
        
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            results = validate(model, val_loader, device)
            sm = results.get('S-measure', 0.0)
            iou = results.get('IoU', 0.0)
            fm = results.get('F-measure', 0.0)
            
            print(f"\nEpoch [{epoch}/{args.epochs}] Results:")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  S-measure: {sm:.4f}")
            print(f"  IoU: {iou:.4f}")
            print(f"  F-measure: {fm:.4f}")
            
            if sm > best_sm:
                best_sm = sm
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_sm': best_sm,
                    'expert_types': args.expert_types,
                    'backbone': args.backbone,
                    'mode': args.mode
                }, output_dir / 'best_model.pth')
                print(f"  ✓ Saved best model (S-measure: {best_sm:.4f})")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_sm': best_sm,
            'expert_types': args.expert_types,
            'backbone': args.backbone,
            'mode': args.mode
        }, output_dir / 'latest.pth')
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best S-measure: {best_sm:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
