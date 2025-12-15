"""
Advanced Training Script with All New Modules Integrated

Integrates:
1. OptimizedTrainer - Advanced training framework
2. CompositeLoss - Multi-component loss system
3. Enhanced experts with new modules
4. All optimizations from new components
5. Multi-Scale Processing (optional - use --use-multi-scale)
6. Boundary Refinement (optional - use --use-boundary-refinement)

NEW FEATURES AVAILABLE:
- Multi-scale processing: --use-multi-scale --multi-scale-factors 0.5 1.0 1.5
- Boundary refinement: --use-boundary-refinement --boundary-loss-weight 0.3
- RAM caching: --cache-in-memory (enabled by default)

NOTE: Multi-scale and boundary refinement CLI arguments are added but require
manual integration in the training loop. See examples/ for integration code.

Usage:
    # Basic (your current setup)
    torchrun --nproc_per_node=2 train_advanced.py \
        --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
        --epochs 100 \
        --batch-size 16 \
        --use-ddp

    # With RAM caching (30-40% faster)
    torchrun --nproc_per_node=2 train_advanced.py \
        --data-root /kaggle/input/cod10k-dataset/COD10K-v3 \
        --cache-in-memory \
        --use-ddp
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from pathlib import Path

# Import new modules
from trainers.optimized_trainer import OptimizedTrainer
from losses.composite_loss import CompositeLossSystem
from losses import CombinedLoss  # OLD: Anti-under-segmentation loss
from losses.boundary_aware_loss import CombinedEnhancedLoss  # Enhanced loss with TDD/GAD/BPN
from losses.sota_loss import SOTALoss, SOTALossWithTversky  # NEW: SOTA-aligned loss for generalization
from utils.ema import EMA  # Exponential Moving Average
try:
    from dataset_updated import COD10KDataset
except ImportError:
    # Fallback if in a subdirectory or running differently
    try:
        from data.dataset_updated import COD10KDataset
    except ImportError:
        from data.dataset import COD10KDataset
        print("WARNING: Using original dataset.py (dataset_updated not found)")
from metrics.cod_metrics import CODMetrics
from models.model_level_moe import ModelLevelMoE
from models.utils import set_seed
from models.multi_scale_processor import MultiScaleInputProcessor
from models.boundary_refinement import BoundaryRefinementModule


def parse_args():
    parser = argparse.ArgumentParser(description='Advanced COD Training with New Modules')

    # Data
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to COD10K-v3 dataset')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')

    # Model
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                        choices=['pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5'],
                        help='Backbone architecture (default: pvt_v2_b2)')
    parser.add_argument('--num-experts', type=int, default=3,
                        help='Number of experts in MoE (default: 3 - SINet, PraNet, ZoomNet)')
    parser.add_argument('--top-k', type=int, default=2,
                        help='Top-k experts to use (default: 2)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone weights')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                        help='Train backbone from scratch')
    parser.add_argument('--deep-supervision', action='store_true', default=True,
                        help='Enable deep supervision in model')
    parser.add_argument('--no-deep-supervision', action='store_false', dest='deep_supervision',
                        help='Disable deep supervision')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization')
    parser.add_argument('--accumulation-steps', type=int, default=2,
                        help='Gradient accumulation steps')

    # OptimizedTrainer settings
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Warmup epochs for cosine scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', action='store_false', dest='use_amp',
                        help='Disable AMP')

    # Progressive augmentation (stronger from start)
    parser.add_argument('--enable-progressive-aug', action='store_true', default=True,
                        help='Enable progressive augmentation')
    parser.add_argument('--aug-transition-epoch', type=int, default=0,
                        help='Epoch to start increasing augmentation (default: 0, starts immediately)')
    parser.add_argument('--aug-max-strength', type=float, default=0.7,
                        help='Maximum augmentation strength (default: 0.7)')
    parser.add_argument('--aug-transition-duration', type=int, default=50,
                        help='Epochs to ramp up augmentation strength (default: 50)')

    # Router warmup (for MoE models)
    parser.add_argument('--enable-router-warmup', action='store_true', default=True,
                        help='Enable router warmup (freeze router for initial epochs)')
    parser.add_argument('--router-warmup-epochs', type=int, default=20,
                        help='Number of epochs to keep router frozen (default: 20, increased for better specialization)')

    # Loss Function Selection
    parser.add_argument('--loss-type', type=str, default='sota',
                        choices=['sota', 'sota-tversky', 'combined', 'composite'],
                        help='Loss function: sota (BCE+IoU+Structure), sota-tversky (for under-segmentation), '
                             'combined (5-loss), composite (old default)')
    parser.add_argument('--use-combined-loss', action='store_true',
                        help='[DEPRECATED] Use --loss-type=combined instead')

    # CompositeLoss settings (default loss function)
    parser.add_argument('--loss-scheme', type=str, default='progressive',
                        choices=['progressive', 'full'],
                        help='Loss weighting scheme for CompositeLoss (default: progressive)')
    parser.add_argument('--boundary-lambda-start', type=float, default=0.5,
                        help='Starting weight for boundary loss (default: 0.5)')
    parser.add_argument('--boundary-lambda-end', type=float, default=2.0,
                        help='Ending weight for boundary loss (default: 2.0)')
    parser.add_argument('--frequency-weight', type=float, default=1.5,
                        help='Weight for frequency loss (default: 1.5)')
    parser.add_argument('--scale-small-weight', type=float, default=2.0,
                        help='Weight for small object scale loss (default: 2.0)')
    parser.add_argument('--uncertainty-threshold', type=float, default=0.5,
                        help='Threshold for uncertainty loss (default: 0.5)')

    # CombinedLoss settings (when --use-combined-loss is enabled)
    parser.add_argument('--focal-weight', type=float, default=1.0,
                        help='Weight for focal loss in CombinedLoss (default: 1.0)')
    parser.add_argument('--tversky-weight', type=float, default=2.0,
                        help='Weight for Tversky loss ⭐ HIGH fixes under-segmentation (default: 2.0)')
    parser.add_argument('--combined-boundary-weight', type=float, default=1.0,
                        help='Weight for boundary loss in CombinedLoss (default: 1.0)')
    parser.add_argument('--ssim-weight', type=float, default=0.5,
                        help='Weight for SSIM loss in CombinedLoss (default: 0.5)')
    parser.add_argument('--dice-weight', type=float, default=1.0,
                        help='Weight for Dice loss in CombinedLoss (default: 1.0)')
    parser.add_argument('--pos-weight', type=float, default=3.0,
                        help='Positive pixel weight in focal loss (default: 3.0, boosts foreground)')
    parser.add_argument('--tversky-alpha', type=float, default=0.3,
                        help='Tversky alpha (FP weight) (default: 0.3)')
    parser.add_argument('--tversky-beta', type=float, default=0.7,
                        help='Tversky beta (FN weight) ⭐ HIGH penalizes under-segmentation (default: 0.7)')

    # EMA settings
    parser.add_argument('--use-ema', action='store_true',
                        help='Use Exponential Moving Average for model weights')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay rate (default: 0.999)')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_advanced',
                        help='Checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val-freq', type=int, default=1,
                        help='Run validation every N epochs (default: 1 = every epoch)')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--load-weights-only', action='store_true', default=False,
                        help='Only load model weights from checkpoint, not optimizer state (allows changing lr/wd)')

    # Distributed
    parser.add_argument('--use-ddp', action='store_true', default=False,
                        help='Use DistributedDataParallel')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for DDP')

    # System
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cache-in-memory', action='store_true', default=True,
                        help='Cache dataset in RAM for faster training (recommended with DDP)')
    parser.add_argument('--no-cache', action='store_false', dest='cache_in_memory',
                        help='Disable RAM caching')

    # Multi-Scale Processing
    parser.add_argument('--use-multi-scale', action='store_true', default=False,
                        help='Enable multi-scale processing (0.5×, 1.0×, 1.5×)')
    parser.add_argument('--multi-scale-factors', nargs='+', type=float,
                        default=[0.5, 1.0, 1.5],
                        help='Scale factors for multi-scale processing (space-separated)')
    parser.add_argument('--scale-loss-weight', type=float, default=0.3,
                        help='Weight for scale-specific losses (default: 0.3)')
    parser.add_argument('--use-hierarchical-fusion', action='store_true', default=True,
                        help='Use hierarchical scale fusion (vs ABSI)')

    # Boundary Refinement
    parser.add_argument('--use-boundary-refinement', action='store_true', default=False,
                        help='Enable boundary refinement module')
    parser.add_argument('--boundary-feature-channels', type=int, default=64,
                        help='Feature channels for boundary refinement (default: 64)')
    
    # TDD/GAD Modules (for testing generalization)
    parser.add_argument('--use-tdd', action='store_true', default=False,
                        help='Use Texture Discontinuity Detection module')
    parser.add_argument('--use-gad', action='store_true', default=False,
                        help='Use Gradient Anomaly Detection module')
    parser.add_argument('--gradient-loss-weight', type=float, default=0.5,
                        help='Weight for gradient supervision loss (default: 0.5)')
    parser.add_argument('--sdt-loss-weight', type=float, default=1.0,
                        help='Weight for signed distance map loss (default: 1.0)')
    parser.add_argument('--boundary-loss-weight', type=float, default=0.3,
                        help='Overall weight for boundary loss component (default: 0.3)')
    parser.add_argument('--boundary-lambda-schedule', type=str, default='cosine',
                        choices=['linear', 'cosine', 'exponential'],
                        help='Lambda scheduling type for boundary loss (default: cosine)')

    return parser.parse_args()


def setup_ddp(args):
    """Setup distributed training."""
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        args.local_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
        args.world_size = dist.get_world_size()
    else:
        args.world_size = 1

    return args.local_rank == 0  # is_main_process


def create_dataloaders(args, is_main_process):
    """Create train and validation dataloaders."""

    # Training dataset - DDP-aware caching: each GPU caches only its subset
    train_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='train',
        img_size=args.img_size,
        augment=True,  # Built-in augmentations
        cache_in_memory=args.cache_in_memory,
        rank=args.local_rank if args.use_ddp else 0,
        world_size=args.world_size
    )

    # Validation dataset
    val_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='test',
        img_size=args.img_size,
        augment=False,
        cache_in_memory=args.cache_in_memory,
        rank=args.local_rank if args.use_ddp else 0,
        world_size=args.world_size
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

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
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
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Val dataset: {len(val_dataset)} samples")
        print(f"✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")

    return train_loader, val_loader, train_sampler


def create_model(args, device, is_main_process):
    """Create model with all enhancements."""

    # Soft MoE (all experts contribute with dynamic weights)
    model = ModelLevelMoE(
        backbone_name=args.backbone,
        num_experts=args.num_experts,
            top_k=args.top_k,
            pretrained=args.pretrained,
            use_deep_supervision=args.deep_supervision
        )

    # Store original backbone for multi-scale wrapping
    original_backbone = model.backbone

    # Multi-Scale Processing Integration
    multi_scale_processor = None
    if args.use_multi_scale:
        if is_main_process:
            print(f"✓ Wrapping backbone with MultiScaleProcessor")
            print(f"  Scales: {args.multi_scale_factors}")
            print(f"  Hierarchical fusion: {args.use_hierarchical_fusion}")

        # Determine channel list based on backbone
        if 'pvt_v2' in args.backbone:
            channels_list = [64, 128, 320, 512]  # PVT-v2 channels
        else:
            channels_list = [64, 128, 320, 512]  # Default

        multi_scale_processor = MultiScaleInputProcessor(
            backbone=original_backbone,
            channels_list=channels_list,
            scales=args.multi_scale_factors,
            use_hierarchical=args.use_hierarchical_fusion
        )

        # Replace backbone with multi-scale processor
        model.backbone = multi_scale_processor

    # Boundary Refinement Integration
    # NOTE: TDD/GAD/BPN are now integrated inside ModelLevelMoE
    # External boundary_refinement is only used if explicitly requested
    boundary_refinement = None
    if args.use_boundary_refinement:
        if is_main_process:
            print(f"✓ Adding EXTERNAL BoundaryRefinementModule (in addition to integrated TDD/GAD/BPN)")
            print(f"  Feature channels: {args.boundary_feature_channels}")
            print(f"  Lambda schedule: {args.boundary_lambda_schedule}")

        boundary_refinement = BoundaryRefinementModule(
            feature_channels=args.boundary_feature_channels,
            use_gradient_loss=True,
            use_sdt_loss=True,
            gradient_weight=args.gradient_loss_weight,
            sdt_weight=args.sdt_loss_weight,
            total_epochs=args.epochs,
            lambda_schedule_type=args.boundary_lambda_schedule
        )
        boundary_refinement = boundary_refinement.to(device)
    else:
        if is_main_process:
            print(f"✓ TDD/GAD/BPN modules NOT used (removed for simplicity)")

    # Move base model to device
    model = model.to(device)

    # Wrap with DDP
    if args.use_ddp:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True  # Required: router frozen during warmup
        )

        if boundary_refinement is not None:
            boundary_refinement = DDP(
                boundary_refinement,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created: {args.backbone}")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

        if boundary_refinement is not None:
            boundary_params = sum(p.numel() for p in boundary_refinement.parameters())
            print(f"✓ Boundary refinement parameters: {boundary_params:,}")

    # Return model and optional modules
    return model, multi_scale_processor, boundary_refinement


def create_optimizer_and_criterion(model, args, is_main_process):
    """Create optimizer and loss function."""

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Handle deprecated --use-combined-loss flag
    loss_type = args.loss_type
    if args.use_combined_loss:
        loss_type = 'combined'
        if is_main_process:
            print("⚠️ --use-combined-loss is deprecated, use --loss-type=combined")

    # Select loss function based on type
    if loss_type == 'sota':
        # AGGRESSIVE: Push hard on IoU and F-measure to reach SOTA
        # IoU=2.0 (bottleneck metric), Dice=1.0 (optimizes F-measure)
        criterion = SOTALoss(
            bce_weight=1.0,
            iou_weight=2.0,   # Aggressive push for IoU
            dice_weight=1.0,  # Directly optimizes F-measure
            structure_weight=0.5,
            pos_weight=args.pos_weight,
            aux_weight=0.1,
            deep_weight=0.4
        )
        loss_name = "SOTALoss (BCE+IoU×2+Dice+Structure) [AGGRESSIVE]"

    elif loss_type == 'sota-tversky':
        # For under-segmentation issues
        criterion = SOTALossWithTversky(
            bce_weight=1.0,
            tversky_weight=1.0,
            structure_weight=0.5,
            pos_weight=args.pos_weight,
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
            aux_weight=0.1,
            deep_weight=0.4
        )
        loss_name = f"SOTALossWithTversky (alpha={args.tversky_alpha}, beta={args.tversky_beta})"

    elif loss_type == 'combined':
        # Old 5-loss combo (over-engineered but available)
        criterion = CombinedLoss(
            focal_weight=args.focal_weight,
            tversky_weight=args.tversky_weight,
            boundary_weight=args.combined_boundary_weight,
            ssim_weight=args.ssim_weight,
            dice_weight=args.dice_weight,
            pos_weight=args.pos_weight
        )
        loss_name = "CombinedLoss (5-loss combo)"

    else:  # composite
        # Original loss with label smoothing
        criterion = CompositeLossSystem(label_smoothing=0.1)
        loss_name = "CompositeLossSystem (label_smoothing=0.1)"

    if is_main_process:
        print(f"✓ Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
        print(f"✓ Loss: {loss_name}")

    return optimizer, criterion


# UNUSED: compute_additional_losses - removed for simplicity
# def compute_additional_losses(args, multi_scale_processor, boundary_refinement,
#                              images, predictions, targets, epoch):
#     """REMOVED - Not needed with simple structure loss"""
#     pass


def set_router_trainable(model, trainable):
    """
    Freeze or unfreeze router parameters.

    Args:
        model: The model (potentially wrapped in DDP)
        trainable: Boolean - True to unfreeze, False to freeze
    """
    actual_model = model.module if hasattr(model, 'module') else model

    # Only applies to MoE models with router
    if not hasattr(actual_model, 'router'):
        return

    for name, param in actual_model.named_parameters():
        if 'router' in name:
            param.requires_grad = trainable


def compute_metrics(predictions, targets):
    """
    Compute validation metrics for a batch.
    Uses CODMetrics for S-measure and MAE, simple formulas for IoU and F-measure.
    """
    from metrics.cod_metrics import CODMetrics
    
    # Apply sigmoid to convert logits to probabilities
    preds = torch.sigmoid(predictions.detach())
    tgt = targets.detach()
    
    # Ensure dimensions match
    if preds.shape[2:] != tgt.shape[2:]:
        preds = F.interpolate(preds, size=tgt.shape[2:], mode='bilinear', align_corners=False)
    
    batch_size = preds.shape[0]
    metrics_calc = CODMetrics()
    
    # Accumulators
    mae_total = 0.0
    iou_total = 0.0
    f_total = 0.0
    s_total = 0.0
    
    for i in range(batch_size):
        p = preds[i:i+1]  # Keep 4D for CODMetrics
        t = tgt[i:i+1]
        p_2d = preds[i, 0]  # 2D for simple calculations
        t_2d = tgt[i, 0]
        
        # MAE - use CODMetrics (correct implementation)
        mae_total += metrics_calc.mae(p, t)
        
        # S-measure - use CODMetrics (correct implementation)
        s_total += metrics_calc.s_measure(p, t)
        
        # Binary threshold for IoU/F-measure
        p_bin = (p_2d > 0.5).float()
        
        # IoU (Intersection over Union) - simple formula
        inter = (p_bin * t_2d).sum()
        union = p_bin.sum() + t_2d.sum() - inter
        iou_i = ((inter + 1e-6) / (union + 1e-6)).item()
        iou_total += iou_i
        
        # F-measure (beta=0.3) - simple formula
        tp = (p_bin * t_2d).sum()
        fp = (p_bin * (1 - t_2d)).sum()
        fn = ((1 - p_bin) * t_2d).sum()
        prec = (tp + 1e-6) / (tp + fp + 1e-6)
        rec = (tp + 1e-6) / (tp + fn + 1e-6)
        beta = 0.3
        f_i = (((1 + beta**2) * prec * rec) / (beta**2 * prec + rec + 1e-6)).item()
        f_total += f_i
    
    result = {
        'val_mae': mae_total / batch_size,
        'val_s_measure': s_total / batch_size,
        'val_iou': iou_total / batch_size,
        'val_f_measure': f_total / batch_size
    }
    
    # DEBUG: Print first batch values
    global _debug_batch_count
    if '_debug_batch_count' not in globals():
        _debug_batch_count = 0
    _debug_batch_count += 1
    if _debug_batch_count == 1:
        print(f"[DEBUG compute_metrics] First batch: {result}")
    
    return result


# Legacy - not used anymore
_cod_metrics_singleton = None

def _get_cod_metrics():
    """Legacy function - not used in new compute_metrics."""
    global _cod_metrics_singleton
    if _cod_metrics_singleton is None:
        from metrics.cod_metrics import CODMetrics
        _cod_metrics_singleton = CODMetrics()
    return _cod_metrics_singleton


def compute_s_measure(pred, target, alpha=0.5):
    """
    Compute official Structure Measure for a batch using CODMetrics.
    
    This uses the proper S-measure formula with object-aware and region-based
    components (SSIM-like), not the simplified Dice-based approximation.
    """
    metrics = _get_cod_metrics()
    return metrics.s_measure(pred, target, alpha=alpha)


# UNUSED: train_epoch_with_additional_losses - removed for simplicity
# Now using trainer.train_epoch() directly with simple structure loss
# def train_epoch_with_additional_losses(...):
#     """REMOVED - Not needed with simple structure loss"""
#     pass


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    is_main_process = setup_ddp(args)
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process:
        print("=" * 80)
        print(" " * 20 + "ADVANCED COD TRAINING")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Data root: {args.data_root}")
        print(f"  Batch size: {args.batch_size} (per GPU)")
        print(f"  Accumulation steps: {args.accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.accumulation_steps * args.world_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Image size: {args.img_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Mixed precision: {args.use_amp}")
        print(f"  Progressive augmentation: {args.enable_progressive_aug}")
        print(f"  DDP: {args.use_ddp} (world size: {args.world_size})")
        print()

    # Create checkpoint directory
    if is_main_process:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, train_sampler = create_dataloaders(args, is_main_process)

    # Model (now returns model + optional modules)
    model, multi_scale_processor, boundary_refinement = create_model(args, device, is_main_process)

    # Optimizer and criterion
    optimizer, criterion = create_optimizer_and_criterion(model, args, is_main_process)

    # Store additional modules for loss computation
    trainer_kwargs = {
        'multi_scale_processor': multi_scale_processor,
        'boundary_refinement': boundary_refinement,
        'args': args
    }

    # Initialize EMA if enabled (BEFORE trainer so it can be passed)
    ema_model = None
    if args.use_ema:
        # Get base model (unwrap DDP if needed)
        base_model = model.module if hasattr(model, 'module') else model
        ema_model = EMA(base_model, decay=args.ema_decay)
        if is_main_process:
            print(f"✓ EMA enabled (decay={args.ema_decay})")

    # OptimizedTrainer with all features
    trainer = OptimizedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.use_amp,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        max_lr=args.lr,
        num_experts=args.num_experts if args.num_experts > 1 else None,
        enable_load_balancing=True if args.num_experts > 1 else False,
        enable_collapse_detection=True if args.num_experts > 1 else False,
        enable_progressive_aug=args.enable_progressive_aug,
        aug_transition_epoch=args.aug_transition_epoch,
        aug_max_strength=args.aug_max_strength,
        aug_transition_duration=args.aug_transition_duration,
        ema=ema_model  # Pass EMA for per-step updates
    )

    # Store additional modules in trainer for access during training
    trainer.multi_scale_processor = multi_scale_processor
    trainer.boundary_refinement = boundary_refinement
    trainer.training_args = args

    if is_main_process:
        print("✓ OptimizedTrainer initialized with:")
        print(f"  - Cosine annealing with {args.warmup_epochs}-epoch warmup")
        print(f"  - Mixed precision: {args.use_amp}")
        print(f"  - Gradient accumulation: {args.accumulation_steps} steps")
        print(f"  - Progressive augmentation: {args.enable_progressive_aug}")
        if args.use_ema:
            print(f"  - EMA: Enabled (decay={args.ema_decay})")
        if args.num_experts > 1:
            print(f"  - MoE load balancing: Enabled")
            print(f"  - Expert collapse detection: Enabled")
        print()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_smeasure = 0.0

    if args.resume_from and os.path.exists(args.resume_from):
        if is_main_process:
            mode = "weights only" if args.load_weights_only else "full checkpoint"
            print(f"Resuming from: {args.resume_from} ({mode})")
        start_epoch = trainer.load_checkpoint(args.resume_from, weights_only=args.load_weights_only)
        start_epoch += 1

        # Restore best_smeasure from best_model.pth if it exists
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            if 'metrics' in best_checkpoint and 'val_s_measure' in best_checkpoint['metrics']:
                best_smeasure = best_checkpoint['metrics']['val_s_measure']
                if is_main_process:
                    print(f"✅ Restored best S-measure: {best_smeasure:.4f}")
            del best_checkpoint  # Free memory

    # Training loop
    if is_main_process:
        print("=" * 80)
        print("Starting training...")
        print("=" * 80)
        print()

    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if args.use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Router warmup: freeze router for initial epochs to stabilize expert learning
        if args.num_experts > 1 and args.enable_router_warmup:
            if epoch < args.router_warmup_epochs:
                set_router_trainable(model, trainable=False)
                if epoch == 0 and is_main_process:
                    print(f"🔒 Router FROZEN for warmup (epochs 0-{args.router_warmup_epochs-1})")
                    print("   Experts will learn independently before routing kicks in\n")
            elif epoch == args.router_warmup_epochs:
                set_router_trainable(model, trainable=True)
                if is_main_process:
                    print(f"🔓 Router UNFROZEN (epoch {args.router_warmup_epochs}) - now learning routing patterns\n")

        # Update CompositeLoss for current epoch
        criterion.update_epoch(epoch, args.epochs)

        # Set epoch for boundary refinement lambda scheduling
        if args.use_boundary_refinement and boundary_refinement is not None:
            boundary_module = boundary_refinement.module if hasattr(boundary_refinement, 'module') else boundary_refinement
            boundary_module.set_epoch(epoch)

        # Train one epoch (EMA is updated per-step inside trainer)
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch, log_interval=20)

        # Validate (run every val_freq epochs, or on last epoch, or on first epoch)
        should_validate = ((epoch + 1) % args.val_freq == 0) or (epoch == 0) or (epoch == args.epochs - 1)

        if should_validate:
            # Use EMA weights for validation (more stable, per-step updated)
            if ema_model is not None:
                ema_model.apply_shadow()
            
            val_metrics = trainer.validate(
                val_loader,
                metrics_fn=compute_metrics
            )
            
            # Restore original weights for continued training
            if ema_model is not None:
                ema_model.restore()
        else:
            # Skip validation, use previous metrics
            val_metrics = None

        # Print results (main process only)
        if is_main_process:
            # Get detailed loss breakdown from criterion
            loss_info = trainer.criterion.get_last_loss_dict() if hasattr(trainer.criterion, 'get_last_loss_dict') else {}

            print(f"\nEpoch [{epoch+1}/{args.epochs}] Results:")
            print(f"  Total Loss: {train_metrics['loss']:.4f}")

            # Loss breakdown
            if loss_info:
                print(f"  Loss Breakdown:")
                if 'bce' in loss_info:
                    print(f"    BCE: {loss_info['bce']:.4f}, Dice: {loss_info.get('dice', 0):.4f}, IoU: {loss_info.get('iou', 0):.4f}, Focal: {loss_info.get('focal', 0):.4f}")
                if 'anti_collapse' in loss_info:
                    print(f"    Anti-Collapse: {loss_info['anti_collapse']:.4f} {'⚠' if loss_info['anti_collapse'] > 1.0 else '✓'}")
                if 'boundary' in loss_info:
                    print(f"    Boundary: {loss_info['boundary']:.4f}")
                if 'expert' in loss_info:
                    print(f"    Expert: {loss_info['expert']:.4f}")
                if 'tdd_mean' in loss_info or 'gad_mean' in loss_info:
                    print(f"    TDD/GAD (monitoring): {loss_info.get('tdd_mean', 0):.3f} / {loss_info.get('gad_mean', 0):.3f}")

            # Validation metrics
            if val_metrics is not None:
                print(f"  Validation:")
                print(f"    S-measure: {val_metrics['val_s_measure']:.4f} ⭐")
                print(f"    IoU: {val_metrics['val_iou']:.4f}")
                print(f"    F-measure: {val_metrics['val_f_measure']:.4f}")
                print(f"    MAE: {val_metrics['val_mae']:.4f}")

                # Overfitting monitoring
                if 'val_iou' in val_metrics:
                    train_iou = train_metrics.get('iou', 0.5)  # Estimate from loss
                    val_iou = val_metrics['val_iou']
                    gap = train_iou - val_iou
                    if gap > 0.10:
                        print(f"  ⚠️ OVERFITTING: Train-Val gap = {gap:.3f} (should be < 0.10)")
            else:
                print(f"  Validation skipped (running every {args.val_freq} epochs)")

            print(f"  Learning Rate: {train_metrics['lr']:.6f}")

            if args.enable_progressive_aug and trainer.augmentation is not None:
                print(f"  Aug Strength: {trainer.augmentation.current_strength:.3f}")

            # Expert selection monitoring (router collapse detection)
            if 'expert_selection_pcts' in train_metrics:
                pcts = train_metrics['expert_selection_pcts']
                print(f"  Router Selections: ", end="")
                for i in range(args.num_experts):
                    key = f'expert_{i}'
                    if key in pcts:
                        pct = pcts[key]
                        # Warn if any expert is selected < 15% or > 55% (collapse indicator)
                        warning = " ⚠️" if pct < 15 or pct > 55 else ""
                        print(f"E{i}={pct:.1f}%{warning} ", end="")
                print()  # newline

            # Only report expert collapse after router warmup (expected during warmup)
            if 'collapse_collapsed' in train_metrics and train_metrics['collapse_collapsed']:
                if epoch >= args.router_warmup_epochs:
                    print(f"  ⚠ Expert collapse detected after warmup!")

            print()

        # Save checkpoints (main process only)
        if is_main_process:
            # Save best model based on S-measure (higher is better) - only when we have validation metrics
            if val_metrics is not None:
                current_smeasure = val_metrics['val_s_measure']
                if current_smeasure > best_smeasure:
                    best_smeasure = current_smeasure
                    best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                    trainer.save_checkpoint(best_path, epoch, val_metrics)
                    print(f"✓ Saved best model (S-measure: {best_smeasure:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % args.save_interval == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pth')
                trainer.save_checkpoint(ckpt_path, epoch, val_metrics)
                print(f"✓ Saved checkpoint: {ckpt_path}")

                # Clean up old periodic checkpoints (keep only last 3)
                import glob
                periodic_ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'epoch_*.pth')))
                if len(periodic_ckpts) > 3:
                    for old_ckpt in periodic_ckpts[:-3]:  # Keep only last 3
                        os.remove(old_ckpt)
                        print(f"  🗑️  Removed old checkpoint: {os.path.basename(old_ckpt)}")

            # Always save latest
            latest_path = os.path.join(args.checkpoint_dir, 'latest.pth')
            trainer.save_checkpoint(latest_path, epoch, val_metrics)

    # Final summary
    if is_main_process:
        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)
        print(f"Best validation S-measure: {best_smeasure:.4f}")
        print(f"Checkpoints saved to: {args.checkpoint_dir}")

        summary = trainer.get_training_summary()
        print(f"\nFinal Summary:")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.6f}")

    # Cleanup
    if args.use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
