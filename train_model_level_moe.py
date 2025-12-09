"""
3-Stage Training for Model-Level MoE Ensemble

Target: 0.80-0.81 IoU (beat SOTA at 0.78-0.79)

Training Strategy:
  STAGE 1 (40 epochs): Train each expert independently
    - Expert 1 (SINet): 40 epochs
    - Expert 2 (PraNet): 40 epochs
    - Expert 3 (ZoomNet): 40 epochs
    - Expert 4 (UJSC): 40 epochs
    - Goal: Each expert reaches 0.73-0.76 IoU individually

  STAGE 2 (30 epochs): Train router with frozen experts
    - Freeze all experts
    - Train router to select best expert per image
    - Goal: Learn optimal routing strategy

  STAGE 3 (80 epochs): Fine-tune everything together
    - Unfreeze all parameters
    - Low learning rate
    - Goal: Ensemble reaches 0.80-0.81 IoU
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.model_level_moe import ModelLevelMoE, count_parameters
try:
    from dataset_updated import COD10KDataset
except ImportError:
    # Fallback if in a subdirectory or running differently
    try:
        from data.dataset_updated import COD10KDataset
    except ImportError:
        from data.dataset import COD10KDataset
        print("WARNING: Using original dataset.py (dataset_updated not found)")
from losses.advanced_loss import CODSpecializedLoss
from metrics.cod_metrics import CODMetrics
from models.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Model-Level MoE 3-Stage Training')

    # Data
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=8)  # Reduced from 12 for 512px
    parser.add_argument('--img-size', type=int, default=512)  # Increased from 448 for SOTA
    parser.add_argument('--num-workers', type=int, default=4)

    # Model
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--num-experts', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=2)

    # Training stages
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='Training stage: 1=Experts, 2=Router, 3=Full')
    parser.add_argument('--expert-id', type=int, default=None,
                        help='For stage 1: which expert to train (0-3, or None for all)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override default epochs for stage')

    # Optimization
    parser.add_argument('--lr', type=float, default=None,
                        help='Override default LR for stage')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps (effective_batch = batch_size * accumulation_steps)')

    # LR Scheduler Configuration
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'warmrestarts', 'plateau', 'step', 'none'],
                        help='LR scheduler type: cosine (smooth decay), warmrestarts (periodic), plateau (adaptive), step (milestones), none (constant)')
    parser.add_argument('--scheduler-t0', type=int, default=20,
                        help='For warmrestarts: initial restart period (epochs)')
    parser.add_argument('--scheduler-tmult', type=int, default=1,
                        help='For warmrestarts: cycle length multiplier (1=same length, 2=doubling)')
    parser.add_argument('--scheduler-patience', type=int, default=5,
                        help='For plateau: epochs to wait before reducing LR')
    parser.add_argument('--scheduler-factor', type=float, default=0.5,
                        help='For plateau/step: LR reduction factor')
    parser.add_argument('--scheduler-min-lr', type=float, default=0.01,
                        help='Minimum LR as ratio of initial LR (e.g., 0.01 = 1%%)')
    parser.add_argument('--scheduler-milestones', type=int, nargs='+', default=[30, 45],
                        help='For step: epochs to reduce LR')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_moe')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--load-experts-from', type=str, default=None,
                        help='For stage 2/3: path to trained experts')

    # Router warmup (for Stage 3)
    parser.add_argument('--enable-router-warmup', action='store_true', default=True,
                        help='Enable router warmup in Stage 3 (freeze router for initial epochs)')
    parser.add_argument('--router-warmup-epochs', type=int, default=15,
                        help='Number of epochs to keep router frozen in Stage 3 (default: 15)')

    # System
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-ddp', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    # torchrun sets LOCAL_RANK as environment variable
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    return args


def setup_ddp(args):
    """Initialize DDP"""
    if args.use_ddp:
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()

        # Set device based on local_rank
        print(f"[Rank {args.rank}] local_rank from args: {args.local_rank}")
        torch.cuda.set_device(args.local_rank)
        print(f"[Rank {args.rank}] torch.cuda.current_device(): {torch.cuda.current_device()}")
    else:
        args.world_size = 1
        args.rank = 0


def is_main_process(args):
    return args.rank == 0


def get_model(model):
    """Get the actual model from DDP wrapper if needed"""
    return model.module if hasattr(model, 'module') else model


def sync_metrics_across_ranks(metrics_dict, args):
    """
    Synchronize metrics across all DDP ranks by averaging

    Args:
        metrics_dict: Dictionary of metrics (each value is a scalar)
        args: Arguments containing DDP info

    Returns:
        Synchronized metrics dictionary (averaged across all ranks)
    """
    if not args.use_ddp:
        return metrics_dict

    # Convert metrics to tensor for all_reduce
    metrics_tensor = torch.tensor(
        list(metrics_dict.values()),
        dtype=torch.float32,
        device=torch.cuda.current_device()
    )

    # Sum across all ranks
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

    # Average by dividing by world size
    metrics_tensor /= args.world_size

    # Convert back to dictionary
    synced_metrics = {
        key: metrics_tensor[i].item()
        for i, key in enumerate(metrics_dict.keys())
    }

    return synced_metrics


def create_scheduler(optimizer, args):
    """
    Create LR scheduler based on args

    Returns:
        scheduler: LR scheduler instance
        needs_metric: Whether scheduler.step() needs validation metric
    """
    min_lr = args.lr * args.scheduler_min_lr

    if args.scheduler == 'cosine':
        # Simple cosine decay - SMOOTH & STABLE (recommended for most cases)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=min_lr
        )
        return scheduler, False

    elif args.scheduler == 'warmrestarts':
        # Cosine with periodic restarts - can escape local minima but may be unstable
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.scheduler_t0, T_mult=args.scheduler_tmult, eta_min=min_lr
        )
        return scheduler, False

    elif args.scheduler == 'plateau':
        # Reduce LR when metric plateaus - adaptive but reactive
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.scheduler_factor,
            patience=args.scheduler_patience, verbose=True, min_lr=min_lr
        )
        return scheduler, True

    elif args.scheduler == 'step':
        # Step decay at milestones - simple but needs manual tuning
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_factor
        )
        return scheduler, False

    elif args.scheduler == 'none':
        # Constant LR - no scheduling
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        return scheduler, False

    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def print_stage_info(stage, args):
    """Print information about current training stage"""
    if not is_main_process(args):
        return

    print("\n" + "="*70)
    if stage == 1:
        if args.expert_id is not None:
            print(f"STAGE 1: TRAINING EXPERT {args.expert_id}")
            expert_names = ["SINet-Style", "PraNet-Style", "ZoomNet-Style", "UJSC-Style"]
            print(f"  Expert: {expert_names[args.expert_id]}")
        else:
            print("STAGE 1: TRAINING ALL EXPERTS SEQUENTIALLY")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        print(f"  Scheduler: {args.scheduler}")
        print(f"  Goal: Each expert reaches 0.73-0.76 IoU")
    elif stage == 2:
        print("STAGE 2: TRAINING ROUTER")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        print(f"  Scheduler: {args.scheduler}")
        print(f"  Experts: FROZEN")
        print(f"  Goal: Learn optimal expert selection")
    elif stage == 3:
        print("STAGE 3: FINE-TUNING FULL ENSEMBLE")
        print(f"  Epochs: {args.epochs}")
        print(f"  LR: {args.lr}")
        print(f"  Scheduler: {args.scheduler}")
        print(f"  All parameters: TRAINABLE")
        print(f"  Goal: Ensemble reaches 0.80-0.81 IoU")
    print("="*70 + "\n")


def train_expert(expert_id, model, train_loader, val_loader, criterion, metrics, args):
    """Train a single expert (Stage 1)"""

    actual_model = get_model(model)

    # Freeze everything except the target expert
    for name, param in model.named_parameters():
        if f'expert_models.{expert_id}' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Also need backbone trainable
    for param in actual_model.backbone.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(args):
        print(f"Trainable parameters: {trainable/1e6:.1f}M")

    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)

    # LR Scheduler (configurable via --scheduler flag)
    scheduler, scheduler_needs_metric = create_scheduler(optimizer, args)

    if is_main_process(args):
        sched_info = {
            'cosine': f"Cosine decay: {args.lr:.6f} → {args.lr * args.scheduler_min_lr:.6f} over {args.epochs} epochs",
            'warmrestarts': f"Warm restarts: T_0={args.scheduler_t0}, T_mult={args.scheduler_tmult}, min_lr={args.lr * args.scheduler_min_lr:.6f}",
            'plateau': f"Reduce on plateau: patience={args.scheduler_patience}, factor={args.scheduler_factor}, min_lr={args.lr * args.scheduler_min_lr:.6f}",
            'step': f"Step decay: milestones={args.scheduler_milestones}, factor={args.scheduler_factor}",
            'none': f"Constant LR: {args.lr:.6f}"
        }
        print(f"Scheduler: {sched_info.get(args.scheduler, args.scheduler)}")

    # AMP: Mixed precision training for 2-3x speedup
    scaler = GradScaler()

    # Training loop
    best_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # Wrap dataloader with tqdm progress bar (only on main process)
        if is_main_process(args):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Expert {expert_id}]")
        else:
            pbar = train_loader

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.cuda()
            masks = masks.cuda()

            # Forward with AMP (mixed precision) + Deep Supervision
            with autocast():
                features = actual_model.backbone(images)
                pred, aux_outputs = actual_model.expert_models[expert_id](features)

                # Main loss
                main_loss, _ = criterion(pred, masks)

                # Deep supervision: auxiliary losses at multiple scales
                aux_loss = 0
                for aux_pred in aux_outputs:
                    aux_l, _ = criterion(aux_pred, masks)
                    aux_loss += aux_l
                aux_loss = aux_loss / len(aux_outputs) if aux_outputs else 0

                # Total loss: main + 0.4 * auxiliary
                loss = main_loss + 0.4 * aux_loss
                loss = loss / args.accumulation_steps

            # Backward (accumulate gradients) with scaled gradients
            scaler.scale(loss).backward()

            # Only step optimizer every accumulation_steps
            if (batch_idx + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accumulation_steps  # Unscale for logging
            num_batches += 1

            # Update progress bar with current loss
            if is_main_process(args):
                pbar.set_postfix({'loss': f'{loss.item() * args.accumulation_steps:.4f}'})

        # Handle remaining gradients if last batch wasn't full accumulation step
        if (batch_idx + 1) % args.accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = total_loss / num_batches

        # Clear CUDA cache to prevent OOM
        torch.cuda.empty_cache()

        # Validation with AMP
        model.eval()
        val_metrics = {}
        all_preds = []  # Collect predictions for debugging
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validating", leave=False) if is_main_process(args) else val_loader
            for images, masks in val_pbar:
                images = images.cuda()
                masks = masks.cuda()

                with autocast():
                    features = actual_model.backbone(images)
                    pred, _ = actual_model.expert_models[expert_id](features)  # Ignore aux outputs in validation
                    pred = torch.sigmoid(pred)

                if is_main_process(args):
                    all_preds.append(pred.detach())

                metrics.update(pred, masks)

        val_metrics = metrics.compute()
        metrics.reset()

        # CRITICAL: Synchronize metrics across all DDP ranks
        # Without this, we only see metrics from rank 0's validation subset!
        val_metrics = sync_metrics_across_ranks(val_metrics, args)

        # Debug: Show prediction statistics
        if is_main_process(args) and len(all_preds) > 0:
            all_preds_tensor = torch.cat(all_preds, dim=0)
            pred_min = all_preds_tensor.min().item()
            pred_max = all_preds_tensor.max().item()
            pred_mean = all_preds_tensor.mean().item()
            pred_std = all_preds_tensor.std().item()
            print(f"  Pred stats: min={pred_min:.4f}, max={pred_max:.4f}, mean={pred_mean:.4f}, std={pred_std:.4f}")

        if is_main_process(args):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f} | LR: {current_lr:.6f}")

            if val_metrics['IoU'] > best_iou:
                best_iou = val_metrics['IoU']
                print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

                # Save expert checkpoint
                checkpoint_path = os.path.join(args.checkpoint_dir, f'expert_{expert_id}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'expert_id': expert_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou
                }, checkpoint_path)

        # Step LR scheduler
        if scheduler_needs_metric:
            # ReduceLROnPlateau needs the validation metric
            scheduler.step(val_metrics['IoU'])
        else:
            # Other schedulers step based on epoch
            scheduler.step()

        # Final cache clear before next epoch
        torch.cuda.empty_cache()

    if is_main_process(args):
        print(f"\nExpert {expert_id} training complete. Best IoU: {best_iou:.4f}")

    return best_iou


def train_router(model, train_loader, val_loader, criterion, metrics, args):
    """Train router with frozen experts (Stage 2)"""

    # Freeze experts and backbone, train only router
    for name, param in model.named_parameters():
        if 'router' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(args):
        print(f"Trainable parameters (router only): {trainable/1e6:.1f}M")

    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)

    # LR Scheduler
    scheduler, scheduler_needs_metric = create_scheduler(optimizer, args)

    # Training loop
    best_iou = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Router]") if is_main_process(args) else train_loader

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.cuda()
            masks = masks.cuda()

            # Forward: router selects experts
            pred, routing_info = model(images, return_routing_info=True)

            # Segmentation loss
            seg_loss, _ = criterion(pred, masks)

            # CRITICAL: Router diversity loss (encourages expert specialization)
            # We want experts to be used roughly equally (prevents collapse to one expert)
            expert_probs = routing_info['expert_probs']  # [B, num_experts]
            avg_expert_usage = expert_probs.mean(dim=0)  # [num_experts]
            uniform_usage = torch.ones_like(avg_expert_usage) / args.num_experts
            diversity_loss = F.kl_div(
                avg_expert_usage.log(),
                uniform_usage,
                reduction='batchmean'
            )

            # Total loss: segmentation + diversity (increased from 0.01 to 0.1 for stronger diversity)
            loss = (seg_loss + 0.1 * diversity_loss) / args.accumulation_steps

            # Backward (accumulate gradients)
            loss.backward()

            # Step optimizer every accumulation_steps
            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accumulation_steps
            num_batches += 1

            if is_main_process(args):
                pbar.set_postfix({'seg_loss': f'{seg_loss.item():.4f}', 'div_loss': f'{diversity_loss.item():.4f}'})

        avg_loss = total_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validating", leave=False) if is_main_process(args) else val_loader
            for images, masks in val_pbar:
                images = images.cuda()
                masks = masks.cuda()

                pred, routing_info = model(images, return_routing_info=True)
                pred = torch.sigmoid(pred)

                metrics.update(pred, masks)

        val_metrics = metrics.compute()
        routing_stats = routing_info['routing_stats'] if 'routing_stats' in routing_info else {}
        metrics.reset()

        # Synchronize metrics across all DDP ranks
        val_metrics = sync_metrics_across_ranks(val_metrics, args)

        if is_main_process(args):
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"IoU: {val_metrics['IoU']:.4f} | "
                  f"Entropy: {routing_stats.get('entropy', 0):.3f}")

            # Print router health diagnostics
            if routing_stats:
                entropy = routing_stats.get('entropy', 0)
                avg_probs = routing_stats.get('avg_expert_probs', [])

                print(f"\n  Router Health:")
                print(f"    Entropy: {entropy:.3f} (healthy: 1.0-1.6)")
                if avg_probs:
                    print(f"    Expert probs: {[f'{p:.2f}' for p in avg_probs]}")

                if entropy < 0.5:
                    print(f"    ⚠️  COLLAPSE DETECTED - Router using same experts for all images!")
                elif entropy > 1.8:
                    print(f"    ⚠️  NO LEARNING - Router selecting randomly!")
                else:
                    print(f"    ✓ Router is learning meaningful routing patterns")

            if val_metrics['IoU'] > best_iou:
                best_iou = val_metrics['IoU']
                print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

                # Save checkpoint
                checkpoint_path = os.path.join(args.checkpoint_dir, 'router_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou
                }, checkpoint_path)

        # Step LR scheduler
        if scheduler_needs_metric:
            scheduler.step(val_metrics['IoU'])
        else:
            scheduler.step()

    if is_main_process(args):
        print(f"\nRouter training complete. Best IoU: {best_iou:.4f}")

    return best_iou


def set_router_trainable(model, trainable):
    """
    Freeze or unfreeze router parameters.

    Args:
        model: The model (potentially wrapped in DDP)
        trainable: Boolean - True to unfreeze, False to freeze
    """
    actual_model = get_model(model)

    for name, param in actual_model.named_parameters():
        if 'router' in name:
            param.requires_grad = trainable


def train_full_ensemble(model, train_loader, val_loader, criterion, metrics, args):
    """Fine-tune full ensemble (Stage 3)"""

    actual_model = get_model(model)

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(args):
        print(f"Trainable parameters (full model): {trainable/1e6:.1f}M")

    # Optimizer with different LRs for different components
    optimizer = AdamW([
        {'params': actual_model.backbone.parameters(), 'lr': args.lr * 0.1},  # Lower LR for backbone
        {'params': actual_model.router.parameters(), 'lr': args.lr},
        {'params': actual_model.expert_models.parameters(), 'lr': args.lr * 0.5}  # Medium LR for experts
    ], weight_decay=args.weight_decay)

    # LR Scheduler
    scheduler, scheduler_needs_metric = create_scheduler(optimizer, args)

    # Training loop
    best_iou = 0.0
    for epoch in range(args.epochs):
        # Router warmup: freeze router for initial epochs to stabilize expert learning
        if args.enable_router_warmup:
            if epoch < args.router_warmup_epochs:
                set_router_trainable(model, trainable=False)
                if epoch == 0 and is_main_process(args):
                    print(f"🔒 Router FROZEN for warmup (epochs 0-{args.router_warmup_epochs-1})")
                    print("   Experts will stabilize before routing patterns are learned\n")
            elif epoch == args.router_warmup_epochs:
                set_router_trainable(model, trainable=True)
                if is_main_process(args):
                    print(f"🔓 Router UNFROZEN (epoch {args.router_warmup_epochs}) - now learning routing patterns\n")

        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Full Ensemble]") if is_main_process(args) else train_loader

        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.cuda()
            masks = masks.cuda()

            # Forward
            pred = model(images)

            # Loss (scale by accumulation steps)
            loss, _ = criterion(pred, masks)
            loss = loss / args.accumulation_steps

            # Backward (accumulate gradients)
            loss.backward()

            # Step optimizer every accumulation_steps
            if (batch_idx + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.accumulation_steps
            num_batches += 1

            if is_main_process(args):
                pbar.set_postfix({'loss': f'{loss.item() * args.accumulation_steps:.4f}'})

        avg_loss = total_loss / num_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validating", leave=False) if is_main_process(args) else val_loader
            for images, masks in val_pbar:
                images = images.cuda()
                masks = masks.cuda()

                pred, routing_info = model(images, return_routing_info=True)
                pred = torch.sigmoid(pred)

                metrics.update(pred, masks)

        val_metrics = metrics.compute()
        routing_stats = routing_info['routing_stats']
        metrics.reset()

        # Synchronize metrics across all DDP ranks
        val_metrics = sync_metrics_across_ranks(val_metrics, args)

        current_lr = optimizer.param_groups[0]['lr']

        if is_main_process(args):
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"IoU: {val_metrics['IoU']:.4f} | Dice: {val_metrics['Dice_Score']:.4f} | "
                  f"LR: {current_lr:.6f}")

            # Print router health diagnostics
            if routing_stats:
                entropy = routing_stats.get('entropy', 0)
                avg_probs = routing_stats.get('avg_expert_probs', [])

                print(f"\n  Router Health:")
                print(f"    Entropy: {entropy:.3f} (healthy: 1.0-1.6)")
                if avg_probs:
                    print(f"    Expert probs: {[f'{p:.2f}' for p in avg_probs]}")

                if entropy < 0.5:
                    print(f"    ⚠️  COLLAPSE DETECTED - Router using same experts for all images!")
                elif entropy > 1.8:
                    print(f"    ⚠️  NO LEARNING - Router selecting randomly!")
                else:
                    print(f"    ✓ Router is learning meaningful routing patterns")

            if val_metrics['IoU'] > best_iou:
                best_iou = val_metrics['IoU']
                print(f"🏆 NEW BEST! IoU: {best_iou:.4f}")

                # Save checkpoint
                checkpoint_path = os.path.join(args.checkpoint_dir, 'ensemble_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou
                }, checkpoint_path)

        # Step LR scheduler
        if scheduler_needs_metric:
            scheduler.step(val_metrics['IoU'])
        else:
            scheduler.step()

    if is_main_process(args):
        print(f"\nEnsemble training complete. Best IoU: {best_iou:.4f}")

    return best_iou


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_ddp(args)

    # Set default hyperparameters per stage
    if args.stage == 1:
        args.epochs = args.epochs or 40
        args.lr = args.lr or 0.0003
    elif args.stage == 2:
        args.epochs = args.epochs or 30
        args.lr = args.lr or 0.0002
    elif args.stage == 3:
        args.epochs = args.epochs or 80
        args.lr = args.lr or 0.00005  # Much lower for fine-tuning

    print_stage_info(args.stage, args)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data
    # DDP-aware caching: Each process only caches images it will actually use
    # Rank 0: caches indices [0, 2, 4, ...] → 3000 train + 400 val
    # Rank 1: caches indices [1, 3, 5, ...] → 3000 train + 400 val
    # Total: 6800 images (same as single GPU!) instead of 13,600

    train_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='train',
        img_size=args.img_size,
        cache_in_memory=True,
        rank=args.rank if args.use_ddp else 0,
        world_size=args.world_size if args.use_ddp else 1
    )
    val_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='val',
        img_size=args.img_size,
        cache_in_memory=True,
        rank=args.rank if args.use_ddp else 0,
        world_size=args.world_size if args.use_ddp else 1
    )

    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
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

    # Model
    target_device = args.local_rank if args.use_ddp else 0
    if is_main_process(args) or args.use_ddp:
        print(f"[Rank {args.rank}] Creating model on cuda:{target_device}")

    model = ModelLevelMoE(
        backbone=args.backbone,
        num_experts=args.num_experts,
        top_k=args.top_k,
        pretrained=True
    ).cuda(target_device)

    if is_main_process(args) or args.use_ddp:
        print(f"[Rank {args.rank}] Model created, checking device...")
        # Verify model is on correct device
        first_param_device = next(model.parameters()).device
        print(f"[Rank {args.rank}] Model parameters are on: {first_param_device}")

    # Load checkpoints if specified
    device = f'cuda:{args.local_rank if args.use_ddp else 0}'
    if args.load_experts_from and args.stage >= 2:
        if is_main_process(args):
            print(f"\nLoading trained experts from: {args.load_experts_from}")
        checkpoint = torch.load(args.load_experts_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.resume_from:
        if is_main_process(args):
            print(f"\nResuming from: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # DDP
    if args.use_ddp:
        print(f"[Rank {args.rank}] Wrapping model with DDP on device_ids=[{args.local_rank}]")
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
        print(f"[Rank {args.rank}] DDP wrapping successful")

    # Loss and metrics
    criterion = CODSpecializedLoss(
        bce_weight=2.0,
        iou_weight=1.5,
        edge_weight=1.0,
        boundary_weight=1.5,
        uncertainty_weight=0.3,
        reverse_attention_weight=0.8,
        aux_weight=0.0
    ).cuda()

    metrics = CODMetrics()

    # Train based on stage
    if args.stage == 1:
        if args.expert_id is not None:
            train_expert(args.expert_id, model, train_loader, val_loader, criterion, metrics, args)
        else:
            # Train all experts sequentially
            for expert_id in range(args.num_experts):
                if is_main_process(args):
                    print(f"\n{'='*70}")
                    print(f"Training Expert {expert_id}")
                    print(f"{'='*70}\n")
                train_expert(expert_id, model, train_loader, val_loader, criterion, metrics, args)

    elif args.stage == 2:
        train_router(model, train_loader, val_loader, criterion, metrics, args)

    elif args.stage == 3:
        train_full_ensemble(model, train_loader, val_loader, criterion, metrics, args)

    if args.use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
