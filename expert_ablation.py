"""
Expert Ablation Study for CamoXpert MoE

This script evaluates:
1. Each expert INDIVIDUALLY (SINet, PraNet, FSPNet)
2. MoE ensemble performance
3. Oracle (best expert per image) - upper bound
4. Simple averaging baseline
5. 2-expert combinations
6. Top-K routing analysis

This proves the value of MoE by showing:
- Which expert is best overall
- If MoE beats individual experts
- The "oracle" gap (how well routing COULD work)
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import pandas as pd
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_level_moe import ModelLevelMoE
from models.multi_scale_processor import MultiScaleInputProcessor
from metrics.cod_metrics import CODMetrics


def load_model(checkpoint_path: str, backbone: str, expert_types: List[str], device: str):
    """Load trained MoE model."""
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Epoch: {epoch}")
    else:
        state_dict = checkpoint
    
    # Handle DDP prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Check for multi-scale
    has_multi_scale = any('backbone.backbone.' in k for k in new_state_dict.keys())
    
    model = ModelLevelMoE(
        backbone_name=backbone,
        num_experts=len(expert_types),
        top_k=2,
        pretrained=False,
        expert_types=expert_types
    )
    
    if has_multi_scale:
        print("✓ Wrapping with MultiScaleInputProcessor")
        channels_list = [64, 128, 320, 512]
        model.backbone = MultiScaleInputProcessor(
            backbone=model.backbone,
            channels_list=channels_list,
            scales=[0.5, 1.0, 1.5],
            use_hierarchical=True
        )
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model


def load_image(image_path: str, img_size: int = 384):
    """Load and preprocess image."""
    image = cv2.imread(image_path)
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = (image.shape[1], image.shape[0])
    
    resized = cv2.resize(original, (img_size, img_size))
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (normalized - mean) / std
    
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, original_size


def load_gt(gt_path: str) -> np.ndarray:
    """Load ground truth mask."""
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    return (gt > 128).astype(np.float32)


@torch.no_grad()
def get_all_expert_predictions(model, image_tensor: torch.Tensor, device: str):
    """
    Get predictions from EACH expert individually + MoE combined.
    
    Returns:
        moe_pred: Combined MoE prediction
        expert_preds: List of individual expert predictions [expert0, expert1, expert2]
        routing_probs: Router's probability distribution [B, num_experts]
    """
    image_tensor = image_tensor.to(device)
    
    # Get backbone features
    features = model.backbone(image_tensor)
    
    # Get router probabilities
    expert_probs, top_k_indices, top_k_weights, _ = model.router(features)
    
    # Get each expert's prediction
    expert_preds = []
    for expert in model.expert_models:
        pred, _ = expert(features, return_aux=False)
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        expert_preds.append(pred.cpu().numpy())
    
    # Get MoE combined prediction
    expert_predictions_stacked = torch.stack([
        torch.from_numpy(p).to(device) for p in expert_preds
    ], dim=1)
    
    moe_pred = torch.sum(
        expert_predictions_stacked * expert_probs.view(-1, len(expert_preds), 1, 1, 1),
        dim=1
    ).cpu().numpy()
    
    return moe_pred, expert_preds, expert_probs.cpu().numpy(), top_k_indices.cpu().numpy()


def compute_iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Compute IoU between prediction and ground truth."""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > 0.5).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    if union == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0
    return intersection / union


def run_ablation(model, img_dir: Path, gt_dir: Path, img_size: int, 
                 device: str, expert_names: List[str]):
    """
    Run comprehensive ablation study.
    """
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    print(f"Found {len(image_files)} test images")
    
    results = []
    num_experts = len(expert_names)
    
    # Metrics calculators for each configuration
    metrics_calculators = {
        'moe': CODMetrics(),
        'simple_avg': CODMetrics(),  # NEW: Simple averaging
        'oracle': CODMetrics(),
        **{f'expert_{i}': CODMetrics() for i in range(num_experts)},
    }
    
    # 2-expert combinations
    for combo in combinations(range(num_experts), 2):
        name = f"pair_{combo[0]}_{combo[1]}"
        metrics_calculators[name] = CODMetrics()
    
    for img_path in tqdm(image_files, desc="Evaluating"):
        # Load image and GT
        tensor, orig_size = load_image(str(img_path), img_size)
        
        gt_path = gt_dir / (img_path.stem + '.png')
        if not gt_path.exists():
            gt_path = gt_dir / (img_path.stem + '.jpg')
        if not gt_path.exists():
            continue
            
        gt = load_gt(str(gt_path))
        
        # Get all predictions
        moe_pred, expert_preds, routing_probs, top_k_indices = get_all_expert_predictions(
            model, tensor, device
        )
        
        # Resize predictions to original size
        moe_pred_resized = cv2.resize(moe_pred.squeeze(), orig_size)
        expert_preds_resized = [
            cv2.resize(p.squeeze(), orig_size) for p in expert_preds
        ]
        
        # Simple averaging baseline
        simple_avg = np.mean(expert_preds_resized, axis=0)
        
        # Compute IoU for each
        moe_iou = compute_iou(moe_pred_resized, gt)
        expert_ious = [compute_iou(p, gt) for p in expert_preds_resized]
        simple_avg_iou = compute_iou(simple_avg, gt)
        
        # Oracle: best expert for this image
        best_expert_idx = np.argmax(expert_ious)
        oracle_pred = expert_preds_resized[best_expert_idx]
        oracle_iou = expert_ious[best_expert_idx]
        
        # Router's choice
        router_choice = np.argmax(routing_probs[0])
        router_chosen_iou = expert_ious[router_choice]
        
        # Top-K analysis
        top_k_experts = top_k_indices[0]
        
        # Store per-image result
        result = {
            'image': img_path.stem,
            'moe_iou': moe_iou,
            'simple_avg_iou': simple_avg_iou,
            'oracle_iou': oracle_iou,
            'best_expert': best_expert_idx,
            'router_choice': router_choice,
            'router_choice_iou': router_chosen_iou,
            'routing_correct': int(router_choice == best_expert_idx),
            'top_k_experts': str(top_k_experts),
            'best_in_top_k': int(best_expert_idx in top_k_experts),
        }
        for i, iou in enumerate(expert_ious):
            result[f'expert_{i}_iou'] = iou
        for i, prob in enumerate(routing_probs[0]):
            result[f'routing_prob_{i}'] = prob
        
        results.append(result)
        
        # Update metrics
        gt_tensor = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)
        
        moe_tensor = torch.from_numpy(moe_pred_resized).float().unsqueeze(0).unsqueeze(0)
        metrics_calculators['moe'].update(moe_tensor, gt_tensor)
        
        avg_tensor = torch.from_numpy(simple_avg).float().unsqueeze(0).unsqueeze(0)
        metrics_calculators['simple_avg'].update(avg_tensor, gt_tensor)
        
        oracle_tensor = torch.from_numpy(oracle_pred).float().unsqueeze(0).unsqueeze(0)
        metrics_calculators['oracle'].update(oracle_tensor, gt_tensor)
        
        for i, pred in enumerate(expert_preds_resized):
            pred_tensor = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
            metrics_calculators[f'expert_{i}'].update(pred_tensor, gt_tensor)
        
        # 2-expert combinations (average of pairs)
        for combo in combinations(range(num_experts), 2):
            pair_avg = (expert_preds_resized[combo[0]] + expert_preds_resized[combo[1]]) / 2
            pair_tensor = torch.from_numpy(pair_avg).float().unsqueeze(0).unsqueeze(0)
            metrics_calculators[f"pair_{combo[0]}_{combo[1]}"].update(pair_tensor, gt_tensor)
    
    return results, metrics_calculators


def print_results(results: List[Dict], metrics: Dict, expert_names: List[str], output_dir: Path):
    """Print and save comprehensive ablation results."""
    df = pd.DataFrame(results)
    num_experts = len(expert_names)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERT ABLATION STUDY")
    print("="*80)
    
    # Table 1: Individual Experts vs Ensembles
    print("\n📊 TABLE 1: Method Comparison")
    print("-"*70)
    print(f"{'Method':<25} {'S-measure':<12} {'IoU':<12} {'F-measure':<12} {'MAE':<10}")
    print("-"*70)
    
    # Print each expert
    for i, name in enumerate(expert_names):
        m = metrics[f'expert_{i}'].compute()
        print(f"{name:<25} {m['S-measure']:<12.4f} {m['IoU']:<12.4f} {m['F-measure']:<12.4f} {m['MAE']:<10.4f}")
    
    print("-"*70)
    
    # Print ensembles
    m = metrics['simple_avg'].compute()
    print(f"{'Simple Average':<25} {m['S-measure']:<12.4f} {m['IoU']:<12.4f} {m['F-measure']:<12.4f} {m['MAE']:<10.4f}")
    
    m = metrics['moe'].compute()
    print(f"{'MoE (Ours)':<25} {m['S-measure']:<12.4f} {m['IoU']:<12.4f} {m['F-measure']:<12.4f} {m['MAE']:<10.4f}")
    
    m = metrics['oracle'].compute()
    print(f"{'Oracle (Upper Bound)':<25} {m['S-measure']:<12.4f} {m['IoU']:<12.4f} {m['F-measure']:<12.4f} {m['MAE']:<10.4f}")
    
    print("-"*70)
    
    # Table 2: 2-Expert Combinations
    print("\n📊 TABLE 2: Expert Pair Combinations (Average)")
    print("-"*70)
    print(f"{'Pair':<25} {'S-measure':<12} {'IoU':<12} {'F-measure':<12}")
    print("-"*70)
    
    for combo in combinations(range(num_experts), 2):
        name = f"{expert_names[combo[0]]} + {expert_names[combo[1]]}"
        m = metrics[f"pair_{combo[0]}_{combo[1]}"].compute()
        print(f"{name:<25} {m['S-measure']:<12.4f} {m['IoU']:<12.4f} {m['F-measure']:<12.4f}")
    
    print("-"*70)
    
    # Table 3: Router Analysis
    print("\n📊 TABLE 3: Router Behavior Analysis")
    print("-"*70)
    
    routing_accuracy = df['routing_correct'].mean() * 100
    top_k_coverage = df['best_in_top_k'].mean() * 100
    
    print(f"Router Accuracy (picks oracle expert):     {routing_accuracy:.1f}%")
    print(f"Top-K Coverage (oracle in top-2):          {top_k_coverage:.1f}%")
    
    print("\nRouter Selection Distribution:")
    for i, name in enumerate(expert_names):
        selected = (df['router_choice'] == i).sum()
        pct = selected / len(df) * 100
        print(f"  {name}: {selected} images ({pct:.1f}%)")
    
    print("\nOracle Best Expert Distribution:")
    for i, name in enumerate(expert_names):
        best = (df['best_expert'] == i).sum()
        pct = best / len(df) * 100
        print(f"  {name}: {best} images ({pct:.1f}%)")
    
    print("-"*70)
    
    # Key Findings
    moe_metrics = metrics['moe'].compute()
    simple_avg_metrics = metrics['simple_avg'].compute()
    oracle_metrics = metrics['oracle'].compute()
    
    expert_sm = [metrics[f'expert_{i}'].compute()['S-measure'] for i in range(num_experts)]
    best_single_idx = np.argmax(expert_sm)
    best_single_name = expert_names[best_single_idx]
    best_single_sm = expert_sm[best_single_idx]
    
    print("\n🔍 KEY FINDINGS:")
    print("-"*70)
    print(f"1. Best Single Expert: {best_single_name} (S-measure: {best_single_sm:.4f})")
    print(f"2. MoE Improvement over Simple Avg: {(moe_metrics['S-measure'] - simple_avg_metrics['S-measure'])*100:+.2f}%")
    print(f"3. MoE vs Best Single Expert: {(moe_metrics['S-measure'] - best_single_sm)*100:+.2f}%")
    print(f"4. Gap to Oracle: {(oracle_metrics['S-measure'] - moe_metrics['S-measure'])*100:.2f}%")
    print("-"*70)
    
    # Conclusion
    print("\n" + "="*80)
    if moe_metrics['S-measure'] > best_single_sm:
        improvement = (moe_metrics['S-measure'] - best_single_sm) * 100
        print(f"✅ SUCCESS: MoE outperforms best single expert by {improvement:.2f}%!")
    elif moe_metrics['S-measure'] > simple_avg_metrics['S-measure']:
        improvement = (moe_metrics['S-measure'] - simple_avg_metrics['S-measure']) * 100
        print(f"✓ MoE outperforms simple averaging by {improvement:.2f}%")
        print(f"  (but underperforms best single expert by {(best_single_sm - moe_metrics['S-measure'])*100:.2f}%)")
    else:
        print("⚠️ MoE underperforms - router may need more training")
    print("="*80)
    
    # Save CSV
    csv_path = output_dir / 'ablation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed results saved to: {csv_path}")
    
    # Save summary
    summary_path = output_dir / 'ablation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("ABLATION STUDY SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write("Individual Experts:\n")
        for i, name in enumerate(expert_names):
            m = metrics[f'expert_{i}'].compute()
            f.write(f"  {name}: S={m['S-measure']:.4f}, IoU={m['IoU']:.4f}\n")
        f.write(f"\nSimple Average: S={simple_avg_metrics['S-measure']:.4f}, IoU={simple_avg_metrics['IoU']:.4f}\n")
        f.write(f"MoE (Ours): S={moe_metrics['S-measure']:.4f}, IoU={moe_metrics['IoU']:.4f}\n")
        f.write(f"Oracle: S={oracle_metrics['S-measure']:.4f}, IoU={oracle_metrics['IoU']:.4f}\n")
        f.write(f"\nRouter Accuracy: {routing_accuracy:.1f}%\n")
        f.write(f"Top-K Coverage: {top_k_coverage:.1f}%\n")
    print(f"✓ Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Expert Ablation Study')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--gt-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./ablation_results')
    parser.add_argument('--img-size', type=int, default=448)
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--expert-types', nargs='+', default=['sinet', 'pranet', 'fspnet'])
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, args.backbone, args.expert_types, device)
    
    # Run ablation
    results, metrics = run_ablation(
        model,
        Path(args.image_dir),
        Path(args.gt_dir),
        args.img_size,
        device,
        args.expert_types
    )
    
    # Print results
    print_results(results, metrics, args.expert_types, output_dir)


if __name__ == '__main__':
    main()

