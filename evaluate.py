"""
Comprehensive Evaluation Script for CamoXpert Model

Supports multiple COD datasets and computes all standard metrics.

Usage:
    python evaluate.py \
        --checkpoint checkpoints/best_model.pth \
        --data-root /path/to/datasets \
        --datasets COD10K CAMO CHAMELEON NC4K \
        --output-dir ./results \
        --save-predictions \
        --batch-size 4

Dataset structure expected:
    {data-root}/{dataset}/Test/Image/
    {data-root}/{dataset}/Test/GT_Object/
"""

import argparse
import os
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# Import model and metrics
from models.model_level_moe import ModelLevelMoE
from metrics.cod_metrics import CODMetrics


# ============================================================================
# Dataset Classes
# ============================================================================

class EvalDataset(Dataset):
    """
    Generic evaluation dataset for COD benchmarks.
    
    Supports multiple directory structures commonly used in COD datasets.
    """
    
    def __init__(
        self,
        root_dir: str,
        img_size: int = 352,
        dataset_name: str = "unknown"
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.dataset_name = dataset_name
        
        # Try different directory structures (common COD dataset layouts)
        possible_structures = [
            # Standard COD10K structure
            {'img': 'Test/Image', 'gt': 'Test/GT_Object'},
            {'img': 'Test/Imgs', 'gt': 'Test/GT'},
            {'img': 'Test/image', 'gt': 'Test/mask'},
            # CAMO dataset structure (no Test subfolder)
            {'img': 'Images', 'gt': 'GT'},
            {'img': 'Image', 'gt': 'GT'},
            {'img': 'Imgs', 'gt': 'GT'},
            # Other common structures
            {'img': 'image', 'gt': 'mask'},
            {'img': 'Image', 'gt': 'GT_Object'},
            {'img': 'images', 'gt': 'masks'},
            {'img': 'img', 'gt': 'gt'},
        ]
        
        self.image_dir = None
        self.gt_dir = None
        
        for struct in possible_structures:
            img_path = self.root_dir / struct['img']
            gt_path = self.root_dir / struct['gt']
            if img_path.exists() and gt_path.exists():
                self.image_dir = img_path
                self.gt_dir = gt_path
                break
        
        if self.image_dir is None:
            raise FileNotFoundError(
                f"Could not find valid dataset structure in {root_dir}\n"
                f"Available directories: {list(self.root_dir.iterdir()) if self.root_dir.exists() else 'N/A'}"
            )
        
        # Get image list
        self.image_list = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        if len(self.image_list) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"  {dataset_name}: {len(self.image_list)} images from {self.image_dir}")
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Dict:
        img_name = self.image_list[idx]
        base_name = os.path.splitext(img_name)[0]
        
        # Load image
        img_path = self.image_dir / img_name
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = (image.shape[1], image.shape[0])  # (W, H)
        
        # Load ground truth (try different extensions)
        gt = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            gt_path = self.gt_dir / (base_name + ext)
            if gt_path.exists():
                gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                break
        
        if gt is None:
            raise IOError(f"Failed to load GT for: {base_name}")
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        gt_resized = cv2.resize(gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        
        # Normalize GT to [0, 1]
        gt_resized = (gt_resized > 128).astype(np.float32)
        
        return {
            'image': torch.from_numpy(image).float(),
            'gt': torch.from_numpy(gt_resized).unsqueeze(0).float(),
            'name': base_name,
            'original_size': original_size  # (W, H) tuple for resizing predictions back
        }


# ============================================================================
# Model Loading
# ============================================================================

def load_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda',
    backbone: str = 'pvt_v2_b2',
    num_experts: Optional[int] = None
) -> torch.nn.Module:
    """
    Load model from checkpoint, handling DDP 'module.' prefix.
    """
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Checkpoint from epoch: {epoch}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DDP 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Auto-detect num_experts from checkpoint
    # Look for the FINAL layer in decision_network (largest index with small output dim)
    if num_experts is None:
        # Find all decision_network weight layers and get the one with smallest output (that's num_experts)
        decision_layers = {}
        for key, value in new_state_dict.items():
            if 'router.decision_network' in key and 'weight' in key and value.dim() == 2:
                # Extract layer index (e.g., "router.decision_network.11.weight" -> 11)
                parts = key.split('.')
                for i, p in enumerate(parts):
                    if p == 'decision_network' and i+1 < len(parts):
                        try:
                            layer_idx = int(parts[i+1])
                            decision_layers[layer_idx] = value.shape[0]
                        except ValueError:
                            pass
        
        if decision_layers:
            # Find the layer with smallest output (that's the num_experts output layer)
            final_layer_idx = max(decision_layers.keys())
            num_experts = decision_layers[final_layer_idx]
            print(f"✓ Auto-detected {num_experts} experts from layer {final_layer_idx}")
        else:
            # Default fallback
            num_experts = 3
            print(f"⚠ Could not detect num_experts, using default: {num_experts}")
    
    # Create model with matching architecture
    model = ModelLevelMoE(
        backbone_name=backbone,
        num_experts=num_experts,
        top_k=2,
        pretrained=False
    )
    
    # Load state dict with strict=False to allow minor mismatches, then verify
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    # Check for critical mismatches
    critical_missing = [k for k in missing_keys if 'router' in k or 'expert' in k]
    if critical_missing:
        print(f"⚠ Missing critical keys: {critical_missing[:5]}...")
    
    if unexpected_keys:
        print(f"  Note: {len(unexpected_keys)} unexpected keys in checkpoint (will be ignored)")
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params:,} parameters")
    print(f"{'='*70}\n")
    
    return model


# ============================================================================
# Evaluation Functions
# ============================================================================

def compute_adaptive_f_measure(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """
    Compute adaptive F-measure and max F-measure.
    
    Returns:
        (adaptive_f, max_f)
    """
    beta2 = 0.3  # Standard beta^2 for F-measure
    
    # Adaptive threshold (2 * mean)
    adaptive_threshold = 2 * pred.mean()
    adaptive_threshold = min(adaptive_threshold, 1.0)
    
    pred_adaptive = (pred > adaptive_threshold).astype(np.float32)
    
    # Compute adaptive F
    tp = (pred_adaptive * gt).sum()
    fp = (pred_adaptive * (1 - gt)).sum()
    fn = ((1 - pred_adaptive) * gt).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    adaptive_f = ((1 + beta2) * precision * recall) / (beta2 * precision + recall + 1e-8)
    
    # Compute max F over thresholds
    max_f = 0.0
    for threshold in np.linspace(0.0, 1.0, 256):
        pred_bin = (pred > threshold).astype(np.float32)
        tp = (pred_bin * gt).sum()
        fp = (pred_bin * (1 - gt)).sum()
        fn = ((1 - pred_bin) * gt).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f = ((1 + beta2) * precision * recall) / (beta2 * precision + recall + 1e-8)
        max_f = max(max_f, f)
    
    return float(adaptive_f), float(max_f)


def evaluate_single_image(
    pred: torch.Tensor,
    gt: torch.Tensor,
    metrics_calculator: CODMetrics
) -> Dict[str, float]:
    """
    Evaluate a single image prediction.
    
    Args:
        pred: [1, 1, H, W] prediction (sigmoid applied)
        gt: [1, 1, H, W] ground truth
        metrics_calculator: CODMetrics instance
    
    Returns:
        Dictionary of metrics
    """
    # Use CODMetrics for standard metrics
    pred_np = pred.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()
    
    # Standard metrics from CODMetrics
    metrics = {
        'MAE': metrics_calculator.mae(pred, gt),
        'IoU': metrics_calculator.iou(pred, gt, threshold=0.5),
        'S-measure': metrics_calculator.s_measure(pred, gt),
        'E-measure': metrics_calculator.e_measure(pred, gt),
        'F-measure': metrics_calculator.f_measure(pred, gt, threshold=0.5),
        'Dice': metrics_calculator.dice_score(pred, gt, threshold=0.5),
    }
    
    # Adaptive and Max F-measure
    adaptive_f, max_f = compute_adaptive_f_measure(pred_np, gt_np)
    metrics['F-adaptive'] = adaptive_f
    metrics['F-max'] = max_f
    
    return metrics


def evaluate_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str,
    batch_size: int = 1,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Evaluate model on a dataset.
    
    Returns:
        (overall_metrics, per_image_results)
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    metrics_calculator = CODMetrics()
    per_image_results = []
    
    # Accumulators for averaging
    metric_sums = {}
    num_images = 0
    
    # Timing
    total_inference_time = 0.0
    
    # Create predictions directory
    if save_predictions and output_dir:
        pred_dir = output_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  Evaluating", leave=False):
            images = batch['image'].to(device)
            gts = batch['gt'].to(device)
            names = batch['name']
            original_sizes = batch['original_size']
            
            # Inference with timing
            start_time = time.time()
            output = model(images)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Extract predictions
            if isinstance(output, dict):
                preds = output.get('pred', output.get('predictions', None))
            elif isinstance(output, (tuple, list)):
                preds = output[0]
            else:
                preds = output
            
            preds = torch.sigmoid(preds)
            
            # Process each image in batch
            for i in range(images.size(0)):
                pred = preds[i:i+1]
                gt = gts[i:i+1]
                name = names[i]
                orig_w, orig_h = original_sizes[0][i].item(), original_sizes[1][i].item()
                
                # Compute metrics
                image_metrics = evaluate_single_image(pred, gt, metrics_calculator)
                image_metrics['name'] = name
                per_image_results.append(image_metrics)
                
                # Accumulate for averaging
                for key, value in image_metrics.items():
                    if key != 'name':
                        metric_sums[key] = metric_sums.get(key, 0.0) + value
                num_images += 1
                
                # Save prediction
                if save_predictions and output_dir:
                    # Resize prediction to original size
                    pred_resized = F.interpolate(
                        pred, size=(orig_h, orig_w),
                        mode='bilinear', align_corners=False
                    )
                    pred_np = (pred_resized.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    pred_img = Image.fromarray(pred_np, mode='L')
                    pred_img.save(pred_dir / f"{name}.png")
    
    # Compute averages
    overall_metrics = {key: value / num_images for key, value in metric_sums.items()}
    
    # Add timing info
    overall_metrics['total_images'] = num_images
    overall_metrics['total_time'] = total_inference_time
    overall_metrics['fps'] = num_images / total_inference_time if total_inference_time > 0 else 0
    overall_metrics['avg_time_per_image'] = total_inference_time / num_images if num_images > 0 else 0
    
    return overall_metrics, per_image_results


# ============================================================================
# Report Generation
# ============================================================================

def save_per_image_csv(results: List[Dict], output_path: Path):
    """Save per-image results to CSV."""
    if not results:
        return
    
    fieldnames = ['name', 'S-measure', 'IoU', 'F-measure', 'F-adaptive', 'F-max', 'MAE', 'E-measure', 'Dice']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"  ✓ Per-image results: {output_path}")


def save_summary_csv(all_results: Dict[str, Dict], output_path: Path):
    """Save summary results for all datasets to CSV."""
    fieldnames = [
        'Dataset', 'S-measure', 'IoU', 'F-measure', 'F-adaptive', 'F-max',
        'MAE', 'E-measure', 'Dice', 'FPS', 'Total_Images'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for dataset_name, metrics in all_results.items():
            row = {
                'Dataset': dataset_name,
                'S-measure': f"{metrics.get('S-measure', 0):.4f}",
                'IoU': f"{metrics.get('IoU', 0):.4f}",
                'F-measure': f"{metrics.get('F-measure', 0):.4f}",
                'F-adaptive': f"{metrics.get('F-adaptive', 0):.4f}",
                'F-max': f"{metrics.get('F-max', 0):.4f}",
                'MAE': f"{metrics.get('MAE', 0):.4f}",
                'E-measure': f"{metrics.get('E-measure', 0):.4f}",
                'Dice': f"{metrics.get('Dice', 0):.4f}",
                'FPS': f"{metrics.get('fps', 0):.1f}",
                'Total_Images': metrics.get('total_images', 0)
            }
            writer.writerow(row)
    
    print(f"\n✓ Summary saved: {output_path}")


def print_results_table(all_results: Dict[str, Dict]):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print(" " * 35 + "EVALUATION RESULTS")
    print("=" * 100)
    
    # Header
    header = f"{'Dataset':<15} {'S-measure':>10} {'IoU':>8} {'F-measure':>10} {'F-max':>8} {'MAE':>8} {'E-measure':>10} {'Dice':>8} {'FPS':>8}"
    print(header)
    print("-" * 100)
    
    # Results
    for dataset_name, metrics in all_results.items():
        if dataset_name == 'Average':
            print("-" * 100)
        row = f"{dataset_name:<15} {metrics.get('S-measure', 0):>10.4f} {metrics.get('IoU', 0):>8.4f} " \
              f"{metrics.get('F-measure', 0):>10.4f} {metrics.get('F-max', 0):>8.4f} {metrics.get('MAE', 0):>8.4f} " \
              f"{metrics.get('E-measure', 0):>10.4f} {metrics.get('Dice', 0):>8.4f} {metrics.get('fps', 0):>8.1f}"
        print(row)
    
    print("=" * 100)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CamoXpert Comprehensive Evaluation')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing dataset folders')
    
    # Dataset arguments
    parser.add_argument('--datasets', nargs='+', type=str,
                        default=['COD10K', 'CAMO', 'CHAMELEON', 'NC4K'],
                        help='List of datasets to evaluate (default: all)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Where to save predictions and results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Whether to save predicted masks as PNG')
    
    # Model arguments
    parser.add_argument('--img-size', type=int, default=352,
                        help='Input size (default: 352)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2',
                        choices=['pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5'],
                        help='Backbone architecture')
    parser.add_argument('--num-experts', type=int, default=None,
                        help='Number of experts (auto-detected if not specified)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(" " * 20 + "CamoXpert Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Datasets: {args.datasets}")
    print(f"Output dir: {args.output_dir}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Save predictions: {args.save_predictions}")
    print(f"Device: {device}")
    
    # Load model
    model = load_checkpoint(
        args.checkpoint,
        device=device,
        backbone=args.backbone,
        num_experts=args.num_experts
    )
    
    # Evaluate each dataset
    all_results = {}
    
    for dataset_name in args.datasets:
        dataset_path = Path(args.data_root) / dataset_name
        
        if not dataset_path.exists():
            print(f"\n⚠ Dataset not found: {dataset_path}")
            continue
        
        print(f"\n{'─'*70}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'─'*70}")
        
        try:
            # Create dataset
            dataset = EvalDataset(
                root_dir=str(dataset_path),
                img_size=args.img_size,
                dataset_name=dataset_name
            )
            
            # Create output directory for this dataset
            dataset_output_dir = output_dir / dataset_name if args.save_predictions else None
            if dataset_output_dir:
                dataset_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Evaluate
            overall_metrics, per_image_results = evaluate_dataset(
                model=model,
                dataset=dataset,
                device=device,
                batch_size=args.batch_size,
                save_predictions=args.save_predictions,
                output_dir=dataset_output_dir
            )
            
            all_results[dataset_name] = overall_metrics
            
            # Save per-image CSV for this dataset
            csv_path = output_dir / dataset_name / 'results.csv'
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            save_per_image_csv(per_image_results, csv_path)
            
            # Print dataset results
            print(f"  S-measure: {overall_metrics['S-measure']:.4f} ⭐")
            print(f"  IoU:       {overall_metrics['IoU']:.4f}")
            print(f"  F-measure: {overall_metrics['F-measure']:.4f}")
            print(f"  F-max:     {overall_metrics['F-max']:.4f}")
            print(f"  MAE:       {overall_metrics['MAE']:.4f}")
            print(f"  E-measure: {overall_metrics['E-measure']:.4f}")
            print(f"  Dice:      {overall_metrics['Dice']:.4f}")
            print(f"  FPS:       {overall_metrics['fps']:.1f}")
            
        except Exception as e:
            print(f"  ✗ Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compute average across datasets
    if len(all_results) > 1:
        avg_metrics = {}
        metric_keys = ['S-measure', 'IoU', 'F-measure', 'F-adaptive', 'F-max', 'MAE', 'E-measure', 'Dice']
        for key in metric_keys:
            values = [r[key] for r in all_results.values() if key in r]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        # Total timing
        total_images = sum(r.get('total_images', 0) for r in all_results.values())
        total_time = sum(r.get('total_time', 0) for r in all_results.values())
        avg_metrics['total_images'] = total_images
        avg_metrics['total_time'] = total_time
        avg_metrics['fps'] = total_images / total_time if total_time > 0 else 0
        
        all_results['Average'] = avg_metrics
    
    # Print final results table
    print_results_table(all_results)
    
    # Save summary CSV
    save_summary_csv(all_results, output_dir / 'summary.csv')
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
