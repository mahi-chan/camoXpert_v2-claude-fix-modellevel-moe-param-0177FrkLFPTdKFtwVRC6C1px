"""
Qualitative Analysis Script for CamoXpert

Generates:
1. Side-by-side comparisons: Input | Ground Truth | Prediction
2. Training curves (loss, IoU, S-measure)
3. Expert routing visualization
4. Failure case analysis
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model_level_moe import ModelLevelMoE
from models.multi_scale_processor import MultiScaleInputProcessor


def load_model(checkpoint_path: str, backbone: str = 'pvt_v2_b2', 
               expert_types: List[str] = None, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    if expert_types is None:
        expert_types = ['sinet', 'pranet', 'zoomnet']
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get state dict
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
    
    # Create model
    model = ModelLevelMoE(
        backbone_name=backbone,
        num_experts=len(expert_types),
        top_k=2,
        pretrained=False,
        expert_types=expert_types
    )
    
    # Wrap with multi-scale if needed
    if has_multi_scale:
        print("✓ Wrapping with MultiScaleInputProcessor")
        channels_list = [64, 128, 320, 512]
        model.backbone = MultiScaleInputProcessor(
            backbone=model.backbone,
            channels_list=channels_list,
            scales=[0.5, 1.0, 1.5],
            use_hierarchical=True
        )
    
    # Load weights
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    return model, checkpoint


def load_image(image_path: str, img_size: int = 384):
    """Load and preprocess image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = (image.shape[1], image.shape[0])  # W, H
    
    # Resize and normalize
    resized = cv2.resize(original, (img_size, img_size))
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (normalized - mean) / std
    
    # To tensor
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    
    return original, tensor, original_size


def load_gt(gt_path: str):
    """Load ground truth mask."""
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    return (gt > 128).astype(np.float32)


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: str = 'cuda'):
    """Run inference and get prediction."""
    image_tensor = image_tensor.to(device)
    
    output = model(image_tensor, return_routing_info=True)
    if isinstance(output, tuple):
        pred, routing_info = output
    else:
        pred = output
        routing_info = {}
    
    # Apply sigmoid if needed
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred = pred.squeeze().cpu().numpy()
    
    return pred, routing_info


def create_comparison_figure(images: List[Dict], output_path: str, title: str = "Qualitative Results"):
    """
    Create side-by-side comparison figure.
    
    images: List of dicts with keys: 'input', 'gt', 'pred', 'name'
    """
    n_images = len(images)
    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4 * n_images))
    
    if n_images == 1:
        axes = [axes]
    
    for i, img_data in enumerate(images):
        # Input
        axes[i][0].imshow(img_data['input'])
        axes[i][0].set_title('Input')
        axes[i][0].axis('off')
        
        # Ground Truth
        if img_data['gt'] is not None:
            axes[i][1].imshow(img_data['gt'], cmap='gray')
        else:
            axes[i][1].text(0.5, 0.5, 'N/A', ha='center', va='center')
        axes[i][1].set_title('Ground Truth')
        axes[i][1].axis('off')
        
        # Prediction
        axes[i][2].imshow(img_data['pred'], cmap='gray')
        axes[i][2].set_title('Prediction')
        axes[i][2].axis('off')
        
        # Overlay
        overlay = img_data['input'].copy()
        pred_mask = (img_data['pred'] > 0.5).astype(np.uint8)
        pred_resized = cv2.resize(pred_mask, (overlay.shape[1], overlay.shape[0]))
        overlay[pred_resized > 0] = [255, 0, 0]  # Red overlay
        axes[i][3].imshow(overlay)
        axes[i][3].set_title(f"Overlay - {img_data['name']}")
        axes[i][3].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_training_curves(checkpoint: Dict, output_path: str):
    """Plot training loss and metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Try to get training history
    history = checkpoint.get('history', {})
    epochs = range(1, checkpoint.get('epoch', 100) + 1)
    
    # If no history, try to parse from logs or create placeholder
    if not history:
        print("⚠ No training history in checkpoint. Creating placeholder plots.")
        
        for ax in axes.flatten():
            ax.text(0.5, 0.5, 'No training history available\nin checkpoint', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        plt.suptitle('Training Curves (Not Available)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Plot loss
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot IoU
    if 'val_iou' in history:
        axes[0, 1].plot(history['val_iou'], 'g-', label='Validation IoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('Validation IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot S-measure
    if 'val_smeasure' in history:
        axes[1, 0].plot(history['val_smeasure'], 'r-', label='S-measure')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('S-measure')
        axes[1, 0].set_title('Validation S-measure')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], 'm-', label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.suptitle('Training Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_expert_routing(routing_stats: Dict, output_path: str):
    """Visualize expert routing distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Expert usage pie chart
    if 'expert_usage' in routing_stats:
        usage = routing_stats['expert_usage']
        labels = [f'Expert {i}' for i in range(len(usage))]
        colors = plt.cm.Set3(np.linspace(0, 1, len(usage)))
        axes[0].pie(usage, labels=labels, autopct='%1.1f%%', colors=colors)
        axes[0].set_title('Expert Usage Distribution')
    else:
        axes[0].text(0.5, 0.5, 'No routing data', ha='center', va='center')
    
    # Routing entropy histogram
    if 'entropy' in routing_stats:
        axes[1].bar(['Routing Entropy'], [routing_stats['entropy']], color='steelblue')
        axes[1].set_ylabel('Entropy')
        axes[1].set_title(f"Routing Entropy: {routing_stats['entropy']:.3f}")
        axes[1].set_ylim(0, np.log(3))  # Max entropy for 3 experts
    else:
        axes[1].text(0.5, 0.5, 'No entropy data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def find_images(data_root: str, dataset: str = 'CAMO'):
    """Find test images and ground truths."""
    data_root = Path(data_root)
    
    # Common dataset structures
    structures = [
        (data_root / dataset / 'Images' / 'Test', data_root / dataset / 'GT' / 'Test'),
        (data_root / dataset / 'Test' / 'Image', data_root / dataset / 'Test' / 'GT'),
        (data_root / dataset / 'test' / 'images', data_root / dataset / 'test' / 'gts'),
        (data_root / 'Test' / 'Image', data_root / 'Test' / 'GT'),
    ]
    
    for img_dir, gt_dir in structures:
        if img_dir.exists():
            return img_dir, gt_dir if gt_dir.exists() else None
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Qualitative Analysis for CamoXpert')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data-root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--dataset', type=str, default='CAMO', help='Dataset name')
    parser.add_argument('--output-dir', type=str, default='./qualitative_results', help='Output directory')
    parser.add_argument('--num-images', type=int, default=8, help='Number of images to visualize')
    parser.add_argument('--img-size', type=int, default=384, help='Input image size')
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2', help='Backbone architecture')
    parser.add_argument('--expert-types', nargs='+', default=['sinet', 'pranet', 'zoomnet'],
                       help='Expert types used in training')
    parser.add_argument('--image-dir', type=str, default=None,
                       help='Explicit path to images folder (bypasses auto-detection)')
    parser.add_argument('--gt-dir', type=str, default=None,
                       help='Explicit path to ground truth folder')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("QUALITATIVE ANALYSIS")
    print("="*60)
    
    # Load model
    model, checkpoint = load_model(
        args.checkpoint, 
        args.backbone, 
        args.expert_types,
        device
    )
    
    # Find images - use explicit paths if provided
    if args.image_dir:
        img_dir = Path(args.image_dir)
        gt_dir = Path(args.gt_dir) if args.gt_dir else None
    else:
        img_dir, gt_dir = find_images(args.data_root, args.dataset)
    
    if img_dir is None or not img_dir.exists():
        print(f"⚠ Could not find images in {args.image_dir or args.data_root}")
        print("Please provide --image-dir explicitly")
        return
    
    print(f"Image directory: {img_dir}")
    print(f"GT directory: {gt_dir}")
    
    # Get image list
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    if not image_files:
        print("❌ No images found!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Select random subset
    np.random.seed(42)
    selected = np.random.choice(len(image_files), min(args.num_images, len(image_files)), replace=False)
    selected_images = [image_files[i] for i in selected]
    
    # Process images
    print("\nProcessing images...")
    results = []
    all_routing_probs = []
    
    for img_path in tqdm(selected_images):
        try:
            # Load image
            original, tensor, orig_size = load_image(str(img_path), args.img_size)
            
            # Load GT if available
            gt = None
            if gt_dir:
                gt_path = gt_dir / (img_path.stem + '.png')
                if not gt_path.exists():
                    gt_path = gt_dir / (img_path.stem + '.jpg')
                if gt_path.exists():
                    gt = load_gt(str(gt_path))
            
            # Predict
            pred, routing_info = predict(model, tensor, device)
            
            # Resize prediction to original size
            pred_resized = cv2.resize(pred, orig_size)
            
            results.append({
                'input': original,
                'gt': gt,
                'pred': pred_resized,
                'name': img_path.stem
            })
            
            # Collect routing info
            if 'routing_probs' in routing_info:
                probs = routing_info['routing_probs'].cpu().numpy()
                all_routing_probs.append(probs)
                # Debug: print per-image routing
                if probs.shape[1] >= 3:
                    print(f"  {img_path.stem}: E0={probs[0,0]*100:.1f}% E1={probs[0,1]*100:.1f}% E2={probs[0,2]*100:.1f}%")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Comparison figure
    create_comparison_figure(
        results, 
        str(output_dir / 'comparison.png'),
        f'Qualitative Results - {args.dataset}'
    )
    
    # 2. Training curves
    plot_training_curves(
        checkpoint,
        str(output_dir / 'training_curves.png')
    )
    
    # 3. Expert routing
    if all_routing_probs:
        avg_probs = np.mean(np.concatenate(all_routing_probs, axis=0), axis=0)
        routing_stats = {
            'expert_usage': avg_probs.tolist(),
            'entropy': -np.sum(avg_probs * np.log(avg_probs + 1e-8))
        }
        plot_expert_routing(
            routing_stats,
            str(output_dir / 'expert_routing.png')
        )
    
    # 4. Save individual predictions
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(exist_ok=True)
    for result in results:
        pred_path = pred_dir / f"{result['name']}_pred.png"
        cv2.imwrite(str(pred_path), (result['pred'] * 255).astype(np.uint8))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - comparison.png: Side-by-side comparisons")
    print(f"  - training_curves.png: Loss and metrics")
    print(f"  - expert_routing.png: Expert usage distribution")
    print(f"  - predictions/: Individual predictions")


if __name__ == '__main__':
    main()
