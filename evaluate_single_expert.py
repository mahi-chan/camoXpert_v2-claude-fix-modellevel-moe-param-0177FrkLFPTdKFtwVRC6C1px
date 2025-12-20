"""
Evaluate Single Expert Model on CAMO Test Set
"""
import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_single_expert import SingleExpertModel, EXPERT_CLASSES, BACKBONE_DIMS
from dataset import COD10KDataset
from metrics.cod_metrics import CODMetrics
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Evaluate Single Expert')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--expert', type=str, default=None, 
                       help='Expert type (auto-detected from checkpoint if not specified)')
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--img-size', type=int, default=448)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--use-multi-scale', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Auto-detect expert type from checkpoint
    expert_type = args.expert or checkpoint.get('expert_type', 'sinet')
    backbone = checkpoint.get('backbone', args.backbone)
    use_multi_scale = checkpoint.get('use_multi_scale', args.use_multi_scale)
    
    print(f"Expert: {expert_type}")
    print(f"Backbone: {backbone}")
    print(f"Multi-scale: {use_multi_scale}")
    
    # Create model
    model = SingleExpertModel(
        backbone_name=backbone,
        expert_type=expert_type,
        pretrained=False,
        use_multi_scale=use_multi_scale
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print(f"\nLoading test dataset from: {args.data_root}")
    test_dataset = COD10KDataset(
        root_dir=args.data_root,
        split='test',
        img_size=args.img_size,
        augment=False,
        cache_in_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    print(f"Test images: {len(test_dataset)}")
    
    # Evaluate
    metrics = CODMetrics()
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            pred, _ = model(images, return_aux=False)
            
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            
            metrics.update(pred, masks)
    
    results = metrics.compute()
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS - {expert_type.upper()}")
    print(f"{'='*50}")
    print(f"  S-measure: {results.get('S-measure', 0):.4f}")
    print(f"  IoU:       {results.get('IoU', 0):.4f}")
    print(f"  F-measure: {results.get('F-measure', 0):.4f}")
    print(f"  MAE:       {results.get('MAE', 0):.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
