"""
Evaluate Single Expert Model on CAMO Test Set
"""
import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_single_expert import SingleExpertModel, EXPERT_CLASSES, BACKBONE_DIMS
from metrics.cod_metrics import CODMetrics


class SimpleTestDataset:
    """Simple dataset that loads from explicit image and GT directories."""
    
    def __init__(self, image_dir, gt_dir, img_size=448):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        
        # Find all images
        self.image_list = sorted([f for f in os.listdir(image_dir) 
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"Found {len(self.image_list)} images in {image_dir}")
        
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        base_name = os.path.splitext(img_name)[0]
        mask = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            mask_path = os.path.join(self.gt_dir, base_name + ext)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                break
        
        if mask is None:
            raise ValueError(f"Mask not found for {img_name}")
        
        mask = (mask > 128).astype(np.float32)
        
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask'].unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Single Expert')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True, help='Path to test images')
    parser.add_argument('--gt-dir', type=str, required=True, help='Path to ground truth masks')
    parser.add_argument('--expert', type=str, default=None)
    parser.add_argument('--backbone', type=str, default='pvt_v2_b2')
    parser.add_argument('--img-size', type=int, default=448)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--use-multi-scale', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary predictions')
    parser.add_argument('--find-best-threshold', action='store_true', help='Search for optimal threshold')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
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
    test_dataset = SimpleTestDataset(args.image_dir, args.gt_dir, args.img_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Collect all predictions for threshold search
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            pred, _ = model(images, return_aux=False)
            
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            
            all_preds.append(pred.cpu())
            all_masks.append(masks.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Find optimal threshold
    if args.find_best_threshold:
        print("\nSearching for optimal threshold...")
        best_iou = 0
        best_thresh = 0.5
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            metrics = CODMetrics()
            metrics.update(all_preds, all_masks, threshold=thresh)
            results = metrics.compute()
            iou = results.get('IoU', 0)
            print(f"  Threshold {thresh:.1f}: IoU={iou:.4f}, F={results.get('F-measure', 0):.4f}")
            if iou > best_iou:
                best_iou = iou
                best_thresh = thresh
        print(f"\nBest threshold: {best_thresh} (IoU={best_iou:.4f})")
        args.threshold = best_thresh
    
    # Final evaluation with chosen threshold
    metrics = CODMetrics()
    metrics.update(all_preds, all_masks, threshold=args.threshold)
    results = metrics.compute()
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS - {expert_type.upper()}")
    print(f"Threshold: {args.threshold}")
    print(f"{'='*50}")
    print(f"  S-measure: {results.get('S-measure', 0):.4f}")
    print(f"  IoU:       {results.get('IoU', 0):.4f}")
    print(f"  F-measure: {results.get('F-measure', 0):.4f}")
    print(f"  MAE:       {results.get('MAE', 0):.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

