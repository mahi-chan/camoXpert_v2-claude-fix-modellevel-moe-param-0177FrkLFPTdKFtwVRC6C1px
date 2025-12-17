"""
Quick debug script to check what's happening with CAMO evaluation.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import sys
sys.path.insert(0, '.')

from models.model_level_moe import ModelLevelMoE

def debug_evaluation():
    print("=" * 60)
    print("CAMO Evaluation Debug")
    print("=" * 60)
    
    # Paths
    checkpoint_path = "./checkpoints_multiscale/best_model.pth"
    camo_img_dir = Path("./CAMO-V.1.0-CVIU2019/Images/Test")
    camo_gt_dir = Path("./CAMO-V.1.0-CVIU2019/GT")
    
    # Check paths exist
    print(f"\n1. Checking paths...")
    print(f"   Checkpoint: {Path(checkpoint_path).exists()}")
    print(f"   CAMO Images: {camo_img_dir.exists()}")
    print(f"   CAMO GT: {camo_gt_dir.exists()}")
    
    if not Path(checkpoint_path).exists():
        print("   ERROR: Checkpoint not found!")
        return
    
    # Load model
    print(f"\n2. Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    model = ModelLevelMoE(backbone='pvt_v2_b2', num_experts=3, top_k=2)
    
    # Handle DDP prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.cuda().eval()
    print("   Model loaded successfully!")
    
    # Get sample image
    print(f"\n3. Testing on sample CAMO image...")
    img_files = list(camo_img_dir.glob("*.jpg"))
    if not img_files:
        img_files = list(camo_img_dir.glob("*.png"))
    
    if not img_files:
        print("   ERROR: No images found!")
        return
    
    sample_img_path = img_files[0]
    base_name = sample_img_path.stem
    print(f"   Sample: {sample_img_path.name}")
    
    # Load image
    image = cv2.imread(str(sample_img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_size = image.shape[:2]
    print(f"   Original size: {orig_size}")
    
    # Load GT
    gt_path = camo_gt_dir / f"{base_name}.png"
    if not gt_path.exists():
        print(f"   ERROR: GT not found at {gt_path}")
        return
    
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    print(f"   GT stats: min={gt.min()}, max={gt.max()}, mean={gt.mean():.2f}")
    print(f"   GT non-zero pixels: {(gt > 128).sum()} / {gt.size}")
    
    # Preprocess
    img_size = 384
    image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image_norm = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    image_norm = (image_norm - mean) / std
    image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
    
    # Inference
    print(f"\n4. Running inference...")
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output
        
        pred_sigmoid = torch.sigmoid(pred)
    
    pred_np = pred_sigmoid.squeeze().cpu().numpy()
    print(f"   Prediction (before sigmoid) stats: min={pred.min():.4f}, max={pred.max():.4f}")
    print(f"   Prediction (after sigmoid) stats: min={pred_np.min():.4f}, max={pred_np.max():.4f}, mean={pred_np.mean():.4f}")
    print(f"   Prediction > 0.5: {(pred_np > 0.5).sum()} / {pred_np.size}")
    
    # Check IoU manually
    gt_resized = cv2.resize(gt, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_resized > 128).astype(np.float32)
    pred_binary = (pred_np > 0.5).astype(np.float32)
    
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    print(f"\n5. Manual IoU calculation:")
    print(f"   GT positive pixels: {gt_binary.sum():.0f}")
    print(f"   Pred positive pixels: {pred_binary.sum():.0f}")
    print(f"   Intersection: {intersection:.0f}")
    print(f"   Union: {union:.0f}")
    print(f"   IoU: {iou:.4f}")
    
    # Save prediction for visual check
    pred_save = (pred_np * 255).astype(np.uint8)
    Image.fromarray(pred_save).save("debug_pred.png")
    print(f"\n   Saved debug_pred.png - please check visually!")
    
    # Also save GT for comparison
    Image.fromarray(gt_resized).save("debug_gt.png")
    print(f"   Saved debug_gt.png - please compare!")

if __name__ == "__main__":
    debug_evaluation()
