"""
CamoXpert Inference Script
Quick inference on single image or folder with all enhancements
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from pathlib import Path

from models.model_level_moe import ModelLevelMoE
from utils.tta_predictor import TTAPredictor
from utils.crf_refiner import CRFRefiner
from utils.threshold_optimizer import ThresholdOptimizer


def load_model(checkpoint_path, num_experts=4, device='cuda'):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        num_experts: Number of experts in the model (default: 4)
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from {checkpoint_path}")

    # Create model
    model = ModelLevelMoE(num_experts=num_experts)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Remove DDP 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully ({num_experts} experts)")

    return model


def preprocess(image_path, image_size=352):
    """
    Load and preprocess image for inference.

    Args:
        image_path: Path to input image
        image_size: Size to resize image to (default: 352)

    Returns:
        image_tensor: Preprocessed tensor [1, 3, H, W]
        image_np: Original image as numpy array
        original_size: (width, height) of original image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)

    # Resize
    image_resized = image.resize((image_size, image_size), Image.BILINEAR)

    # Convert to numpy
    image_np = np.array(image_resized).astype(np.float32) / 255.0

    # To tensor and normalize
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # HWC -> CHW

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    return image_tensor.unsqueeze(0), image_np, original_size


def postprocess(pred, original_size, threshold=0.5):
    """
    Convert prediction to binary mask at original size.

    Args:
        pred: Prediction tensor [1, 1, H, W]
        original_size: (width, height) to resize to
        threshold: Binary threshold

    Returns:
        pred_resized: Probability map at original size
        pred_binary: Binary mask at original size (uint8, 0-255)
    """
    # Convert to numpy
    pred_np = pred.squeeze().cpu().numpy()

    # Resize to original size
    pred_resized = cv2.resize(pred_np, original_size, interpolation=cv2.INTER_LINEAR)

    # Binarize
    pred_binary = (pred_resized > threshold).astype(np.uint8) * 255

    return pred_resized, pred_binary


def infer_single(model, image_path, args, device, tta=None, crf=None):
    """
    Run inference on a single image.

    Args:
        model: Loaded model
        image_path: Path to input image
        args: Command-line arguments
        device: Device to run on
        tta: TTA predictor (optional)
        crf: CRF refiner (optional)

    Returns:
        pred_prob: Probability map at original size
        pred_binary: Binary mask at original size
        threshold: Threshold used
    """
    # Preprocess
    image_tensor, image_np, original_size = preprocess(image_path, args.image_size)
    image_tensor = image_tensor.to(device)

    # Predict
    with torch.no_grad():
        if tta is not None:
            # Use TTA
            pred = tta.predict(image_tensor)
        else:
            # Standard forward pass
            output = model(image_tensor)
            pred = output['pred'] if isinstance(output, dict) else output
            pred = torch.sigmoid(pred)

    # Extract prediction as numpy
    pred_np = pred[0, 0].cpu().numpy()

    # Apply CRF refinement if requested
    if crf is not None:
        # CRF expects uint8 RGB image
        image_uint8 = (image_np * 255).astype(np.uint8)
        pred_np = crf.refine(image_uint8, pred_np)

    # Determine threshold
    if args.threshold_method == 'otsu':
        threshold = ThresholdOptimizer.otsu(pred_np)
    elif args.threshold_method == 'adaptive':
        threshold = ThresholdOptimizer.adaptive(pred_np, method='mean')
    else:
        threshold = args.threshold

    # Postprocess to original size
    pred_prob, pred_binary = postprocess(
        torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0),
        original_size,
        threshold
    )

    return pred_prob, pred_binary, threshold


def create_overlay(image_path, mask_binary):
    """
    Create overlay visualization.

    Args:
        image_path: Path to original image
        mask_binary: Binary mask (0-255)

    Returns:
        Overlay image as numpy array
    """
    # Load original image
    image = cv2.imread(str(image_path))

    # Resize to match mask
    image = cv2.resize(image, (mask_binary.shape[1], mask_binary.shape[0]))

    # Create colored mask (green for detected object)
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask_binary > 127] = [0, 255, 0]  # Green in BGR

    # Blend
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

    return overlay


def main():
    parser = argparse.ArgumentParser(description='CamoXpert Inference')

    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or folder containing images')
    parser.add_argument('--output', type=str, default='./inference_results',
                        help='Output directory for results (default: ./inference_results)')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num-experts', type=int, default=4,
                        help='Number of experts in the model (default: 4)')
    parser.add_argument('--image-size', type=int, default=352,
                        help='Input image size (default: 352)')

    # Enhancements
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation (multi-scale + flip)')
    parser.add_argument('--use-crf', action='store_true',
                        help='Apply CRF post-processing for boundary refinement')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary threshold (default: 0.5)')
    parser.add_argument('--threshold-method', type=str, default='fixed',
                        choices=['fixed', 'otsu', 'adaptive'],
                        help='Threshold selection method (default: fixed)')

    # Output options
    parser.add_argument('--save-prob', action='store_true',
                        help='Save probability maps (grayscale heatmaps)')
    parser.add_argument('--save-overlay', action='store_true',
                        help='Save overlay visualizations')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*70)
    print("CAMOEXPERT INFERENCE")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"TTA: {'Enabled' if args.tta else 'Disabled'}")
    print(f"CRF: {'Enabled' if args.use_crf else 'Disabled'}")
    print(f"Threshold method: {args.threshold_method}")
    if args.threshold_method == 'fixed':
        print(f"  Threshold: {args.threshold}")
    print(f"Device: {device}")
    print("="*70 + "\n")

    # Load model
    model = load_model(args.checkpoint, args.num_experts, device)

    # Setup enhancements
    tta = TTAPredictor(model) if args.tta else None
    crf = CRFRefiner() if args.use_crf else None

    if args.tta:
        print(f"✓ TTA enabled: {tta.get_num_augmentations()} augmentations")
    if args.use_crf:
        try:
            from utils.crf_refiner import HAS_CRF
            if HAS_CRF:
                print("✓ CRF enabled: Using Dense CRF (pydensecrf)")
            else:
                print("✓ CRF enabled: Using morphological refinement")
        except:
            print("✓ CRF enabled: Using morphological refinement")

    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        # Find all images in directory
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
        image_paths = sorted(set(image_paths))
    else:
        raise ValueError(f"Input path does not exist: {args.input}")

    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {args.input}")

    print(f"\nProcessing {len(image_paths)} image(s)...\n")

    # Process each image
    for image_path in tqdm(image_paths, desc="Inference"):
        name = image_path.stem

        try:
            # Run inference
            pred_prob, pred_binary, threshold = infer_single(
                model, image_path, args, device, tta, crf
            )

            # Save binary mask
            mask_path = output_dir / f'{name}_mask.png'
            cv2.imwrite(str(mask_path), pred_binary)

            # Save probability map
            if args.save_prob:
                prob_uint8 = (pred_prob * 255).astype(np.uint8)
                prob_path = output_dir / f'{name}_prob.png'
                cv2.imwrite(str(prob_path), prob_uint8)

            # Save overlay
            if args.save_overlay:
                overlay = create_overlay(image_path, pred_binary)
                overlay_path = output_dir / f'{name}_overlay.png'
                cv2.imwrite(str(overlay_path), overlay)

        except Exception as e:
            print(f"\n❌ Error processing {image_path.name}: {e}")
            continue

    print(f"\n✅ Inference complete! Results saved to: {output_dir}")
    print(f"\nSaved outputs:")
    print(f"  - Binary masks: *_mask.png")
    if args.save_prob:
        print(f"  - Probability maps: *_prob.png")
    if args.save_overlay:
        print(f"  - Overlays: *_overlay.png")
    print()


if __name__ == '__main__':
    main()
