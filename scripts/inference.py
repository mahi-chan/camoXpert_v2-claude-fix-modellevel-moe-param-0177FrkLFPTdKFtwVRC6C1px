import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
from pathlib import Path

from models.camoxpert import CamoXpert
from models.utils import load_checkpoint


def preprocess_image(image_path, img_size):
    """Preprocess the input image for inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    image = image / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    return image, original_size


def postprocess_mask(mask, original_size):
    """Postprocess the output mask."""
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    print("Loading model...")
    model = CamoXpert(in_channels=3, num_classes=1)
    load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Preprocess the input image
    print(f"Loading image: {args.image_path}")
    image, original_size = preprocess_image(args.image_path, args.img_size)
    image = image.to(device)

    # Perform inference
    print("Running inference...")
    with torch.no_grad():
        output, _ = model(image)

    # Postprocess the mask
    binary_mask = postprocess_mask(output, original_size)

    # Save the output mask
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"{Path(args.image_path).stem}_mask.png"
    cv2.imwrite(str(output_path), binary_mask * 255)

    print(f"Mask saved to {output_path}")


# Create parser at module level
parser = argparse.ArgumentParser(description="CamoXpert Inference Script")
parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save the output mask")
parser.add_argument("--img-size", type=int, default=352, help="Image size for inference")
parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)