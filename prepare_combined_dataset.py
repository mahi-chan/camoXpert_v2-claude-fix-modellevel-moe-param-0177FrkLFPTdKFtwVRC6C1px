"""
Prepare Combined Dataset for SOTA COD Training

Combines COD10K and CAMO training sets for multi-dataset training.

Usage:
    python prepare_combined_dataset.py
    
This will create:
    combined_dataset/
    ├── Train/
    │   ├── Image/  (COD10K + CAMO train images)
    │   └── GT_Object/  (COD10K + CAMO train GTs)
    └── Test/
        ├── Image/  (COD10K test images only)
        └── GT_Object/  (COD10K test GTs only)
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration - adjust these paths if needed
COD10K_ROOT = Path("./COD10K-v3")
CAMO_ROOT = Path("./CAMO-V.1.0-CVIU2019")
OUTPUT_ROOT = Path("./combined_dataset")

def count_files(directory, extensions=['.jpg', '.png', '.jpeg']):
    """Count image files in directory."""
    if not directory.exists():
        return 0
    count = 0
    for ext in extensions:
        count += len(list(directory.glob(f'*{ext}')))
        count += len(list(directory.glob(f'*{ext.upper()}')))
    return count

def copy_files(src_dir, dst_dir, file_list=None, desc="Copying"):
    """Copy files from src to dst."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if file_list is None:
        # Copy all image files
        extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        file_list = []
        for ext in extensions:
            file_list.extend(src_dir.glob(f'*{ext}'))
    
    for src_file in tqdm(file_list, desc=desc, leave=False):
        dst_file = dst_dir / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)

def find_matching_gt(image_name, gt_dir):
    """Find GT file matching image name (handles different extensions)."""
    base_name = Path(image_name).stem
    
    for ext in ['.png', '.jpg', '.PNG', '.JPG']:
        gt_path = gt_dir / f"{base_name}{ext}"
        if gt_path.exists():
            return gt_path
    return None

def prepare_combined_dataset():
    print("=" * 60)
    print("Preparing Combined Dataset (COD10K + CAMO)")
    print("=" * 60)
    
    # Check source directories
    print("\n1. Checking source directories...")
    
    cod10k_train_img = COD10K_ROOT / "Train" / "Image"
    cod10k_train_gt = COD10K_ROOT / "Train" / "GT_Object"
    cod10k_test_img = COD10K_ROOT / "Test" / "Image"
    cod10k_test_gt = COD10K_ROOT / "Test" / "GT_Object"
    
    camo_train_img = CAMO_ROOT / "Images" / "Train"
    camo_gt = CAMO_ROOT / "GT"  # All GTs in one folder
    
    print(f"   COD10K Train Images: {count_files(cod10k_train_img)} files")
    print(f"   COD10K Train GT: {count_files(cod10k_train_gt)} files")
    print(f"   COD10K Test Images: {count_files(cod10k_test_img)} files")
    print(f"   CAMO Train Images: {count_files(camo_train_img)} files")
    print(f"   CAMO GT (all): {count_files(camo_gt)} files")
    
    # Verify directories exist
    missing = []
    for path, name in [(cod10k_train_img, "COD10K Train Images"), 
                       (cod10k_train_gt, "COD10K Train GT"),
                       (camo_train_img, "CAMO Train Images"),
                       (camo_gt, "CAMO GT")]:
        if not path.exists():
            missing.append(f"  - {name}: {path}")
    
    if missing:
        print("\n❌ Missing directories:")
        for m in missing:
            print(m)
        print("\nPlease check your dataset paths and try again.")
        return False
    
    # Create output directories
    print("\n2. Creating output directories...")
    out_train_img = OUTPUT_ROOT / "Train" / "Image"
    out_train_gt = OUTPUT_ROOT / "Train" / "GT_Object"
    out_test_img = OUTPUT_ROOT / "Test" / "Image"
    out_test_gt = OUTPUT_ROOT / "Test" / "GT_Object"
    
    for d in [out_train_img, out_train_gt, out_test_img, out_test_gt]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Copy COD10K Train
    print("\n3. Copying COD10K Train set...")
    copy_files(cod10k_train_img, out_train_img, desc="COD10K Train Images")
    copy_files(cod10k_train_gt, out_train_gt, desc="COD10K Train GT")
    
    # Copy COD10K Test (keep separate for evaluation)
    print("\n4. Copying COD10K Test set...")
    copy_files(cod10k_test_img, out_test_img, desc="COD10K Test Images")
    copy_files(cod10k_test_gt, out_test_gt, desc="COD10K Test GT")
    
    # Copy CAMO Train (find matching GTs)
    print("\n5. Copying CAMO Train set...")
    camo_train_images = list(camo_train_img.glob('*.jpg')) + list(camo_train_img.glob('*.png'))
    
    copied = 0
    missing_gt = 0
    for img_path in tqdm(camo_train_images, desc="CAMO Train"):
        # Copy image
        dst_img = out_train_img / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        
        # Find and copy matching GT
        gt_path = find_matching_gt(img_path.name, camo_gt)
        if gt_path:
            dst_gt = out_train_gt / gt_path.name
            if not dst_gt.exists():
                shutil.copy2(gt_path, dst_gt)
            copied += 1
        else:
            missing_gt += 1
    
    if missing_gt > 0:
        print(f"   ⚠ Warning: {missing_gt} CAMO images had no matching GT")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Combined Dataset Created!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_ROOT.absolute()}")
    print(f"\nDataset Statistics:")
    print(f"   Train Images: {count_files(out_train_img)}")
    print(f"   Train GTs: {count_files(out_train_gt)}")
    print(f"   Test Images: {count_files(out_test_img)}")
    print(f"   Test GTs: {count_files(out_test_gt)}")
    
    print(f"\n✓ Ready for training! Use:")
    print(f"   --data-root {OUTPUT_ROOT}")
    
    return True

if __name__ == "__main__":
    prepare_combined_dataset()
