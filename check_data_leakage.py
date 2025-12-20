"""
Check for data leakage between training and test sets.
Verifies no CAMO test images are accidentally in combined_dataset/train.
"""
import os
from pathlib import Path

def get_basenames(directory):
    """Get set of basenames (without extension) from directory."""
    if not os.path.exists(directory):
        return set()
    
    basenames = set()
    for f in os.listdir(directory):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            basenames.add(os.path.splitext(f)[0].lower())
    return basenames

def main():
    print("=" * 60)
    print("DATA LEAKAGE CHECK")
    print("=" * 60)
    
    # Paths
    combined_train = Path("./combined_dataset/Train/Image")
    camo_test = Path("./CAMO-V.1.0-CVIU2019/Images/Test")
    cod10k_test = Path("./COD10K-v3/Test/Image")
    
    # Get basenames
    print("\nLoading file lists...")
    train_names = get_basenames(combined_train)
    camo_test_names = get_basenames(camo_test)
    cod10k_test_names = get_basenames(cod10k_test)
    
    print(f"  Combined train: {len(train_names)} images")
    print(f"  CAMO test: {len(camo_test_names)} images")
    print(f"  COD10K test: {len(cod10k_test_names)} images")
    
    # Check for overlap
    print("\n" + "=" * 60)
    print("CHECKING FOR LEAKAGE...")
    print("=" * 60)
    
    # CAMO test in train?
    camo_overlap = train_names & camo_test_names
    if camo_overlap:
        print(f"\n❌ LEAKAGE FOUND: {len(camo_overlap)} CAMO test images in training!")
        print("  Examples:", list(camo_overlap)[:10])
    else:
        print("\n✓ No CAMO test images in training set")
    
    # COD10K test in train?
    cod10k_overlap = train_names & cod10k_test_names
    if cod10k_overlap:
        print(f"\n❌ LEAKAGE FOUND: {len(cod10k_overlap)} COD10K test images in training!")
        print("  Examples:", list(cod10k_overlap)[:10])
    else:
        print("\n✓ No COD10K test images in training set")
    
    # Additional check: What's in combined train?
    print("\n" + "=" * 60)
    print("TRAINING SET COMPOSITION")
    print("=" * 60)
    
    camo_train = Path("./CAMO-V.1.0-CVIU2019/Images/Train")
    cod10k_train = Path("./COD10K-v3/Train/Image")
    
    camo_train_names = get_basenames(camo_train)
    cod10k_train_names = get_basenames(cod10k_train)
    
    # How much of combined_train is from each?
    from_camo = train_names & camo_train_names
    from_cod10k = train_names & cod10k_train_names
    
    print(f"\n  From CAMO train: {len(from_camo)}")
    print(f"  From COD10K train: {len(from_cod10k)}")
    print(f"  Total in combined: {len(train_names)}")
    
    unknown = train_names - from_camo - from_cod10k
    if unknown:
        print(f"\n  ⚠ Unknown source: {len(unknown)} images")
        print("    Examples:", list(unknown)[:5])


if __name__ == "__main__":
    main()
