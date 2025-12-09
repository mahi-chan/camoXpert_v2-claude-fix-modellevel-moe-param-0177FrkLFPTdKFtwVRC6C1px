import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COD10KDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=512, augment=True, cache_in_memory=True,
                 rank=0, world_size=1):
        """
        Args:
            rank: DDP rank (0, 1, 2, ...) - which GPU process this is
            world_size: Total number of DDP processes
            img_size: Improved default to 512 for better small object detection
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        self.cache_in_memory = cache_in_memory
        self.rank = rank
        self.world_size = world_size

        # Try multiple possible directory structures
        possible_structures = [
            # Structure 1: Train/Image, Train/GT_Object
            {
                'train_img': 'Train/Image',
                'train_mask': 'Train/GT_Object',
                'test_img': 'Test/Image',
                'test_mask': 'Test/GT_Object'
            },
            # Structure 2: Train/Imgs, Train/GT
            {
                'train_img': 'Train/Imgs',
                'train_mask': 'Train/GT',
                'test_img': 'Test/Imgs',
                'test_mask': 'Test/GT'
            },
            # Structure 3: Direct Train/Test folders
            {
                'train_img': 'Train',
                'train_mask': 'TrainGT',
                'test_img': 'Test',
                'test_mask': 'TestGT'
            },
            # Structure 4: Flat structure with subfolders
            {
                'train_img': 'TrainDataset/Imgs',
                'train_mask': 'TrainDataset/GT',
                'test_img': 'TestDataset/Imgs',
                'test_mask': 'TestDataset/GT'
            }
        ]

        # Find the correct structure
        structure = None
        for struct in possible_structures:
            if split == 'train':
                img_path = os.path.join(root_dir, struct['train_img'])
            else:
                img_path = os.path.join(root_dir, struct['test_img'])

            if os.path.exists(img_path):
                structure = struct
                break

        if structure is None:
            # List available directories to help debug
            available = os.listdir(root_dir) if os.path.exists(root_dir) else []
            raise FileNotFoundError(
                f"Could not find COD10K dataset structure in {root_dir}\n"
                f"Available directories: {available}\n"
                f"Please check your dataset structure."
            )

        # Set paths based on found structure
        if split == 'train':
            self.image_dir = os.path.join(root_dir, structure['train_img'])
            self.mask_dir = os.path.join(root_dir, structure['train_mask'])
            self.image_list = sorted(os.listdir(self.image_dir))
        elif split == 'val':
            self.image_dir = os.path.join(root_dir, structure['test_img'])
            self.mask_dir = os.path.join(root_dir, structure['test_mask'])
            all_images = sorted(os.listdir(self.image_dir))
            # Use 20% of test set for validation
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[:val_size]
        else:  # test
            self.image_dir = os.path.join(root_dir, structure['test_img'])
            self.mask_dir = os.path.join(root_dir, structure['test_mask'])
            all_images = sorted(os.listdir(self.image_dir))
            # Use remaining 80% for testing
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[val_size:]

        if self.rank == 0:
            print(f"{split.upper()} dataset initialized: {len(self.image_list)} images")
            print(f"  Image dir: {self.image_dir}")
            print(f"  Mask dir:  {self.mask_dir}")
            print(f"  Resolution: {img_size}x{img_size}")

        # Cache RESIZED images and masks in memory
        self.image_cache = {}
        self.mask_cache = {}

        if self.cache_in_memory:
            # DDP-aware caching: only cache images this rank will use
            if self.world_size > 1:
                # Calculate which indices this rank will handle
                indices_to_cache = list(range(self.rank, len(self.image_list), self.world_size))
                images_to_cache = [self.image_list[i] for i in indices_to_cache]
                if self.rank == 0:
                    print(f"  [Rank {self.rank}/{self.world_size}] Caching {len(images_to_cache)}/{len(self.image_list)} RESIZED images ({img_size}px) in RAM...")
            else:
                images_to_cache = self.image_list
                if self.rank == 0:
                    print(f"  Caching {len(images_to_cache)} RESIZED images ({img_size}px) in RAM...")

            from tqdm import tqdm
            pbar = tqdm(images_to_cache, desc=f"Loading {split} [Rank {self.rank}]") if self.rank == 0 else images_to_cache
            
            for img_name in pbar:
                # Load image
                img_path = os.path.join(self.image_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue  # Skip broken images
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # RESIZE NOW to save RAM
                image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                self.image_cache[img_name] = image  # Keep as uint8

                # Load mask
                mask_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']
                base_name = os.path.splitext(img_name)[0]
                mask = None
                for ext in mask_extensions:
                    mask_path = os.path.join(self.mask_dir, base_name + ext)
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        break
                
                if mask is None:
                    continue # Skip if no mask found
                    
                # RESIZE mask too
                mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                self.mask_cache[img_name] = (mask > 128).astype(np.float32)

            mem_mb = len(self.image_cache) * img_size * img_size * 3 / (1024**2)
            if self.rank == 0:
                print(f"  ✓ Cached {len(self.image_cache)} images in RAM (~{mem_mb:.0f}MB)")

        if self.augment:
            # FIX: Updated augmentations to avoid warnings and errors
            self.transform = A.Compose([
                # Geometric - Stronger for COD
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                
                # FIX 1: Replace ShiftScaleRotate with Affine
                A.Affine(
                    scale=(0.75, 1.25),
                    translate_percent=(0.0, 0.15),
                    rotate=(-20, 20),
                    shear=(-10, 10),
                    p=0.6,
                    mode=cv2.BORDER_CONSTANT
                ),

                # Noise/Blur
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ], p=0.4),

                # Color - Stronger
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),

                # FIX 2: Correct CoarseDropout arguments
                # 'max_holes', 'max_height', etc. are deprecated or invalid in newer versions
                # Use 'num_holes_range', 'hole_height_range', 'hole_width_range'
                A.CoarseDropout(
                    num_holes_range=(2, 8),
                    hole_height_range=(16, 32),
                    hole_width_range=(16, 32),
                    p=0.3
                ),

                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]

        # Check if cached
        if self.cache_in_memory and img_name in self.image_cache:
            image = self.image_cache[img_name].copy()
            mask = self.mask_cache[img_name].copy()
        else:
            # Load from disk
            img_path = os.path.join(self.image_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                # Return a zero image if load fails (shouldn't happen with correct data)
                image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
                mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

                # Mask
                mask_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']
                base_name = os.path.splitext(img_name)[0]
                mask = None
                for ext in mask_extensions:
                    mask_path = os.path.join(self.mask_dir, base_name + ext)
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        break
                
                if mask is None:
                    mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
                else:
                    mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 128).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask'].unsqueeze(0)


class CAMODataset(COD10KDataset):
    def __init__(self, root_dir, split='train', img_size=512, augment=True):
        super().__init__(root_dir, split, img_size, augment)
        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'Images/Train')
            self.mask_dir = os.path.join(root_dir, 'GT/Train')
        else:
            self.image_dir = os.path.join(root_dir, 'Images/Test')
            self.mask_dir = os.path.join(root_dir, 'GT/Test')
        self.image_list = sorted(os.listdir(self.image_dir))


class NC4KDataset(COD10KDataset):
    def __init__(self, root_dir, split='test', img_size=512, augment=False):
        super().__init__(root_dir, split, img_size, augment)
        self.image_dir = os.path.join(root_dir, 'image')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.image_list = sorted(os.listdir(self.image_dir))
