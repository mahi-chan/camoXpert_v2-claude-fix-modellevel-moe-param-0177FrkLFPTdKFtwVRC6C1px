import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COD10KDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=352, augment=True, cache_in_memory=True,
                 rank=0, world_size=1):
        """
        Args:
            rank: DDP rank (0, 1, 2, ...) - which GPU process this is
            world_size: Total number of DDP processes

        DDP-aware caching: Each process only caches images it will actually use
        - rank=0, world_size=2: caches indices [0, 2, 4, 6, ...]
        - rank=1, world_size=2: caches indices [1, 3, 5, 7, ...]
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
                mask_path = os.path.join(root_dir, struct['train_mask'])
            else:
                img_path = os.path.join(root_dir, struct['test_img'])
                mask_path = os.path.join(root_dir, struct['test_mask'])

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
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[:val_size]
        else:  # test
            self.image_dir = os.path.join(root_dir, structure['test_img'])
            self.mask_dir = os.path.join(root_dir, structure['test_mask'])
            all_images = sorted(os.listdir(self.image_dir))
            val_size = int(len(all_images) * 0.2)
            self.image_list = all_images[val_size:]

        print(f"{split.upper()} dataset initialized: {len(self.image_list)} images")
        print(f"  Image dir: {self.image_dir}")
        print(f"  Mask dir:  {self.mask_dir}")

        # Cache RESIZED images and masks in memory (much smaller!)
        # Caching full-size images uses too much RAM
        self.image_cache = {}
        self.mask_cache = {}

        if self.cache_in_memory:
            # DDP-aware caching: only cache images this rank will use
            if self.world_size > 1:
                # Calculate which indices this rank will handle
                indices_to_cache = list(range(self.rank, len(self.image_list), self.world_size))
                images_to_cache = [self.image_list[i] for i in indices_to_cache]
                print(f"  [Rank {self.rank}/{self.world_size}] Caching {len(images_to_cache)}/{len(self.image_list)} RESIZED images ({img_size}px) in RAM...")
            else:
                images_to_cache = self.image_list
                print(f"  Caching {len(images_to_cache)} RESIZED images ({img_size}px) in RAM...")

            from tqdm import tqdm
            for img_name in tqdm(images_to_cache, desc=f"Loading {split} [Rank {self.rank}]"):
                # Load image
                img_path = os.path.join(self.image_dir, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # RESIZE NOW to save RAM (320x320 vs 400x600 = 60% smaller!)
                image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                self.image_cache[img_name] = image  # Keep as uint8 (4x smaller than float32)

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
                    raise ValueError(f"Failed to load mask for: {img_name}")
                # RESIZE mask too
                mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                self.mask_cache[img_name] = (mask > 128).astype(np.float32)

            mem_mb = len(self.image_cache) * img_size * img_size * 3 / (1024**2)
            if self.world_size > 1:
                print(f"  ✓ [Rank {self.rank}] Cached {len(self.image_cache)} images in RAM (~{mem_mb:.0f}MB)")
            else:
                print(f"  ✓ Cached {len(self.image_cache)} images in RAM (~{mem_mb:.0f}MB)")

        if self.augment:
            # NO RESIZE in transform - already done in cache!
            # STRONGER augmentation from start to prevent overfitting
            self.transform = A.Compose([
                # Geometric - STRONGER
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),  # Increased from 0.3
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=20, p=0.6),  # Stronger

                # Noise/Blur - Keep similar
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ], p=0.4),  # Increased from 0.3

                # Color - STRONGER
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),  # Stronger
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),  # Stronger

                # Cutout - ADD THIS (very effective for overfitting)
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=16, min_width=16, p=0.3),

                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # NO RESIZE in transform - already done in cache!
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]

        # Check if cached (DDP-aware caching may not have all images)
        if self.cache_in_memory and img_name in self.image_cache:
            # Get from RAM cache (fast!)
            image = self.image_cache[img_name].copy()  # Copy to avoid modifying cache
            mask = self.mask_cache[img_name].copy()
        else:
            # Load from disk (happens when: 1) caching disabled, or 2) cache miss in DDP)
            img_path = os.path.join(self.image_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # RESIZE to match cached images
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            # Try multiple mask extensions
            mask_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']
            base_name = os.path.splitext(img_name)[0]

            mask = None
            for ext in mask_extensions:
                mask_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    break

            if mask is None:
                raise ValueError(f"Failed to load mask for: {img_name}")

            # RESIZE mask too
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 128).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask'].unsqueeze(0)


class CAMODataset(COD10KDataset):
    def __init__(self, root_dir, split='train', img_size=352, augment=True):
        super().__init__(root_dir, split, img_size, augment)
        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'Images/Train')
            self.mask_dir = os.path.join(root_dir, 'GT/Train')
        else:
            self.image_dir = os.path.join(root_dir, 'Images/Test')
            self.mask_dir = os.path.join(root_dir, 'GT/Test')
        self.image_list = sorted(os.listdir(self.image_dir))


class NC4KDataset(COD10KDataset):
    def __init__(self, root_dir, split='test', img_size=352, augment=False):
        super().__init__(root_dir, split, img_size, augment)
        self.image_dir = os.path.join(root_dir, 'image')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.image_list = sorted(os.listdir(self.image_dir))