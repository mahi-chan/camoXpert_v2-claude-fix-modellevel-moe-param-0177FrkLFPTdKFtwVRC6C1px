# CamoXpert Training on Google Colab
# =====================================
# Run each cell in order

# %% [markdown]
# # 🦎 CamoXpert Training Setup
# 
# This notebook runs CamoXpert MoE training on Colab's free T4 GPU.

# %% Cell 1: Check GPU
!nvidia-smi

# %% Cell 2: Clone your repository
# Option A: From GitHub (if you pushed your code)
# !git clone https://github.com/YOUR_USERNAME/camoXpert_v2.git
# %cd camoXpert_v2

# Option B: From Google Drive (recommended)
from google.colab import drive
drive.mount('/content/drive')

# Copy your code from Drive (adjust path as needed)
!cp -r "/content/drive/MyDrive/camoXpert_v2-claude-fix-modellevel-moe-param-0177FrkLFPTdKFtwVRC6C1px" /content/camoXpert
%cd /content/camoXpert

# %% Cell 3: Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q timm albumentations opencv-python-headless tqdm

# %% Cell 4: Download COD10K dataset
# Option A: From your Drive
!cp -r "/content/drive/MyDrive/datasets/COD10K-v3" /content/COD10K-v3

# Option B: Download directly (if you have a link)
# !gdown --id YOUR_DRIVE_FILE_ID -O cod10k.zip
# !unzip -q cod10k.zip -d /content/

# %% Cell 5: Verify dataset structure
!ls /content/COD10K-v3/
!ls /content/COD10K-v3/Train/
!ls /content/COD10K-v3/Test/

# %% Cell 6: Training (Single GPU - no DDP)
# Key differences from Kaggle:
# - No --use-ddp (single GPU)
# - Smaller batch size (T4 has 16GB VRAM)
# - Can use --use-ema now (no DDP conflict!)

!python train_advanced.py \
    --data-root /content/COD10K-v3 \
    --batch-size 8 \
    --img-size 448 \
    --epochs 200 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --accumulation-steps 4 \
    --warmup-epochs 10 \
    --aug-transition-epoch 0 \
    --aug-max-strength 0.8 \
    --loss-type sota \
    --router-warmup-epochs 20 \
    --use-ema \
    --ema-decay 0.999 \
    --val-freq 2 \
    --save-interval 10 \
    --checkpoint-dir /content/drive/MyDrive/checkpoints_colab

# %% Cell 7: Resume from Kaggle checkpoint (if you have one)
# First, upload your Kaggle checkpoint to Drive, then:
# 
# !python train_advanced.py \
#     --data-root /content/COD10K-v3 \
#     --resume-from "/content/drive/MyDrive/kaggle_checkpoints/best_model.pth" \
#     --load-weights-only \
#     --batch-size 8 \
#     --img-size 448 \
#     --epochs 200 \
#     ... (other args)

# %% Cell 8: Evaluate on all benchmarks
# Make sure you have CAMO, CHAMELEON, NC4K datasets
!python evaluate.py \
    --checkpoint /content/drive/MyDrive/checkpoints_colab/best_model.pth \
    --data-root /content/ \
    --datasets COD10K CAMO CHAMELEON NC4K \
    --use-tta \
    --output-dir /content/drive/MyDrive/eval_results

# %% Cell 9: Save final results to Drive
!cp -r /content/camoXpert/checkpoints_colab/* "/content/drive/MyDrive/checkpoints_colab/"
