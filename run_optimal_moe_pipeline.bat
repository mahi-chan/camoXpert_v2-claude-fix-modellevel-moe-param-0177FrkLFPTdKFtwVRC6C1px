@echo off
REM ============================================================
REM OPTIMAL 3-EXPERT TRAINING WITH SPECIALIZED SETTINGS
REM ============================================================
REM Each expert is trained with different settings to specialize:
REM - FSPNet: Standard (already specializes in frequency)
REM - ZoomNeXt: Larger images, focus on scale variation
REM - PraNet: Higher boundary weights for edge specialization
REM ============================================================

echo ============================================================
echo OPTIMAL 3-EXPERT TRAINING PIPELINE
echo ============================================================
echo Expert 1: FSPNet  - Texture/Frequency specialist
echo Expert 2: ZoomNeXt - Multi-scale specialist (TPAMI 2024)
echo Expert 3: PraNet  - Boundary/Edge specialist
echo ============================================================

REM ============================================================
REM EXPERT 1: FSPNet (Frequency/Texture Expert)
REM Settings: Standard - FSPNet architecture already specializes in frequency
REM ============================================================

echo.
echo ============================================================
echo TRAINING FSPNet: Texture/Frequency Expert
echo Settings: img=448, batch=16, iou=2.0, dice=2.0, pos=5.0
echo ============================================================
if exist ".\checkpoints_fspnet\best_model.pth" (
    echo FSPNet already trained, skipping...
) else (
    python train_single_expert.py ^
        --expert fspnet ^
        --data-root ./combined_dataset ^
        --backbone pvt_v2_b2 ^
        --epochs 100 ^
        --batch-size 16 ^
        --img-size 448 ^
        --lr 1e-4 ^
        --checkpoint-dir ./checkpoints_fspnet ^
        --val-freq 5 ^
        --iou-weight 2.0 ^
        --dice-weight 2.0 ^
        --pos-weight 5.0 ^
        --enable-progressive-aug ^
        --use-multi-scale
    if %errorlevel% neq 0 (
        echo ERROR: FSPNet training failed!
        pause
        exit /b 1
    )
)

REM ============================================================
REM EXPERT 2: ZoomNeXt (Multi-Scale Expert) - TPAMI 2024
REM Settings: Larger image (512), higher IoU for accurate segmentation
REM ============================================================

echo.
echo ============================================================
echo TRAINING ZoomNeXt: Multi-Scale Expert (TPAMI 2024)
echo Settings: img=512, batch=12, iou=3.0, dice=2.0, pos=3.0
echo ============================================================
if exist ".\checkpoints_zoomnext\best_model.pth" (
    echo ZoomNeXt already trained, skipping...
) else (
    python train_single_expert.py ^
        --expert zoomnext ^
        --data-root ./combined_dataset ^
        --backbone pvt_v2_b2 ^
        --epochs 100 ^
        --batch-size 12 ^
        --img-size 512 ^
        --lr 1e-4 ^
        --checkpoint-dir ./checkpoints_zoomnext ^
        --val-freq 5 ^
        --iou-weight 3.0 ^
        --dice-weight 2.0 ^
        --pos-weight 3.0 ^
        --enable-progressive-aug ^
        --use-multi-scale
    if %errorlevel% neq 0 (
        echo ERROR: ZoomNeXt training failed!
        pause
        exit /b 1
    )
)

REM ============================================================
REM EXPERT 3: PraNet (Boundary Expert)
REM Settings: Higher Dice and pos_weight for boundary emphasis
REM ============================================================

echo.
echo ============================================================
echo TRAINING PraNet: Boundary/Edge Expert
echo Settings: img=448, batch=16, iou=2.0, dice=3.0, pos=7.0
echo ============================================================
if exist ".\checkpoints_pranet\best_model.pth" (
    echo PraNet already trained, skipping...
) else (
    python train_single_expert.py ^
        --expert pranet ^
        --data-root ./combined_dataset ^
        --backbone pvt_v2_b2 ^
        --epochs 100 ^
        --batch-size 16 ^
        --img-size 448 ^
        --lr 1e-4 ^
        --checkpoint-dir ./checkpoints_pranet ^
        --val-freq 5 ^
        --iou-weight 2.0 ^
        --dice-weight 3.0 ^
        --pos-weight 7.0 ^
        --enable-progressive-aug ^
        --use-multi-scale
    if %errorlevel% neq 0 (
        echo ERROR: PraNet training failed!
        pause
        exit /b 1
    )
)

REM ============================================================
REM PHASE 2: EVALUATE INDIVIDUAL EXPERTS
REM ============================================================

echo.
echo ============================================================
echo EVALUATING ALL EXPERTS ON CAMO DATASET
echo ============================================================

echo.
echo --- Evaluating FSPNet ---
python evaluate_single_expert.py --checkpoint ./checkpoints_fspnet/best_model.pth --expert fspnet --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --find-best-threshold

echo.
echo --- Evaluating ZoomNeXt ---
python evaluate_single_expert.py --checkpoint ./checkpoints_zoomnext/best_model.pth --expert zoomnext --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --find-best-threshold

echo.
echo --- Evaluating PraNet ---
python evaluate_single_expert.py --checkpoint ./checkpoints_pranet/best_model.pth --expert pranet --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --find-best-threshold

REM ============================================================
REM PHASE 3: TRAIN 3-EXPERT MoE
REM ============================================================

echo.
echo ============================================================
echo TRAINING MoE ROUTER (50 epochs, experts frozen)
echo ============================================================
python load_pretrained_experts.py ^
    --fspnet-checkpoint ./checkpoints_fspnet/best_model.pth ^
    --zoomnext-checkpoint ./checkpoints_zoomnext/best_model.pth ^
    --pranet-checkpoint ./checkpoints_pranet/best_model.pth ^
    --expert-types fspnet zoomnext pranet ^
    --mode router-only ^
    --data-root ./combined_dataset ^
    --backbone pvt_v2_b2 ^
    --epochs 50 ^
    --batch-size 16 ^
    --img-size 448 ^
    --lr 1e-4 ^
    --output-dir ./checkpoints_moe_router ^
    --val-freq 1
if %errorlevel% neq 0 (
    echo ERROR: MoE router training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo FINE-TUNING MoE (100 epochs, everything unfrozen)
echo ============================================================
python load_pretrained_experts.py ^
    --fspnet-checkpoint ./checkpoints_moe_router/best_model.pth ^
    --expert-types fspnet zoomnext pranet ^
    --mode full ^
    --data-root ./combined_dataset ^
    --backbone pvt_v2_b2 ^
    --epochs 100 ^
    --batch-size 16 ^
    --img-size 448 ^
    --lr 5e-6 ^
    --output-dir ./checkpoints_moe_finetuned ^
    --val-freq 1
if %errorlevel% neq 0 (
    echo ERROR: MoE fine-tuning failed!
    pause
    exit /b 1
)

REM ============================================================
REM PHASE 4: FINAL EVALUATION
REM ============================================================

echo.
echo ============================================================
echo FINAL EVALUATION ON CAMO
echo ============================================================
python evaluate_moe.py --checkpoint ./checkpoints_moe_finetuned/best_model.pth --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --find-best-threshold

echo.
echo ============================================================
echo FINAL EVALUATION ON COD10K
echo ============================================================
python evaluate_moe.py --checkpoint ./checkpoints_moe_finetuned/best_model.pth --image-dir "./COD10K-v3/Test/Image" --gt-dir "./COD10K-v3/Test/GT_Object" --find-best-threshold

REM ============================================================
REM PHASE 5: ABLATION STUDY
REM ============================================================

echo.
echo ============================================================
echo RUNNING ABLATION STUDY
echo ============================================================
python expert_ablation.py --checkpoint ./checkpoints_moe_finetuned/best_model.pth --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --output-dir ./ablation_final

echo.
echo ============================================================
echo                    ALL DONE!
echo ============================================================
echo Results saved to:
echo   - Individual experts: checkpoints_fspnet/, checkpoints_zoomnext/, checkpoints_pranet/
echo   - MoE Router: checkpoints_moe_router/
echo   - MoE Fine-tuned: checkpoints_moe_finetuned/
echo   - Ablation: ablation_final/
echo ============================================================
pause
