@echo off
REM Run MoE training pipeline - stops on any error

echo ============================================================
echo STEP 1: Router-only training (50 epochs)
echo ============================================================
python load_pretrained_experts.py --sinet-checkpoint ./checkpoints_sinet/best_model.pth --pranet-checkpoint ./checkpoints_pranet/best_model.pth --fspnet-checkpoint ./checkpoints_fspnet/best_model.pth --expert-types sinet pranet fspnet --mode router-only --data-root ./combined_dataset --backbone pvt_v2_b2 --epochs 50 --batch-size 16 --img-size 448 --lr 1e-4 --output-dir ./checkpoints_moe_router --val-freq 1
if %errorlevel% neq 0 (
    echo ERROR: Step 1 failed!
    pause
    exit /b 1
)

echo ============================================================
echo STEP 2: Evaluate MoE on CAMO
echo ============================================================
python evaluate_moe.py --checkpoint ./checkpoints_moe_router/best_model.pth --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --find-best-threshold
if %errorlevel% neq 0 (
    echo WARNING: CAMO evaluation failed, continuing...
)

echo ============================================================
echo STEP 3: Evaluate MoE on COD10K
echo ============================================================
python evaluate_moe.py --checkpoint ./checkpoints_moe_router/best_model.pth --image-dir "./COD10K-v3/Test/Image" --gt-dir "./COD10K-v3/Test/GT_Object" --find-best-threshold
if %errorlevel% neq 0 (
    echo WARNING: COD10K evaluation failed, continuing...
)

echo ============================================================
echo STEP 4: Full fine-tuning (100 epochs)
echo ============================================================
python load_pretrained_experts.py --sinet-checkpoint ./checkpoints_sinet/best_model.pth --pranet-checkpoint ./checkpoints_pranet/best_model.pth --fspnet-checkpoint ./checkpoints_fspnet/best_model.pth --expert-types sinet pranet fspnet --mode full --data-root ./combined_dataset --backbone pvt_v2_b2 --epochs 100 --batch-size 16 --img-size 448 --lr 1e-5 --output-dir ./checkpoints_moe_finetuned --val-freq 1
if %errorlevel% neq 0 (
    echo ERROR: Step 4 failed!
    pause
    exit /b 1
)

echo ============================================================
echo STEP 5: Final Evaluation on CAMO
echo ============================================================
python evaluate_moe.py --checkpoint ./checkpoints_moe_finetuned/best_model.pth --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --find-best-threshold

echo ============================================================
echo STEP 6: Final Evaluation on COD10K
echo ============================================================
python evaluate_moe.py --checkpoint ./checkpoints_moe_finetuned/best_model.pth --image-dir "./COD10K-v3/Test/Image" --gt-dir "./COD10K-v3/Test/GT_Object" --find-best-threshold

echo ============================================================
echo ALL DONE!
echo ============================================================
pause
