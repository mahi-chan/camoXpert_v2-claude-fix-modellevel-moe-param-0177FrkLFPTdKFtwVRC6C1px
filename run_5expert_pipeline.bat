@echo off
REM 5-Expert MoE Training Pipeline
REM Trains diverse experts then combines with MoE for SOTA performance

echo ============================================================
echo 5-EXPERT MOE TRAINING PIPELINE
echo ============================================================
echo Experts: FSPNet, PraNet, BASNet, CPD, GCPANet
echo Goal: Achieve 0.85-0.90 S-measure through diverse experts
echo ============================================================

REM ============================================================
REM PHASE 1: TRAIN INDIVIDUAL EXPERTS (if not trained)
REM ============================================================

echo.
echo ============================================================
echo STEP 1A: Train BASNet Expert (Boundary-Aware)
echo ============================================================
if exist ".\checkpoints_basnet\best_model.pth" (
    echo BASNet already trained, skipping...
) else (
    python train_single_expert.py --expert-type basnet --data-root ./combined_dataset --backbone pvt_v2_b2 --epochs 100 --batch-size 16 --img-size 448 --lr 1e-4 --output-dir ./checkpoints_basnet --val-freq 5 --iou-weight 2.0 --dice-weight 2.0 --pos-weight 5.0 --use-progressive-aug --multi-scale
    if %errorlevel% neq 0 (
        echo ERROR: BASNet training failed!
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo STEP 1B: Train CPD Expert (Multi-Scale)
echo ============================================================
if exist ".\checkpoints_cpd\best_model.pth" (
    echo CPD already trained, skipping...
) else (
    python train_single_expert.py --expert-type cpd --data-root ./combined_dataset --backbone pvt_v2_b2 --epochs 100 --batch-size 16 --img-size 448 --lr 1e-4 --output-dir ./checkpoints_cpd --val-freq 5 --iou-weight 2.0 --dice-weight 2.0 --pos-weight 5.0 --use-progressive-aug --multi-scale
    if %errorlevel% neq 0 (
        echo ERROR: CPD training failed!
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo STEP 1C: Train GCPANet Expert (Global Context)
echo ============================================================
if exist ".\checkpoints_gcpanet\best_model.pth" (
    echo GCPANet already trained, skipping...
) else (
    python train_single_expert.py --expert-type gcpanet --data-root ./combined_dataset --backbone pvt_v2_b2 --epochs 100 --batch-size 16 --img-size 448 --lr 1e-4 --output-dir ./checkpoints_gcpanet --val-freq 5 --iou-weight 2.0 --dice-weight 2.0 --pos-weight 5.0 --use-progressive-aug --multi-scale
    if %errorlevel% neq 0 (
        echo ERROR: GCPANet training failed!
        pause
        exit /b 1
    )
)

REM ============================================================
REM PHASE 2: EVALUATE INDIVIDUAL EXPERTS
REM ============================================================

echo.
echo ============================================================
echo STEP 2: Evaluate All Experts on CAMO
echo ============================================================
for %%e in (fspnet pranet basnet cpd gcpanet) do (
    echo Evaluating %%e on CAMO...
    python evaluate_single_expert.py --checkpoint ./checkpoints_%%e/best_model.pth --expert-type %%e --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --find-best-threshold
)

REM ============================================================
REM PHASE 3: TRAIN 5-EXPERT MOE
REM ============================================================

echo.
echo ============================================================
echo STEP 3A: MoE Router-Only Training (50 epochs)
echo ============================================================
python load_pretrained_experts.py --fspnet-checkpoint ./checkpoints_fspnet/best_model.pth --pranet-checkpoint ./checkpoints_pranet/best_model.pth --basnet-checkpoint ./checkpoints_basnet/best_model.pth --cpd-checkpoint ./checkpoints_cpd/best_model.pth --gcpanet-checkpoint ./checkpoints_gcpanet/best_model.pth --expert-types fspnet pranet basnet cpd gcpanet --mode router-only --data-root ./combined_dataset --backbone pvt_v2_b2 --epochs 50 --batch-size 16 --img-size 448 --lr 1e-4 --output-dir ./checkpoints_moe5_router --val-freq 1
if %errorlevel% neq 0 (
    echo ERROR: MoE router training failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo STEP 3B: MoE Full Fine-Tuning (100 epochs)
echo ============================================================
python load_pretrained_experts.py --fspnet-checkpoint ./checkpoints_moe5_router/best_model.pth --expert-types fspnet pranet basnet cpd gcpanet --mode full --data-root ./combined_dataset --backbone pvt_v2_b2 --epochs 100 --batch-size 16 --img-size 448 --lr 5e-6 --output-dir ./checkpoints_moe5_finetuned --val-freq 1
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
echo STEP 4A: Final Evaluation on CAMO
echo ============================================================
python evaluate_moe.py --checkpoint ./checkpoints_moe5_finetuned/best_model.pth --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --find-best-threshold

echo.
echo ============================================================
echo STEP 4B: Final Evaluation on COD10K
echo ============================================================
python evaluate_moe.py --checkpoint ./checkpoints_moe5_finetuned/best_model.pth --image-dir "./COD10K-v3/Test/Image" --gt-dir "./COD10K-v3/Test/GT_Object" --find-best-threshold

echo.
echo ============================================================
echo STEP 5: Run Ablation Study
echo ============================================================
python expert_ablation.py --checkpoint ./checkpoints_moe5_finetuned/best_model.pth --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --output-dir ./ablation_moe5_camo --expert-types fspnet pranet basnet cpd gcpanet

echo.
echo ============================================================
echo ALL DONE!
echo ============================================================
echo Check ablation_moe5_camo for detailed results
echo ============================================================
pause
