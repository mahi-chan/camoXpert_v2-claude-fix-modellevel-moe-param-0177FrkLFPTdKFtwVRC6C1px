@echo off
REM Run MoE training pipeline - runs all steps sequentially

echo ============================================================
echo STEP 1: Router-only training (30 epochs)
echo ============================================================
python load_pretrained_experts.py --sinet-checkpoint ./checkpoints_sinet/best_model.pth --pranet-checkpoint ./checkpoints_pranet/best_model.pth --fspnet-checkpoint ./checkpoints_fspnet/best_model.pth --expert-types sinet pranet fspnet --mode router-only --data-root ./combined_dataset --backbone pvt_v2_b2 --epochs 30 --batch-size 16 --img-size 448 --lr 1e-4 --output-dir ./checkpoints_moe_router --val-freq 5

echo ============================================================
echo STEP 2: Evaluate MoE on CAMO
echo ============================================================
python evaluate.py --checkpoint ./checkpoints_moe_router/best_model.pth --data-root . --datasets CAMO --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --img-size 448

echo ============================================================
echo STEP 3: Evaluate MoE on COD10K
echo ============================================================
python evaluate.py --checkpoint ./checkpoints_moe_router/best_model.pth --data-root . --datasets COD10K --image-dir "./COD10K-v3/Test/Image" --gt-dir "./COD10K-v3/Test/GT_Object" --img-size 448

echo ============================================================
echo STEP 4: Full fine-tuning (50 epochs) - Optional
echo ============================================================
python load_pretrained_experts.py --sinet-checkpoint ./checkpoints_moe_router/best_model.pth --pranet-checkpoint ./checkpoints_moe_router/best_model.pth --fspnet-checkpoint ./checkpoints_moe_router/best_model.pth --expert-types sinet pranet fspnet --mode full --data-root ./combined_dataset --backbone pvt_v2_b2 --epochs 50 --batch-size 16 --img-size 448 --lr 1e-5 --output-dir ./checkpoints_moe_finetuned --val-freq 5

echo ============================================================
echo STEP 5: Final Evaluation on CAMO
echo ============================================================
python evaluate.py --checkpoint ./checkpoints_moe_finetuned/best_model.pth --data-root . --datasets CAMO --image-dir "./CAMO-V.1.0-CVIU2019/Images/Test" --gt-dir "./CAMO-V.1.0-CVIU2019/GT" --img-size 448

echo ============================================================
echo STEP 6: Final Evaluation on COD10K
echo ============================================================
python evaluate.py --checkpoint ./checkpoints_moe_finetuned/best_model.pth --data-root . --datasets COD10K --image-dir "./COD10K-v3/Test/Image" --gt-dir "./COD10K-v3/Test/GT_Object" --img-size 448

echo ============================================================
echo ALL DONE!
echo ============================================================
pause
