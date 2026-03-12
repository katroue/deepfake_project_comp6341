#!/bin/bash
# Evaluate all 6 trained strategies on both c40 and c23 (cross-compression).
# Run after training is complete.

CHECKPOINTS=(
    "results/models/strategy1_baseline/best_model.pth"
    "results/models/strategy2_augmented/best_model.pth"
    "results/models/strategy3_curriculum/best_model.pth"
    "results/models/strategy4_ssl/best_model.pth"
    "results/models/strategy5_hard_neg/best_model.pth"
    "results/models/strategy6_multitask/best_model.pth"
)

COMPRESSIONS=("c40" "c23")

echo "========================================="
echo "Evaluating All Strategies"
echo "========================================="

for ckpt in "${CHECKPOINTS[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "Skipping $ckpt (not found)"
        continue
    fi

    for compression in "${COMPRESSIONS[@]}"; do
        echo ""
        echo "-----------------------------------------"
        echo "Checkpoint:  $ckpt"
        echo "Compression: $compression"
        echo "-----------------------------------------"
        python scripts/run_evaluation.py \
            --checkpoint "$ckpt" \
            --compression "$compression" \
            --split test
    done
done

echo ""
echo "========================================="
echo "All evaluations complete."
echo "Results saved in each strategy's model dir."
echo "========================================="
