#!/bin/bash
# Evaluate all trained models (c40 and c23) on both compressions.
# Run after both train_all_strategies.sh and train_all_strategies_c23.sh complete.

STRATEGIES=(
    "strategy1_baseline"
    "strategy2_augmented"
    "strategy3_curriculum"
    "strategy4_ssl"
    "strategy5_hard_neg"
    "strategy6_multitask"
)

TRAIN_COMPRESSIONS=("c40") # add c23 when is it done training
EVAL_COMPRESSIONS=("c40" "c23")

echo "========================================="
echo "Evaluating All Strategies"
echo "========================================="

for train_c in "${TRAIN_COMPRESSIONS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        ckpt="results/models/${train_c}/${strategy}/best_model.pth"

        if [ ! -f "$ckpt" ]; then
            echo "Skipping $ckpt (not found)"
            continue
        fi

        for eval_c in "${EVAL_COMPRESSIONS[@]}"; do
            echo ""
            echo "-----------------------------------------"
            echo "Model:       $ckpt"
            echo "Eval on:     $eval_c"
            echo "-----------------------------------------"
            python scripts/run_evaluation.py \
                --checkpoint "$ckpt" \
                --compression "$eval_c" \
                --split test
        done
    done
done

echo ""
echo "========================================="
echo "All evaluations complete."
echo "Summary table: results/logs/evaluation_results.csv"
echo "========================================="
