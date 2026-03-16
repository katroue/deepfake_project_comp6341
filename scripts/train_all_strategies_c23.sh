#!/bin/bash
# Train all 6 strategies on c23 compression sequentially

export CONFIG_DIR="configs/c23"

echo "========================================="
echo "Training All Strategies (c23)"
echo "========================================="

echo "Training Strategy 1: Baseline"
python -m src.training.train_baseline

echo "Training Strategy 2: Heavy Augmentation"
python -m src.training.train_augmented

echo "Training Strategy 3: Curriculum Learning"
python -m src.training.train_curriculum

echo "Training Strategy 4: Self-Supervised"
python -m src.training.train_ssl

echo "Training Strategy 5: Hard Negative Mining"
python -m src.training.train_hard_neg

echo "Training Strategy 6: Multi-Task"
python -m src.training.train_multitask

echo "========================================="
echo "All c23 strategies trained!"
echo "========================================="

# To resume training baseline strategy, run:
CONFIG_DIR=configs/c23 python -m src.training.train_baseline \
    --resume results/models/c23/strategy1_baseline/last_model.pth