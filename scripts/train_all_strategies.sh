#!/bin/bash

# Train all 6 strategies sequentially

echo "========================================="
echo "Training All Strategies"
echo "========================================="

# Strategy 1: Baseline
echo "Training Strategy 1: Baseline"
python -m src.training.train_baseline

# Strategy 2: Heavy Augmentation
echo "Training Strategy 2: Heavy Augmentation"
python -m src.training.train_augmented

# Strategy 3: Curriculum Learning
echo "Training Strategy 3: Curriculum Learning"
python -m src.training.train_curriculum

# Strategy 4: Self-Supervised Learning
echo "Training Strategy 4: Self-Supervised"
python -m src.training.train_ssl

# Strategy 5: Hard Negative Mining
echo "Training Strategy 5: Hard Negative Mining"
python -m src.training.train_hard_neg

# Strategy 6: Multi-Task Learning
echo "Training Strategy 6: Multi-Task"
python -m src.training.train_multitask

echo "========================================="
echo "All strategies trained!"
echo "========================================="