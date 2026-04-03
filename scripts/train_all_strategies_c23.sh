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

echo "Training Strategy 4: Self-Supervised (SimSiam)"
python -m src.training.train_ssl

echo "Training Strategy 5: Hard Negative Mining"
python -m src.training.train_hard_neg

echo "Training Strategy 6: Multi-Task"
python -m src.training.train_multitask

echo "========================================="
echo "All c23 strategies trained!"
echo "========================================="

# To rerun self-supervised strategy (SimSiam), run:
# CONFIG_DIR=configs/c23 python -m src.training.train_ssl
# To resume Phase 1 from a saved checkpoint:
# CONFIG_DIR=configs/c23 python -m src.training.train_ssl --resume-phase1 results/models/c23/strategy4_ssl/phase1_pretrained/phase1_resume.pth
# To skip Phase 1 and resume Phase 2:
# CONFIG_DIR=configs/c23 python -m src.training.train_ssl --resume results/models/c23/strategy4_ssl/last_model.pth