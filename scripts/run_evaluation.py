"""
Evaluate a trained model checkpoint on a given compression level.

Usage:
    python scripts/run_evaluation.py \
        --checkpoint results/models/strategy1_baseline/best_model.pth \
        --compression c40 \
        --split test

    # Cross-compression: model trained on c40, evaluated on c23
    python scripts/run_evaluation.py \
        --checkpoint results/models/strategy1_baseline/best_model.pth \
        --compression c23 \
        --split test
"""
import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import FaceForensicsDataset
from src.data.augmentation import get_val_transforms
from src.models.efficientnet import EfficientNetB1
from src.models.multitask import MultiTaskEfficientNet
from src.evaluation.evaluate import evaluate_model, save_metrics
from src.utils.device import get_device


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    is_multitask = config.get('model', '') == 'multitask_efficientnet_b1'

    if is_multitask:
        model = MultiTaskEfficientNet(pretrained=False)
    else:
        model = EfficientNetB1(
            num_classes=config.get('num_classes', 2),
            pretrained=False,
            dropout=config.get('dropout', 0.3)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    train_compression = config.get('compression', 'unknown')
    strategy_name = config.get('strategy_name', 'unknown')
    return model, is_multitask, strategy_name, train_compression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model.pth')
    parser.add_argument('--compression', default='c40', choices=['c23', 'c40'],
                        help='Compression to evaluate on (default: c40)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--data_root', default='data/',
                        help='Path to FF++ data root (default: data/)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, is_multitask, strategy_name, train_compression = load_model(args.checkpoint, device)
    print(f"Strategy:         {strategy_name}")
    print(f"Trained on:       {train_compression}")
    print(f"Evaluating on:    {args.compression} ({args.split} split)")

    dataset = FaceForensicsDataset(
        data_root=args.data_root,
        split=args.split,
        compression=args.compression,
        transform=get_val_transforms()
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=False)
    print(f"Samples: {len(dataset)}\n")

    metrics = evaluate_model(model, loader, device, is_multitask=is_multitask)
    metrics['strategy_name'] = strategy_name
    metrics['trained_on'] = train_compression
    metrics['evaluated_on'] = args.compression
    metrics['split'] = args.split

    # Save JSON alongside the checkpoint
    save_dir = os.path.dirname(args.checkpoint)
    filename = f"eval_{args.split}_{args.compression}.json"
    save_metrics(metrics, os.path.join(save_dir, filename))

    # Append a row to results/logs/evaluation_results.csv
    log_dir = 'results/logs'
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'evaluation_results.csv')
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['strategy', 'trained_on', 'evaluated_on', 'split',
                             'accuracy', 'precision', 'recall', 'f1', 'auc',
                             'false_negative_rate'])
        writer.writerow([
            strategy_name, train_compression, args.compression, args.split,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['auc']:.4f}",
            f"{metrics['false_negative_rate']:.4f}",
        ])
    print(f"Row appended to {csv_path}")


if __name__ == '__main__':
    main()
