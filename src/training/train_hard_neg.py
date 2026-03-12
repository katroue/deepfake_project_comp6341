"""
Strategy 5: Hard Negative Mining

Trains with standard SGD for warmup, then identifies low-confidence samples
(hard negatives) and oversamples them in subsequent epochs.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml
import os
import numpy as np
from tqdm import tqdm

from src.models.efficientnet import EfficientNetB1
from src.data.dataset import FaceForensicsDataset
from src.data.augmentation import get_baseline_transforms, get_val_transforms
from src.training.base_trainer import BaseTrainer
from src.utils.device import get_device


def find_hard_negatives(model, dataset, device, config_hn):
    """Compute per-sample confidence and return indices of hard samples."""
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        num_workers=2, pin_memory=False)
    confidences = []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc='Mining hard negatives'):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            # Confidence = probability of the predicted class
            conf, _ = probs.max(dim=1)
            confidences.extend(conf.cpu().numpy().tolist())

    confidences = np.array(confidences)
    return confidences


def build_weighted_sampler(confidences, hard_sample_ratio, threshold):
    """
    Assign higher sampling weight to hard (low-confidence) samples.
    """
    weights = np.ones(len(confidences))
    hard_mask = confidences < threshold
    # Hard samples get upweighted, easy samples get downweighted
    hard_weight = hard_sample_ratio / (hard_mask.sum() + 1e-8) * len(confidences)
    easy_weight = (1 - hard_sample_ratio) / ((~hard_mask).sum() + 1e-8) * len(confidences)
    weights[hard_mask] = hard_weight
    weights[~hard_mask] = easy_weight
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(confidences),
        replacement=True
    )
    return sampler


def main():
    with open(os.environ.get('CONFIG_DIR', 'configs/c40') + '/strategy5_hard_neg.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config.get('hard_negatives_dir', config['save_dir']), exist_ok=True)

    hn_cfg = config['hard_negative_mining']
    start_epoch = hn_cfg['start_epoch']
    mining_freq = hn_cfg['mining_frequency']
    base_threshold = hn_cfg['confidence_threshold']
    hard_ratio = hn_cfg['hard_sample_ratio']
    threshold_schedule = {int(k.split('_')[1]): v
                          for k, v in hn_cfg.get('threshold_schedule', {}).items()}

    train_dataset = FaceForensicsDataset(
        data_root=config['data_root'],
        split='train',
        compression=config['compression'],
        transform=get_baseline_transforms()
    )
    val_dataset = FaceForensicsDataset(
        data_root=config['data_root'],
        split='val',
        compression=config['compression'],
        transform=get_val_transforms()
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'], pin_memory=False)

    model = EfficientNetB1(num_classes=2, pretrained=True)
    model = model.to(device)
    print(f"Model parameters: {model.get_num_parameters()/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )

    # Initial loader (uniform sampling)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'], pin_memory=False)

    trainer = BaseTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, config=config
    )

    print(f"\n{'='*60}")
    print(f"Training {config['strategy_name']} (Hard Negative Mining)")
    print(f"{'='*60}\n")

    import time
    start_time = time.time()

    for epoch in range(config['num_epochs']):
        # Update threshold from schedule
        threshold = threshold_schedule.get(epoch + 1, base_threshold)

        # Re-mine hard negatives at specified frequency
        if hn_cfg['enabled'] and epoch >= start_epoch and (epoch - start_epoch) % mining_freq == 0:
            print(f"\nMining hard negatives at epoch {epoch+1} (threshold={threshold:.2f})...")
            confidences = find_hard_negatives(model, train_dataset, device, hn_cfg)
            n_hard = (confidences < threshold).sum()
            print(f"Found {n_hard}/{len(train_dataset)} hard samples ({100*n_hard/len(train_dataset):.1f}%)")

            sampler = build_weighted_sampler(confidences, hard_ratio, threshold)
            trainer.train_loader = DataLoader(
                train_dataset, batch_size=config['batch_size'],
                sampler=sampler, num_workers=config['num_workers'], pin_memory=False
            )

        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)

        train_loss, train_acc = trainer.train_epoch()
        trainer.train_losses.append(train_loss)

        val_loss, val_acc = trainer.validate()
        trainer.val_losses.append(val_loss)
        trainer.val_accuracies.append(val_acc)

        scheduler.step(val_acc)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        is_best = val_acc > trainer.best_val_acc
        if is_best:
            trainer.best_val_acc = val_acc
        trainer.save_checkpoint(epoch, is_best)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Best Val Acc: {trainer.best_val_acc:.2f}%")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
