import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os

from src.models.efficientnet import EfficientNetB1
from src.data.dataset import FaceForensicsDataset
from src.data.augmentation import get_baseline_transforms, get_val_transforms
from src.training.base_trainer import BaseTrainer
from src.utils.device import get_device


class CurriculumTrainer(BaseTrainer):
    """
    Strategy 3: Curriculum Learning trainer.
    Progressively adds harder manipulation types as training progresses.
    """

    def __init__(self, curriculum_schedule, data_root, compression, **kwargs):
        super().__init__(**kwargs)
        self.curriculum_schedule = curriculum_schedule
        self.data_root = data_root
        self.compression = compression

    def _get_manipulations_for_epoch(self, epoch):
        """Return the list of manipulations to use at a given epoch."""
        active = []
        for phase, cfg in self.curriculum_schedule.items():
            start, end = cfg['epochs']
            if start <= epoch <= end:
                active = cfg['manipulations']
        return active

    def train(self, num_epochs):
        print(f"\n{'='*60}")
        print(f"Training {self.config['strategy_name']} (Curriculum Learning)")
        print(f"{'='*60}\n")

        import time
        start_time = time.time()

        for epoch in range(num_epochs):
            # Update dataset manipulations based on curriculum
            manipulations = self._get_manipulations_for_epoch(epoch)
            print(f"\nEpoch {epoch+1}/{num_epochs} — Manipulations: {manipulations}")
            print("-" * 40)

            # Rebuild train loader with current curriculum manipulations
            train_dataset = FaceForensicsDataset(
                data_root=self.data_root,
                split='train',
                compression=self.compression,
                transform=get_baseline_transforms(),
                manipulations=manipulations
            )
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=False
            )

            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)

            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if self.scheduler:
                self.scheduler.step(val_acc)

            self.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, is_best)

            if self.check_early_stop(val_acc):
                break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total Time: {total_time/3600:.2f} hours")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")


def main():
    with open(os.environ.get('CONFIG_DIR', 'configs/c40') + '/strategy3_curriculum.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(config['save_dir'], exist_ok=True)

    # Validation dataset (fixed, uses all manipulations)
    val_dataset = FaceForensicsDataset(
        data_root=config['data_root'],
        split='val',
        compression=config['compression'],
        transform=get_val_transforms()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )

    # Dummy train loader (will be rebuilt each epoch by CurriculumTrainer)
    train_loader = val_loader

    model = EfficientNetB1(num_classes=2, pretrained=True)
    model = model.to(device)
    print(f"Model parameters: {model.get_num_parameters()/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )

    trainer = CurriculumTrainer(
        curriculum_schedule=config['curriculum'],
        data_root=config['data_root'],
        compression=config['compression'],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )

    trainer.train(num_epochs=config['num_epochs'])

if __name__ == '__main__':
    main()
