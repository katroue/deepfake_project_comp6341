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

def main():
    # Load config
    with open(os.environ.get('CONFIG_DIR', 'configs/c40') + '/strategy1_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Datasets
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
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False  # Not needed for MPS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model = EfficientNetB1(num_classes=2, pretrained=True)
    model = model.to(device)
    
    print(f"Model parameters: {model.get_num_parameters()/1e6:.2f}M")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
    )
    
    # Trainer
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])

if __name__ == '__main__':
    main()
    