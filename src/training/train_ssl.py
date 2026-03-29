"""
Strategy 4: Self-Supervised Learning (SimCLR)

Phase 1: Contrastive pretraining on all face images (real + fake, no labels)
Phase 2: Supervised fine-tuning on FF++ with pretrained backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import os
import argparse
from PIL import Image
from tqdm import tqdm

from src.models.efficientnet import EfficientNetB1
from src.data.dataset import FaceForensicsDataset
from src.data.augmentation import get_ssl_transforms, get_baseline_transforms, get_val_transforms
from src.training.base_trainer import BaseTrainer
from src.utils.device import get_device
import timm


class ContrastivePairDataset(Dataset):
    """Wraps FaceForensicsDataset to return two augmented views per image."""

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_path, label, manip_type = self.base_dataset.samples[idx]
        image = Image.open(img_path).convert('RGB')
        view1 = self.transform(image)
        view2 = self.transform(image)
        return view1, view2


class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)

        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)

        loss = F.cross_entropy(sim, targets)
        return loss


def pretrain_simclr(config, device):
    """Phase 1: SimCLR contrastive pretraining."""
    phase1 = config['phase_1']
    if not phase1.get('enabled', True):
        print("Phase 1 (SSL pretraining) disabled, skipping.")
        return None

    print("\n" + "="*60)
    print("Phase 1: SimCLR Contrastive Pretraining")
    print("="*60)

    save_dir = os.path.join(config['save_dir'], 'phase1_pretrained')
    os.makedirs(save_dir, exist_ok=True)

    # Build dataset of all face images (labels ignored)
    base_dataset = FaceForensicsDataset(
        data_root=config['data_root'],
        split='train',
        compression=config['compression'],
        transform=None  # transforms applied in ContrastivePairDataset
    )
    ssl_transform = get_ssl_transforms()
    pair_dataset = ContrastivePairDataset(base_dataset, ssl_transform)
    loader = DataLoader(
        pair_dataset,
        batch_size=phase1['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=False,
        drop_last=True
    )

    # Backbone (no classifier)
    backbone = timm.create_model('efficientnet_b1', pretrained=False, num_classes=0, global_pool='avg')
    feat_dim = backbone.num_features
    projection_head = SimCLRProjectionHead(feat_dim, proj_dim=phase1['projection_dim'])

    backbone = backbone.to(device)
    projection_head = projection_head.to(device)

    optimizer = optim.Adam(
        list(backbone.parameters()) + list(projection_head.parameters()),
        lr=phase1['learning_rate']
    )
    criterion = NTXentLoss(temperature=phase1['temperature'])

    for epoch in range(phase1['num_epochs']):
        backbone.train()
        projection_head.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"SSL Epoch {epoch+1}/{phase1['num_epochs']}")
        for view1, view2 in pbar:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()
            z1 = projection_head(backbone(view1))
            z2 = projection_head(backbone(view2))
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(loader)
        print(f"SSL Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Save checkpoint after every epoch so work survives session timeout
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        torch.save({'backbone_state_dict': backbone.state_dict()}, checkpoint_path)
        if os.path.exists('/kaggle/output'):
            import shutil
            shutil.copytree(save_dir, '/kaggle/output/results/models/c23/strategy4_ssl/phase1_pretrained', dirs_exist_ok=True)

    print(f"Saved SSL pretrained backbone to {checkpoint_path}")
    return checkpoint_path


def finetune(config, pretrained_path, device, resume_checkpoint=None):
    """Phase 2: Supervised fine-tuning with pretrained backbone."""
    phase2 = config['phase_2']
    if not phase2.get('enabled', True):
        print("Phase 2 (fine-tuning) disabled, skipping.")
        return

    print("\n" + "="*60)
    print("Phase 2: Supervised Fine-Tuning")
    print("="*60)

    os.makedirs(config['save_dir'], exist_ok=True)

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
    train_loader = DataLoader(train_dataset, batch_size=phase2['batch_size'],
                              shuffle=True, num_workers=phase2['num_workers'], pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=phase2['batch_size'],
                            shuffle=False, num_workers=phase2['num_workers'], pin_memory=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    model = EfficientNetB1(num_classes=2, pretrained=False)

    # Load pretrained backbone weights if available
    if phase2.get('load_pretrained', True) and pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        backbone_state = checkpoint['backbone_state_dict']
        # Load matching keys into model
        model_state = model.model.state_dict()
        matched = {k: v for k, v in backbone_state.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(matched)
        model.model.load_state_dict(model_state, strict=False)
        print(f"Loaded {len(matched)} pretrained layers from {pretrained_path}")

    model = model.to(device)
    print(f"Model parameters: {model.get_num_parameters()/1e6:.2f}M")

    freeze_epochs = phase2.get('freeze_backbone_epochs', 0)
    class_weight_real = phase2.get('class_weight_real', 1.0)
    weight = torch.tensor([class_weight_real, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=phase2['learning_rate'],
                            weight_decay=phase2['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )

    # Patch config for BaseTrainer
    ft_config = {**config, 'num_epochs': phase2['num_epochs'],
                 'save_dir': config['save_dir']}

    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=ft_config
    )

    start_epoch = 0
    if resume_checkpoint:
        start_epoch = trainer.load_checkpoint(resume_checkpoint)

    # Optionally freeze backbone for first N epochs (skip if already past freeze period)
    if freeze_epochs > 0 and start_epoch < freeze_epochs:
        print(f"Freezing backbone for first {freeze_epochs} epochs...")
        for name, param in model.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    for epoch in range(start_epoch, phase2['num_epochs']):
        if epoch == freeze_epochs and freeze_epochs > 0:
            print("Unfreezing backbone...")
            for param in model.parameters():
                param.requires_grad = True

        print(f"\nEpoch {epoch+1}/{phase2['num_epochs']}")
        print("-" * 40)

        train_loss, train_acc = trainer.train_epoch()
        trainer.train_losses.append(train_loss)

        val_loss, val_acc = trainer.validate()
        trainer.val_losses.append(val_loss)
        trainer.val_accuracies.append(val_acc)

        scheduler.step(val_acc)

        trainer.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if trainer.check_early_stop(val_acc):
            break

        is_best = val_acc > trainer.best_val_acc
        if is_best:
            trainer.best_val_acc = val_acc
        trainer.save_checkpoint(epoch, is_best)

        # Sync to /kaggle/output/ after every checkpoint so work survives session timeout
        if os.path.exists('/kaggle/output'):
            import shutil
            shutil.copytree(config['save_dir'], f"/kaggle/output/results/models/c23/strategy4_ssl", dirs_exist_ok=True)

    print(f"\nBest Val Acc: {trainer.best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to Phase 2 checkpoint to resume from (skips Phase 1)')
    args = parser.parse_args()

    with open(os.environ.get('CONFIG_DIR', 'configs/c40') + '/strategy4_ssl.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    if args.resume:
        pretrained_path = config['phase_2'].get('pretrained_path')
    else:
        pretrained_path = pretrain_simclr(config, device)
    finetune(config, pretrained_path, device, resume_checkpoint=args.resume)

if __name__ == '__main__':
    main()
