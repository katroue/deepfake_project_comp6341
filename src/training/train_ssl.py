"""
Strategy 4: Self-Supervised Learning (SimSiam)

Phase 1: SimSiam pretraining on all face images (real + fake, no labels)
Phase 2: Supervised fine-tuning on FF++ with pretrained backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
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


class SimSiamProjectionHead(nn.Module):
    """3-layer MLP projector with BN (no affine on final BN, per SimSiam paper)."""
    def __init__(self, in_dim, proj_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class SimSiamPredictorHead(nn.Module):
    """2-layer MLP predictor."""
    def __init__(self, proj_dim=512, pred_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


def simsiam_loss(p, z):
    """Negative cosine similarity with stop-gradient on z."""
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def pretrain_simsiam(config, device, resume_checkpoint=None):
    """Phase 1: SimSiam self-supervised pretraining."""
    phase1 = config['phase_1']
    if not phase1.get('enabled', True):
        print("Phase 1 (SSL pretraining) disabled, skipping.")
        return None

    print("\n" + "="*60)
    print("Phase 1: SimSiam Self-Supervised Pretraining")
    print("="*60)

    save_dir = os.path.join(config['save_dir'], 'phase1_pretrained')
    os.makedirs(save_dir, exist_ok=True)

    # Build dataset of all face images (labels ignored)
    base_dataset = FaceForensicsDataset(
        data_root=config['data_root'],
        split='train',
        compression=config['compression'],
        transform=None
    )
    ssl_transform = get_ssl_transforms()
    pair_dataset = ContrastivePairDataset(base_dataset, ssl_transform)
    loader = DataLoader(
        pair_dataset,
        batch_size=phase1['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

    # Backbone (no classifier) + projector + predictor
    backbone = timm.create_model('efficientnet_b1', pretrained=False, num_classes=0, global_pool='avg')
    feat_dim = backbone.num_features
    proj_dim = phase1['projection_dim']
    pred_dim = phase1['pred_dim']
    projector = SimSiamProjectionHead(feat_dim, proj_dim=proj_dim)
    predictor = SimSiamPredictorHead(proj_dim=proj_dim, pred_dim=pred_dim)

    backbone = backbone.to(device)
    projector = projector.to(device)
    predictor = predictor.to(device)

    optimizer = optim.Adam(
        list(backbone.parameters()) + list(projector.parameters()) + list(predictor.parameters()),
        lr=phase1['learning_rate']
    )
    scaler = GradScaler(device=device.type, enabled=(device.type == 'cuda'))

    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        projector.load_state_dict(ckpt['projector_state_dict'])
        predictor.load_state_dict(ckpt['predictor_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed Phase 1 from epoch {ckpt['epoch'] + 1} (loss={ckpt.get('loss', '?'):.4f})")

    for epoch in range(start_epoch, phase1['num_epochs']):
        backbone.train()
        projector.train()
        predictor.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"SSL Epoch {epoch+1}/{phase1['num_epochs']}")
        for view1, view2 in pbar:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                z1 = projector(backbone(view1))
                z2 = projector(backbone(view2))
                p1 = predictor(z1)
                p2 = predictor(z2)
                loss = simsiam_loss(p1, z2) / 2 + simsiam_loss(p2, z1) / 2
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(projector.parameters()) + list(predictor.parameters()),
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(loader)
        print(f"SSL Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Save full resumable checkpoint after every epoch
        resume_path = os.path.join(save_dir, 'phase1_resume.pth')
        torch.save({
            'epoch': epoch,
            'loss': avg_loss,
            'backbone_state_dict': backbone.state_dict(),
            'projector_state_dict': projector.state_dict(),
            'predictor_state_dict': predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, resume_path)
        # Save backbone-only checkpoint for Phase 2
        checkpoint_path = os.path.join(save_dir, 'best_model.pth')
        torch.save({'backbone_state_dict': backbone.state_dict()}, checkpoint_path)
        print(f"Checkpoints saved (resume: {resume_path})")
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
                              shuffle=True, num_workers=phase2['num_workers'], pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=phase2['batch_size'],
                            shuffle=False, num_workers=phase2['num_workers'], pin_memory=(device.type == 'cuda'))

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
        # Reset BN running stats — Phase 1 stats are calibrated for SSL augmented data
        # (color jitter, grayscale, blur), which causes eval-mode mismatch in Phase 2
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.reset_running_stats()
        print("Reset BatchNorm running stats for Phase 2 adaptation")

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
    parser.add_argument('--resume-phase1', type=str, default=None,
                        help='Path to Phase 1 resume checkpoint (phase1_resume.pth) to continue SSL pretraining')
    args = parser.parse_args()

    with open(os.environ.get('CONFIG_DIR', 'configs/c40') + '/strategy4_ssl.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    if args.resume:
        pretrained_path = config['phase_2'].get('pretrained_path')
    else:
        pretrained_path = pretrain_simsiam(config, device, resume_checkpoint=args.resume_phase1)
        # If Phase 1 was disabled/skipped, fall back to the configured pretrained path
        if pretrained_path is None:
            pretrained_path = config['phase_2'].get('pretrained_path')
    finetune(config, pretrained_path, device, resume_checkpoint=args.resume)

if __name__ == '__main__':
    main()
