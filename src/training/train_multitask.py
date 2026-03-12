"""
Strategy 6: Multi-Task Learning

Jointly trains binary (real/fake) and manipulation-type classification heads
on a shared EfficientNet-B1 backbone.
"""
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import os
import time
from tqdm import tqdm
from PIL import Image

from src.models.multitask import MultiTaskEfficientNet
from src.data.dataset import FaceForensicsDataset
from src.data.augmentation import get_baseline_transforms, get_val_transforms
from src.utils.device import get_device

# Map manipulation type string to multiclass label
MANIP_TO_LABEL = {
    'Real': 0,
    'Deepfakes': 1,
    'Face2Face': 2,
    'FaceSwap': 3,
    'NeuralTextures': 4
}


class MultiTaskDataset(Dataset):
    """
    Wraps FaceForensicsDataset to provide both binary and multiclass labels.
    """

    def __init__(self, base_dataset):
        self.samples = base_dataset.samples
        self.transform = base_dataset.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, binary_label, manip_type = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        multiclass_label = MANIP_TO_LABEL.get(manip_type, 0)
        return image, binary_label, multiclass_label


def train_epoch(model, loader, criterion_binary, criterion_multi,
                optimizer, device, loss_weights):
    model.train()
    running_loss = 0.0
    binary_correct = 0
    multi_correct = 0
    total = 0
    w_bin, w_multi = loss_weights['binary'], loss_weights['multiclass']

    pbar = tqdm(loader, desc='Training')
    for images, binary_labels, multi_labels in pbar:
        images = images.to(device)
        binary_labels = binary_labels.to(device)
        multi_labels = multi_labels.to(device)

        optimizer.zero_grad()
        binary_out, multi_out = model(images)

        loss = w_bin * criterion_binary(binary_out, binary_labels) + \
               w_multi * criterion_multi(multi_out, multi_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, bin_pred = binary_out.max(1)
        _, multi_pred = multi_out.max(1)
        total += binary_labels.size(0)
        binary_correct += bin_pred.eq(binary_labels).sum().item()
        multi_correct += multi_pred.eq(multi_labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'bin_acc': f'{100.*binary_correct/total:.1f}%',
            'multi_acc': f'{100.*multi_correct/total:.1f}%'
        })

    return (running_loss / len(loader),
            100. * binary_correct / total,
            100. * multi_correct / total)


def validate(model, loader, criterion_binary, criterion_multi,
             device, loss_weights):
    model.eval()
    running_loss = 0.0
    binary_correct = 0
    multi_correct = 0
    total = 0
    w_bin, w_multi = loss_weights['binary'], loss_weights['multiclass']

    with torch.no_grad():
        for images, binary_labels, multi_labels in tqdm(loader, desc='Validation'):
            images = images.to(device)
            binary_labels = binary_labels.to(device)
            multi_labels = multi_labels.to(device)

            binary_out, multi_out = model(images)
            loss = w_bin * criterion_binary(binary_out, binary_labels) + \
                   w_multi * criterion_multi(multi_out, multi_labels)

            running_loss += loss.item()
            _, bin_pred = binary_out.max(1)
            _, multi_pred = multi_out.max(1)
            total += binary_labels.size(0)
            binary_correct += bin_pred.eq(binary_labels).sum().item()
            multi_correct += multi_pred.eq(multi_labels).sum().item()

    return (running_loss / len(loader),
            100. * binary_correct / total,
            100. * multi_correct / total)


def main():
    with open(os.environ.get('CONFIG_DIR', 'configs/c40') + '/strategy6_multitask.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(config['save_dir'], exist_ok=True)

    base_train = FaceForensicsDataset(
        data_root=config['data_root'],
        split='train',
        compression=config['compression'],
        transform=get_baseline_transforms()
    )
    base_val = FaceForensicsDataset(
        data_root=config['data_root'],
        split='val',
        compression=config['compression'],
        transform=get_val_transforms()
    )

    train_dataset = MultiTaskDataset(base_train)
    val_dataset = MultiTaskDataset(base_val)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'], pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'], pin_memory=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    model = MultiTaskEfficientNet(pretrained=True)
    model = model.to(device)

    loss_weights = config['loss_weights']
    criterion_binary = nn.CrossEntropyLoss()
    criterion_multi = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
    )

    print(f"\n{'='*60}")
    print(f"Training {config['strategy_name']} (Multi-Task Learning)")
    print(f"{'='*60}\n")

    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stopping_patience = config.get('early_stopping_patience', 5)
    start_time = time.time()

    log_dir = config.get('log_dir', 'results/logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'multi-task_learning.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_bin_acc', 'train_multi_acc',
                                'val_loss', 'val_bin_acc', 'val_multi_acc'])

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)

        train_loss, train_bin_acc, train_multi_acc = train_epoch(
            model, train_loader, criterion_binary, criterion_multi,
            optimizer, device, loss_weights
        )
        val_loss, val_bin_acc, val_multi_acc = validate(
            model, val_loader, criterion_binary, criterion_multi,
            device, loss_weights
        )

        scheduler.step(val_bin_acc)

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1,
                                    f'{train_loss:.4f}', f'{train_bin_acc:.2f}', f'{train_multi_acc:.2f}',
                                    f'{val_loss:.4f}', f'{val_bin_acc:.2f}', f'{val_multi_acc:.2f}'])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Binary Acc: {train_bin_acc:.2f}% | Multi Acc: {train_multi_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Binary Acc: {val_bin_acc:.2f}% | Multi Acc: {val_multi_acc:.2f}%")

        is_best = val_bin_acc > best_val_acc
        if is_best:
            best_val_acc = val_bin_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_bin_acc': val_bin_acc,
            'val_multi_acc': val_multi_acc,
            'config': config
        }
        torch.save(checkpoint, os.path.join(config['save_dir'], 'last_model.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f'Saved best model with binary accuracy: {val_bin_acc:.2f}%')

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping: val_acc has not improved for {early_stopping_patience} epochs.")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Best Val Binary Acc: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
