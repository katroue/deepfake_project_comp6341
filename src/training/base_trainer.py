import csv
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import time
import os

class BaseTrainer:
    """
    Base trainer class for all 6 strategies
    """
    
    def __init__(self, model, train_loader, val_loader, 
                criterion, optimizer, scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        self._epochs_no_improve = 0
        self.use_amp = device.type == 'cuda'
        self.scaler = GradScaler(device=device.type, enabled=(device.type == 'cuda'))

        # CSV log setup
        log_dir = config.get('log_dir', 'results/logs')
        os.makedirs(log_dir, exist_ok=True)
        strategy_name = config.get('strategy_name', 'unknown').replace(' ', '_').lower()
        self.log_path = os.path.join(log_dir, f'{strategy_name}.csv')
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    def check_early_stop(self, val_acc) -> bool:
        """Update counter and return True if training should stop."""
        if val_acc > self.best_val_acc:
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
        if self._epochs_no_improve >= self.early_stopping_patience:
            print(f"\nEarly stopping: val_acc has not improved for {self.early_stopping_patience} epochs.")
            return True
        return False

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Append one row to the CSV log."""
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1,
                                    f'{train_loss:.4f}', f'{train_acc:.2f}',
                                    f'{val_loss:.4f}', f'{val_acc:.2f}'])

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': self.val_accuracies[-1],
            'config': self.config
        }
        
        # Save last checkpoint
        save_path = os.path.join(self.config['save_dir'], 'last_model.pth')
        torch.save(checkpoint, save_path)
        
        # Save best checkpoint
        if is_best:
            save_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f'✅ Saved best model with accuracy: {self.val_accuracies[-1]:.2f}%')
    
    def load_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint. Returns the next epoch to train."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['val_acc']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from {checkpoint_path} (epoch {checkpoint['epoch']+1}, val_acc {checkpoint['val_acc']:.2f}%)")
        return start_epoch

    def train(self, num_epochs, start_epoch=0):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Training {self.config['strategy_name']}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_acc)
            
            # Log to CSV
            self.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Early stopping (must check before updating best_val_acc)
            if self.check_early_stop(val_acc):
                break

            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, is_best)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total Time: {total_time/3600:.2f} hours")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        