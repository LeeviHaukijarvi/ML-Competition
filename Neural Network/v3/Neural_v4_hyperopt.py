"""
Targeted hyperparameter optimization for CSI classification.
Based on ensemble analysis showing deeper models (5-layer) perform best.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import json
from datetime import datetime

# ===========================
# Core Components (same as v4)
# ===========================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_csi_features(csi_data):
    """Extract enhanced features from raw CSI I/Q data."""
    n_samples = csi_data.shape[0]
    inactive_indices = [0] + list(range(27, 38))
    active_indices = [i for i in range(64) if i not in inactive_indices]

    csi_reshaped = csi_data.reshape(n_samples, 64, 2, 2)
    I = csi_reshaped[:, :, :, 0]
    Q = csi_reshaped[:, :, :, 1]

    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)

    amplitude_active = amplitude[:, active_indices, :]
    phase_active = phase[:, active_indices, :]

    amp_features = amplitude_active.reshape(n_samples, -1)
    phase_features = phase_active.reshape(n_samples, -1)

    amp_diff = np.abs(amplitude_active[:, :, 0] - amplitude_active[:, :, 1])
    phase_diff = phase_active[:, :, 0] - phase_active[:, :, 1]
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

    n_active = len(active_indices)
    band_size = n_active // 3
    low_band = amplitude_active[:, :band_size, :]
    mid_band = amplitude_active[:, band_size:2*band_size, :]
    high_band = amplitude_active[:, 2*band_size:, :]

    band_stats = []
    for band in [low_band, mid_band, high_band]:
        for antenna_idx in range(2):
            band_stats.append(np.mean(band[:, :, antenna_idx], axis=1))
            band_stats.append(np.std(band[:, :, antenna_idx], axis=1))
            band_stats.append(np.max(band[:, :, antenna_idx], axis=1))

    band_stats = np.column_stack(band_stats)

    features = np.concatenate([
        amp_features, phase_features, amp_diff, phase_diff, band_stats
    ], axis=1)

    return features


class CSIDataset(Dataset):
    def __init__(self, X_meta, X_csi, y=None):
        self.X_meta = torch.FloatTensor(X_meta)
        self.X_csi = torch.FloatTensor(X_csi)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X_meta)

    def __getitem__(self, idx):
        x = torch.cat([self.X_meta[idx], self.X_csi[idx]])
        if self.y is not None:
            return x, self.y[idx]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class CSIResNetClassifier(nn.Module):
    """Optimized deeper architecture based on ensemble findings."""

    def __init__(self, input_dim, num_classes=10, hidden_dims=[1024, 512, 256, 128, 64],
                 dropout_schedule=[0.4, 0.35, 0.3, 0.25, 0.2]):
        super(CSIResNetClassifier, self).__init__()

        self.hidden_dims = hidden_dims
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_schedule[0])

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            dropout = dropout_schedule[min(i+1, len(dropout_schedule) - 1)]
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout=dropout)
            )

        # Final layers
        self.fc1 = nn.Linear(hidden_dims[-1], 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.2)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        x = self.input_dropout(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc_out(x)
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = torch.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)

        if self.reduction == 'mean':
            loss = loss.mean()
            nll = nll.mean()
        return (1 - self.epsilon) * nll + self.epsilon * loss / n_classes


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + (self.base_lr - self.lr_min) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        if use_mixup and np.random.rand() > 0.5:
            X_batch, y_a, y_b, lam = mixup_data(X_batch, y_batch, mixup_alpha)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), 100. * correct / total


def train_with_config(config, train_loader, val_loader, device, config_id):
    """Train a model with specific hyperparameters."""

    set_seed(42)  # Same seed for fair comparison

    # Build model
    model = CSIResNetClassifier(
        input_dim=config['input_dim'],
        num_classes=10,
        hidden_dims=config['hidden_dims'],
        dropout_schedule=config['dropout_schedule']
    ).to(device)

    # Setup training
    criterion = LabelSmoothingCrossEntropy(epsilon=config['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(),
                           lr=config['lr'],
                           weight_decay=config['weight_decay'])

    scheduler = WarmupCosineScheduler(optimizer,
                                     warmup_epochs=config['warmup_epochs'],
                                     total_epochs=config['num_epochs'])

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=True, mixup_alpha=config['mixup_alpha']
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'hyperopt_model_{config_id}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break

    return best_val_acc


def main():
    print("="*70)
    print("CSI CLASSIFICATION - HYPERPARAMETER OPTIMIZATION")
    print("="*70)

    # Load and preprocess data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test_nolabels.csv')

    X_train_full = train_df.drop(['position'], axis=1).values
    y_train_full = train_df['position'].values

    if X_train_full.shape[1] > 260:
        X_train_full = X_train_full[:, 1:]

    X_test = test_df.values
    if X_test.shape[1] > 260:
        X_test = X_test[:, 1:]

    # Feature engineering
    X_train_meta = X_train_full[:, :5]
    X_train_csi = X_train_full[:, 5:]
    X_test_meta = X_test[:, :5]
    X_test_csi = X_test[:, 5:]

    print("Applying feature engineering...")
    X_train_csi_features = extract_csi_features(X_train_csi)
    X_test_csi_features = extract_csi_features(X_test_csi)

    # Train/val split
    indices = np.arange(len(X_train_full))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, random_state=42, stratify=y_train_full
    )

    X_train_meta_split = X_train_meta[train_idx]
    X_train_csi_split = X_train_csi_features[train_idx]
    y_train_split = y_train_full[train_idx]

    X_val_meta_split = X_train_meta[val_idx]
    X_val_csi_split = X_train_csi_features[val_idx]
    y_val_split = y_train_full[val_idx]

    # Normalize
    scaler_meta = StandardScaler()
    X_train_meta_split = scaler_meta.fit_transform(X_train_meta_split)
    X_val_meta_split = scaler_meta.transform(X_val_meta_split)
    X_test_meta_norm = scaler_meta.transform(X_test_meta)

    scaler_csi = StandardScaler()
    X_train_csi_split = scaler_csi.fit_transform(X_train_csi_split)
    X_val_csi_split = scaler_csi.transform(X_val_csi_split)
    X_test_csi_norm = scaler_csi.transform(X_test_csi_features)

    # Create dataloaders
    batch_size = 256
    train_dataset = CSIDataset(X_train_meta_split, X_train_csi_split, y_train_split)
    val_dataset = CSIDataset(X_val_meta_split, X_val_csi_split, y_val_split)
    test_dataset = CSIDataset(X_test_meta_norm, X_test_csi_norm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 5 + X_train_csi_features.shape[1]

    # Define hyperparameter search space
    # Based on ensemble analysis: focus on deeper models with varied regularization

    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH SPACE")
    print("="*70)

    param_grid = {
        'lr': [0.0005, 0.0008, 0.001, 0.0012],
        'weight_decay': [5e-6, 1e-5, 2e-5, 5e-5],
        'dropout_base': [0.35, 0.40, 0.45],  # First layer dropout
        'label_smoothing': [0.05, 0.1, 0.15],
        'mixup_alpha': [0.15, 0.2, 0.25]
    }

    # Architecture: Use best performing 5-layer deep model
    base_architecture = [1024, 512, 256, 128, 64]

    print(f"Learning rates: {param_grid['lr']}")
    print(f"Weight decay: {param_grid['weight_decay']}")
    print(f"Dropout (first layer): {param_grid['dropout_base']}")
    print(f"Label smoothing: {param_grid['label_smoothing']}")
    print(f"Mixup alpha: {param_grid['mixup_alpha']}")
    print(f"Architecture: {base_architecture}")

    # Smart grid search: test most promising combinations
    # Focus on lr and weight_decay first, then refine dropout

    results = []
    config_id = 0

    # Phase 1: Optimize LR and weight decay
    print("\n" + "="*70)
    print("PHASE 1: OPTIMIZE LEARNING RATE & WEIGHT DECAY")
    print("="*70)

    for lr in param_grid['lr']:
        for wd in param_grid['weight_decay']:
            config_id += 1
            config = {
                'input_dim': input_dim,
                'hidden_dims': base_architecture,
                'dropout_schedule': [0.4, 0.35, 0.3, 0.25, 0.2, 0.2],
                'lr': lr,
                'weight_decay': wd,
                'label_smoothing': 0.1,
                'mixup_alpha': 0.2,
                'warmup_epochs': 10,
                'num_epochs': 150,
                'patience': 30
            }

            print(f"\n[{config_id}] lr={lr}, wd={wd}")
            val_acc = train_with_config(config, train_loader, val_loader, device, config_id)

            results.append({
                'config_id': config_id,
                'lr': lr,
                'weight_decay': wd,
                'dropout_base': 0.4,
                'label_smoothing': 0.1,
                'mixup_alpha': 0.2,
                'val_acc': val_acc
            })

            print(f"Val Accuracy: {val_acc:.2f}%")

    # Find best LR and weight decay
    best_phase1 = max(results, key=lambda x: x['val_acc'])
    best_lr = best_phase1['lr']
    best_wd = best_phase1['weight_decay']

    print(f"\n{'='*70}")
    print(f"BEST FROM PHASE 1: lr={best_lr}, wd={best_wd}, acc={best_phase1['val_acc']:.2f}%")
    print(f"{'='*70}")

    # Phase 2: Optimize dropout and regularization with best LR/WD
    print("\n" + "="*70)
    print("PHASE 2: OPTIMIZE DROPOUT & REGULARIZATION")
    print("="*70)

    for dropout_base in param_grid['dropout_base']:
        for label_smooth in param_grid['label_smoothing']:
            for mixup in param_grid['mixup_alpha']:
                config_id += 1

                # Create dropout schedule
                dropout_schedule = [dropout_base, dropout_base-0.05, dropout_base-0.1,
                                   dropout_base-0.15, dropout_base-0.2, 0.2]
                dropout_schedule = [max(0.15, d) for d in dropout_schedule]  # Minimum 0.15

                config = {
                    'input_dim': input_dim,
                    'hidden_dims': base_architecture,
                    'dropout_schedule': dropout_schedule,
                    'lr': best_lr,
                    'weight_decay': best_wd,
                    'label_smoothing': label_smooth,
                    'mixup_alpha': mixup,
                    'warmup_epochs': 10,
                    'num_epochs': 150,
                    'patience': 30
                }

                print(f"\n[{config_id}] dropout={dropout_base}, label_smooth={label_smooth}, mixup={mixup}")
                val_acc = train_with_config(config, train_loader, val_loader, device, config_id)

                results.append({
                    'config_id': config_id,
                    'lr': best_lr,
                    'weight_decay': best_wd,
                    'dropout_base': dropout_base,
                    'label_smoothing': label_smooth,
                    'mixup_alpha': mixup,
                    'val_acc': val_acc
                })

                print(f"Val Accuracy: {val_acc:.2f}%")

    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_acc', ascending=False)
    results_df.to_csv('hyperopt_results.csv', index=False)

    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*70)
    print("\nTop 10 configurations:")
    print(results_df.head(10).to_string(index=False))

    best_config = results_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION:")
    print(f"  Learning rate: {best_config['lr']}")
    print(f"  Weight decay: {best_config['weight_decay']}")
    print(f"  Dropout (base): {best_config['dropout_base']}")
    print(f"  Label smoothing: {best_config['label_smoothing']}")
    print(f"  Mixup alpha: {best_config['mixup_alpha']}")
    print(f"  Validation Accuracy: {best_config['val_acc']:.2f}%")
    print(f"  Improvement over baseline (95.71%): {best_config['val_acc'] - 95.71:.2f}%")
    print(f"  Improvement over previous best (96.48%): {best_config['val_acc'] - 96.48:.2f}%")
    print(f"{'='*70}")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # LR vs Accuracy
    axes[0, 0].scatter(results_df['lr'], results_df['val_acc'])
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Learning Rate vs Accuracy')
    axes[0, 0].axhline(y=96.48, color='r', linestyle='--', alpha=0.5, label='Previous Best')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Weight Decay vs Accuracy
    axes[0, 1].scatter(results_df['weight_decay'], results_df['val_acc'])
    axes[0, 1].set_xlabel('Weight Decay')
    axes[0, 1].set_ylabel('Validation Accuracy (%)')
    axes[0, 1].set_title('Weight Decay vs Accuracy')
    axes[0, 1].axhline(y=96.48, color='r', linestyle='--', alpha=0.5, label='Previous Best')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Dropout vs Accuracy
    axes[0, 2].scatter(results_df['dropout_base'], results_df['val_acc'])
    axes[0, 2].set_xlabel('Dropout (Base)')
    axes[0, 2].set_ylabel('Validation Accuracy (%)')
    axes[0, 2].set_title('Dropout vs Accuracy')
    axes[0, 2].axhline(y=96.48, color='r', linestyle='--', alpha=0.5, label='Previous Best')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Label Smoothing vs Accuracy
    axes[1, 0].scatter(results_df['label_smoothing'], results_df['val_acc'])
    axes[1, 0].set_xlabel('Label Smoothing')
    axes[1, 0].set_ylabel('Validation Accuracy (%)')
    axes[1, 0].set_title('Label Smoothing vs Accuracy')
    axes[1, 0].axhline(y=96.48, color='r', linestyle='--', alpha=0.5, label='Previous Best')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Mixup vs Accuracy
    axes[1, 1].scatter(results_df['mixup_alpha'], results_df['val_acc'])
    axes[1, 1].set_xlabel('Mixup Alpha')
    axes[1, 1].set_ylabel('Validation Accuracy (%)')
    axes[1, 1].set_title('Mixup Alpha vs Accuracy')
    axes[1, 1].axhline(y=96.48, color='r', linestyle='--', alpha=0.5, label='Previous Best')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Top configs
    top_10 = results_df.head(10)
    axes[1, 2].barh(range(len(top_10)), top_10['val_acc'])
    axes[1, 2].set_yticks(range(len(top_10)))
    axes[1, 2].set_yticklabels([f"Config {int(x)}" for x in top_10['config_id']])
    axes[1, 2].set_xlabel('Validation Accuracy (%)')
    axes[1, 2].set_title('Top 10 Configurations')
    axes[1, 2].axvline(x=96.48, color='r', linestyle='--', alpha=0.5, label='Previous Best')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hyperopt_analysis.png', dpi=150)
    print('\nHyperparameter analysis saved to hyperopt_analysis.png')

    # Save best config for later use
    best_config_dict = {
        'lr': float(best_config['lr']),
        'weight_decay': float(best_config['weight_decay']),
        'dropout_base': float(best_config['dropout_base']),
        'label_smoothing': float(best_config['label_smoothing']),
        'mixup_alpha': float(best_config['mixup_alpha']),
        'val_acc': float(best_config['val_acc']),
        'architecture': base_architecture
    }

    with open('best_hyperparams.json', 'w') as f:
        json.dump(best_config_dict, f, indent=2)

    print('\nBest hyperparameters saved to best_hyperparams.json')


if __name__ == '__main__':
    main()
