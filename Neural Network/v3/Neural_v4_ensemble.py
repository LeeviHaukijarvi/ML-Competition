"""
Ensemble training script for CSI classification.
Trains multiple model variants and generates ensemble predictions.
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
import copy
import os
from typing import List, Dict

# Import from Neural_v4.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reproduce all necessary components
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===========================
# Feature Engineering (same as v4)
# ===========================

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


# ===========================
# Model Architecture (same as v4)
# ===========================

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
    def __init__(self, input_dim, num_classes=10, use_attention=False, hidden_dims=[1024, 512, 256, 128]):
        super(CSIResNetClassifier, self).__init__()

        self.use_attention = use_attention
        self.hidden_dims = hidden_dims

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(0.4)

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        dropout_values = [0.35, 0.3, 0.25, 0.2]

        for i in range(len(hidden_dims) - 1):
            dropout = dropout_values[min(i, len(dropout_values) - 1)]
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout=dropout)
            )

        # Optional attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.Tanh(),
                nn.Linear(64, 1)
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

        if self.use_attention:
            attention_weights = torch.softmax(self.attention(x), dim=1)
            x = x * attention_weights

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc_out(x)
        return x


# ===========================
# Training Utilities (same as v4)
# ===========================

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
        elif self.reduction == 'sum':
            loss = loss.sum()
            nll = nll.sum()

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


def get_predictions(model, loader, device):
    """Get softmax probabilities for a dataset."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # Handle both labeled and unlabeled datasets
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                X_batch = batch[0].to(device)
            else:
                X_batch = batch.to(device)

            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


# ===========================
# Ensemble Training
# ===========================

def train_ensemble_model(config, train_loader, val_loader, device, model_name):
    """Train a single model variant for the ensemble."""

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(f"Config: {config}")

    # Set seed
    set_seed(config['seed'])

    # Create model
    model = CSIResNetClassifier(
        input_dim=config['input_dim'],
        num_classes=10,
        use_attention=config.get('use_attention', False),
        hidden_dims=config.get('hidden_dims', [1024, 512, 256, 128])
    ).to(device)

    # Setup training
    criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.get('lr', 0.001),
                           weight_decay=config.get('weight_decay', 1e-5))

    scheduler = WarmupCosineScheduler(optimizer,
                                     warmup_epochs=config.get('warmup_epochs', 10),
                                     total_epochs=config['num_epochs'])

    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=config.get('use_mixup', True),
            mixup_alpha=config.get('mixup_alpha', 0.2)
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr = scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                  f'Val Acc: {val_acc:.2f}%, Train Acc: {train_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{model_name}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 40):
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Best Val Acc: {best_val_acc:.2f}%')

    # Load best model
    model.load_state_dict(torch.load(f'{model_name}.pth'))

    return model, best_val_acc, history


def main():
    print("="*70)
    print("CSI CLASSIFICATION - ENSEMBLE TRAINING")
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

    # Define ensemble configurations
    ensemble_configs = [
        {
            'name': 'model1_base_seed42',
            'seed': 42,
            'input_dim': input_dim,
            'hidden_dims': [1024, 512, 256, 128],
            'use_attention': False,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'num_epochs': 200,
            'patience': 40,
            'use_mixup': True,
            'mixup_alpha': 0.2
        },
        {
            'name': 'model2_base_seed123',
            'seed': 123,
            'input_dim': input_dim,
            'hidden_dims': [1024, 512, 256, 128],
            'use_attention': False,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'num_epochs': 200,
            'patience': 40,
            'use_mixup': True,
            'mixup_alpha': 0.2
        },
        {
            'name': 'model3_base_seed456',
            'seed': 456,
            'input_dim': input_dim,
            'hidden_dims': [1024, 512, 256, 128],
            'use_attention': False,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'num_epochs': 200,
            'patience': 40,
            'use_mixup': True,
            'mixup_alpha': 0.2
        },
        {
            'name': 'model4_wider',
            'seed': 42,
            'input_dim': input_dim,
            'hidden_dims': [1024, 1024, 512, 256],
            'use_attention': False,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'num_epochs': 200,
            'patience': 40,
            'use_mixup': True,
            'mixup_alpha': 0.2
        },
        {
            'name': 'model5_deeper',
            'seed': 42,
            'input_dim': input_dim,
            'hidden_dims': [1024, 512, 256, 128, 64],
            'use_attention': False,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'num_epochs': 200,
            'patience': 40,
            'use_mixup': True,
            'mixup_alpha': 0.2
        },
        {
            'name': 'model6_with_attention',
            'seed': 789,
            'input_dim': input_dim,
            'hidden_dims': [1024, 512, 256, 128],
            'use_attention': True,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'num_epochs': 200,
            'patience': 40,
            'use_mixup': True,
            'mixup_alpha': 0.2
        },
        {
            'name': 'model7_stronger_mixup',
            'seed': 42,
            'input_dim': input_dim,
            'hidden_dims': [1024, 512, 256, 128],
            'use_attention': False,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'num_epochs': 200,
            'patience': 40,
            'use_mixup': True,
            'mixup_alpha': 0.3
        }
    ]

    # Train all models
    models = []
    val_accs = []
    histories = []

    for config in ensemble_configs:
        model, val_acc, history = train_ensemble_model(
            config, train_loader, val_loader, device, config['name']
        )
        models.append((config['name'], model, config))
        val_accs.append(val_acc)
        histories.append(history)

    # Print ensemble summary
    print("\n" + "="*70)
    print("ENSEMBLE SUMMARY")
    print("="*70)
    for i, (name, _, _) in enumerate(models):
        print(f"{name}: {val_accs[i]:.2f}%")

    print(f"\nMean validation accuracy: {np.mean(val_accs):.2f}%")
    print(f"Best single model: {max(val_accs):.2f}%")

    # Generate ensemble predictions on validation set
    print("\nGenerating ensemble predictions on validation set...")
    val_probs_ensemble = []

    for name, model, config in models:
        probs = get_predictions(model, val_loader, device)
        val_probs_ensemble.append(probs)

    # Average predictions
    val_probs_avg = np.mean(val_probs_ensemble, axis=0)
    val_preds_ensemble = np.argmax(val_probs_avg, axis=1)

    # Calculate ensemble accuracy
    ensemble_val_acc = 100. * np.mean(val_preds_ensemble == y_val_split)

    print(f"\n{'='*70}")
    print(f"ENSEMBLE VALIDATION ACCURACY: {ensemble_val_acc:.2f}%")
    print(f"Improvement over best single model: {ensemble_val_acc - max(val_accs):.2f}%")
    print(f"Improvement over baseline (95.71%): {ensemble_val_acc - 95.71:.2f}%")
    print(f"{'='*70}")

    # Generate test predictions
    print("\nGenerating ensemble predictions on test set...")
    test_probs_ensemble = []

    for name, model, config in models:
        probs = get_predictions(model, test_loader, device)
        test_probs_ensemble.append(probs)

    test_probs_avg = np.mean(test_probs_ensemble, axis=0)
    test_preds_ensemble = np.argmax(test_probs_avg, axis=1)

    # Create submission
    submission = pd.DataFrame({
        'id': range(len(test_preds_ensemble)),
        'position': test_preds_ensemble
    })

    submission.to_csv('submission_v4_ensemble.csv', index=False)
    print(f"\nEnsemble predictions saved to 'submission_v4_ensemble.csv'")
    print(f"Prediction distribution:\n{pd.Series(test_preds_ensemble).value_counts().sort_index()}")

    # Plot ensemble results
    plt.figure(figsize=(15, 10))

    # Individual model accuracies
    plt.subplot(2, 2, 1)
    model_names = [name for name, _, _ in models]
    plt.barh(model_names, val_accs)
    plt.axvline(x=95.71, color='r', linestyle='--', label='Baseline')
    plt.axvline(x=ensemble_val_acc, color='g', linestyle='--', label='Ensemble')
    plt.xlabel('Validation Accuracy (%)')
    plt.title('Individual Model Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Training curves for best models
    top_3_idx = np.argsort(val_accs)[-3:]

    plt.subplot(2, 2, 2)
    for idx in top_3_idx:
        plt.plot(histories[idx]['val_acc'], label=f'{model_names[idx][:15]}...')
    plt.axhline(y=95.71, color='r', linestyle='--', alpha=0.5, label='Baseline')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Top 3 Models - Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    for idx in top_3_idx:
        plt.plot(histories[idx]['val_loss'], label=f'{model_names[idx][:15]}...')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Top 3 Models - Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    improvement = [acc - 95.71 for acc in val_accs]
    plt.barh(model_names, improvement, color=['green' if x > 0 else 'red' for x in improvement])
    plt.xlabel('Improvement over Baseline (%)')
    plt.title('Accuracy Improvement by Model')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ensemble_results.png', dpi=150)
    print('\nEnsemble analysis saved to ensemble_results.png')


if __name__ == '__main__':
    main()
