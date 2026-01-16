"""
Advanced ensemble with:
1. 5-fold cross-validation (uses ALL training data)
2. Architectural diversity (ResNet + 1D CNN)
3. Multiple seeds per architecture

Target: 97%+ validation accuracy
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
import copy

# ===========================
# Setup
# ===========================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===========================
# Feature Engineering
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


def extract_csi_2d(csi_data):
    """
    Extract 2D representation for CNN: (n_samples, channels, subcarriers)
    Channels: [amp_ant1, amp_ant2, phase_ant1, phase_ant2, amp_diff, phase_diff]
    """
    n_samples = csi_data.shape[0]
    inactive_indices = [0] + list(range(27, 38))
    active_indices = [i for i in range(64) if i not in inactive_indices]

    csi_reshaped = csi_data.reshape(n_samples, 64, 2, 2)
    I = csi_reshaped[:, :, :, 0]
    Q = csi_reshaped[:, :, :, 1]

    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)

    # Filter to active subcarriers
    amp_active = amplitude[:, active_indices, :]  # (n, 52, 2)
    phase_active = phase[:, active_indices, :]  # (n, 52, 2)

    # Create channels
    amp_ant1 = amp_active[:, :, 0]  # (n, 52)
    amp_ant2 = amp_active[:, :, 1]  # (n, 52)
    phase_ant1 = phase_active[:, :, 0]  # (n, 52)
    phase_ant2 = phase_active[:, :, 1]  # (n, 52)
    amp_diff = np.abs(amp_ant1 - amp_ant2)  # (n, 52)
    phase_diff = phase_ant1 - phase_ant2
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))  # (n, 52)

    # Stack into channels: (n, 6, 52)
    csi_2d = np.stack([amp_ant1, amp_ant2, phase_ant1, phase_ant2, amp_diff, phase_diff], axis=1)

    return csi_2d


# ===========================
# Dataset Classes
# ===========================

class CSIDataset(Dataset):
    """Dataset for MLP/ResNet models."""
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


class CSIDataset2D(Dataset):
    """Dataset for CNN models with 2D CSI representation."""
    def __init__(self, X_meta, X_csi_2d, y=None):
        self.X_meta = torch.FloatTensor(X_meta)
        self.X_csi_2d = torch.FloatTensor(X_csi_2d)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X_meta)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_meta[idx], self.X_csi_2d[idx], self.y[idx]
        return self.X_meta[idx], self.X_csi_2d[idx]


# ===========================
# Model Architectures
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
    """ResNet-style MLP classifier."""
    def __init__(self, input_dim, num_classes=10, hidden_dims=[1024, 512, 256, 128, 64],
                 dropout_base=0.45):
        super(CSIResNetClassifier, self).__init__()

        dropout_schedule = [dropout_base - i*0.05 for i in range(len(hidden_dims))]
        dropout_schedule = [max(0.15, d) for d in dropout_schedule]

        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_schedule[0])

        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            dropout = dropout_schedule[min(i+1, len(dropout_schedule) - 1)]
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout=dropout)
            )

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


class CSI1DCNN(nn.Module):
    """
    1D CNN that treats subcarriers as a sequence.
    Captures local frequency patterns that MLP might miss.
    """
    def __init__(self, meta_dim=5, num_classes=10):
        super(CSI1DCNN, self).__init__()

        # CNN for CSI data: input (batch, 6 channels, 52 subcarriers)
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv1d(6, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Second conv block
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 52 -> 26
            nn.Dropout(0.2),

            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 26 -> 13
            nn.Dropout(0.3),

            # Fourth conv block
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling -> (batch, 256, 1)
        )

        # MLP for metadata
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_meta, x_csi):
        # CNN path
        cnn_out = self.cnn(x_csi)  # (batch, 256, 1)
        cnn_out = cnn_out.squeeze(-1)  # (batch, 256)

        # Metadata path
        meta_out = self.meta_mlp(x_meta)  # (batch, 64)

        # Combine and classify
        combined = torch.cat([cnn_out, meta_out], dim=1)  # (batch, 320)
        out = self.classifier(combined)

        return out


# ===========================
# Training Utilities
# ===========================

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = torch.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1).mean()
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


def train_resnet_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        if mixup_alpha > 0 and np.random.rand() > 0.5:
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


def train_cnn_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_meta, X_csi, y_batch in loader:
        X_meta, X_csi, y_batch = X_meta.to(device), X_csi.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_meta, X_csi)
        loss = criterion(outputs, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), 100. * correct / total


def validate_resnet(model, loader, criterion, device):
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


def validate_cnn(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_meta, X_csi, y_batch in loader:
            X_meta, X_csi, y_batch = X_meta.to(device), X_csi.to(device), y_batch.to(device)
            outputs = model(X_meta, X_csi)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), 100. * correct / total


def get_resnet_predictions(model, loader, device):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                X_batch = batch[0].to(device)
            else:
                X_batch = batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def get_cnn_predictions(model, loader, device):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                X_meta, X_csi, _ = batch
            else:
                X_meta, X_csi = batch
            X_meta, X_csi = X_meta.to(device), X_csi.to(device)
            outputs = model(X_meta, X_csi)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


# ===========================
# Main Training
# ===========================

def main():
    print("="*70)
    print("ADVANCED ENSEMBLE: 5-FOLD CV + ARCHITECTURAL DIVERSITY")
    print("="*70)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test_nolabels.csv')

    X_full = train_df.drop(['position'], axis=1).values
    y_full = train_df['position'].values

    if X_full.shape[1] > 260:
        X_full = X_full[:, 1:]

    X_test = test_df.values
    if X_test.shape[1] > 260:
        X_test = X_test[:, 1:]

    print(f"Training samples: {len(X_full)}")
    print(f"Test samples: {len(X_test)}")

    # Prepare features
    X_meta = X_full[:, :5]
    X_csi_raw = X_full[:, 5:]

    X_test_meta = X_test[:, :5]
    X_test_csi_raw = X_test[:, 5:]

    print("\nApplying feature engineering...")
    X_csi_flat = extract_csi_features(X_csi_raw)
    X_csi_2d = extract_csi_2d(X_csi_raw)

    X_test_csi_flat = extract_csi_features(X_test_csi_raw)
    X_test_csi_2d = extract_csi_2d(X_test_csi_raw)

    print(f"Flat CSI features: {X_csi_flat.shape}")
    print(f"2D CSI features: {X_csi_2d.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 5 + X_csi_flat.shape[1]

    # 5-Fold Cross Validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_models = []
    all_oof_preds = np.zeros((len(X_full), 10))
    fold_accuracies = []

    # Training configurations
    configs = [
        {'type': 'resnet', 'seed': 42, 'name': 'ResNet_s42'},
        {'type': 'resnet', 'seed': 123, 'name': 'ResNet_s123'},
        {'type': 'cnn', 'seed': 42, 'name': 'CNN_s42'},
        {'type': 'cnn', 'seed': 123, 'name': 'CNN_s123'},
    ]

    print(f"\n{'='*70}")
    print(f"TRAINING {len(configs)} MODELS × {n_folds} FOLDS = {len(configs) * n_folds} TOTAL MODELS")
    print(f"{'='*70}")

    for config in configs:
        print(f"\n{'='*70}")
        print(f"MODEL: {config['name']}")
        print(f"{'='*70}")

        model_oof_preds = np.zeros((len(X_full), 10))
        model_test_preds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            print(f"\n--- Fold {fold+1}/{n_folds} ---")

            set_seed(config['seed'] + fold)

            # Split data
            X_train_meta = X_meta[train_idx]
            X_val_meta = X_meta[val_idx]

            # Normalize
            scaler_meta = StandardScaler()
            X_train_meta_norm = scaler_meta.fit_transform(X_train_meta)
            X_val_meta_norm = scaler_meta.transform(X_val_meta)
            X_test_meta_norm = scaler_meta.transform(X_test_meta)

            if config['type'] == 'resnet':
                X_train_csi = X_csi_flat[train_idx]
                X_val_csi = X_csi_flat[val_idx]

                scaler_csi = StandardScaler()
                X_train_csi_norm = scaler_csi.fit_transform(X_train_csi)
                X_val_csi_norm = scaler_csi.transform(X_val_csi)
                X_test_csi_norm = scaler_csi.transform(X_test_csi_flat)

                train_dataset = CSIDataset(X_train_meta_norm, X_train_csi_norm, y_full[train_idx])
                val_dataset = CSIDataset(X_val_meta_norm, X_val_csi_norm, y_full[val_idx])
                test_dataset = CSIDataset(X_test_meta_norm, X_test_csi_norm)

                model = CSIResNetClassifier(input_dim, num_classes=10).to(device)
                train_fn = train_resnet_epoch
                val_fn = validate_resnet
                pred_fn = get_resnet_predictions

            else:  # CNN
                X_train_csi = X_csi_2d[train_idx]
                X_val_csi = X_csi_2d[val_idx]

                # Normalize 2D CSI per channel
                n_channels = X_train_csi.shape[1]
                X_train_csi_norm = np.zeros_like(X_train_csi)
                X_val_csi_norm = np.zeros_like(X_val_csi)
                X_test_csi_norm = np.zeros_like(X_test_csi_2d)

                for c in range(n_channels):
                    mean = X_train_csi[:, c, :].mean()
                    std = X_train_csi[:, c, :].std() + 1e-8
                    X_train_csi_norm[:, c, :] = (X_train_csi[:, c, :] - mean) / std
                    X_val_csi_norm[:, c, :] = (X_val_csi[:, c, :] - mean) / std
                    X_test_csi_norm[:, c, :] = (X_test_csi_2d[:, c, :] - mean) / std

                train_dataset = CSIDataset2D(X_train_meta_norm, X_train_csi_norm, y_full[train_idx])
                val_dataset = CSIDataset2D(X_val_meta_norm, X_val_csi_norm, y_full[val_idx])
                test_dataset = CSIDataset2D(X_test_meta_norm, X_test_csi_norm)

                model = CSI1DCNN(meta_dim=5, num_classes=10).to(device)
                train_fn = train_cnn_epoch
                val_fn = validate_cnn
                pred_fn = get_cnn_predictions

            # DataLoaders
            batch_size = 256
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

            # Training setup
            criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-6)
            scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=150)

            # Train
            best_val_acc = 0
            patience_counter = 0
            best_model_state = None

            for epoch in range(150):
                train_loss, train_acc = train_fn(model, train_loader, criterion, optimizer, device, mixup_alpha=0.2)
                val_loss, val_acc = val_fn(model, val_loader, criterion, device)
                scheduler.step()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 30:
                        break

                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")

            print(f"  Best Val Acc: {best_val_acc:.2f}%")
            fold_accuracies.append(best_val_acc)

            # Load best model and get predictions
            model.load_state_dict(best_model_state)

            # OOF predictions
            val_probs = pred_fn(model, val_loader, device)
            model_oof_preds[val_idx] = val_probs

            # Test predictions
            test_probs = pred_fn(model, test_loader, device)
            model_test_preds.append(test_probs)

            # Save model
            model_name = f"{config['name']}_fold{fold+1}"
            torch.save(best_model_state, f'{model_name}.pth')
            all_models.append({
                'name': model_name,
                'type': config['type'],
                'fold': fold,
                'val_acc': best_val_acc
            })

        # Average test predictions across folds for this model type
        avg_test_preds = np.mean(model_test_preds, axis=0)
        all_models[-1]['test_preds'] = avg_test_preds

        # Add OOF to ensemble
        all_oof_preds += model_oof_preds / len(configs)

    # Calculate OOF accuracy
    oof_final_preds = np.argmax(all_oof_preds, axis=1)
    oof_accuracy = 100.0 * accuracy_score(y_full, oof_final_preds)

    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Mean Fold Accuracy: {np.mean(fold_accuracies):.2f}% (+/- {np.std(fold_accuracies):.2f}%)")
    print(f"OOF Ensemble Accuracy: {oof_accuracy:.2f}%")
    print(f"Improvement over previous (96.59%): {oof_accuracy - 96.59:.2f}%")
    print(f"Improvement over baseline (95.71%): {oof_accuracy - 95.71:.2f}%")

    # Generate final test predictions
    print(f"\n{'='*70}")
    print("GENERATING FINAL TEST PREDICTIONS")
    print(f"{'='*70}")

    # Collect all test predictions
    all_test_preds = []

    for config in configs:
        print(f"\nLoading {config['name']} models...")
        model_test_preds = []

        for fold in range(n_folds):
            model_name = f"{config['name']}_fold{fold+1}"

            set_seed(config['seed'] + fold)

            # Rebuild scalers and data (same process as training)
            train_idx = list(skf.split(X_full, y_full))[fold][0]

            scaler_meta = StandardScaler()
            scaler_meta.fit(X_meta[train_idx])
            X_test_meta_norm = scaler_meta.transform(X_test_meta)

            if config['type'] == 'resnet':
                scaler_csi = StandardScaler()
                scaler_csi.fit(X_csi_flat[train_idx])
                X_test_csi_norm = scaler_csi.transform(X_test_csi_flat)

                test_dataset = CSIDataset(X_test_meta_norm, X_test_csi_norm)
                model = CSIResNetClassifier(input_dim, num_classes=10).to(device)
                pred_fn = get_resnet_predictions
            else:
                X_train_csi = X_csi_2d[train_idx]
                X_test_csi_norm = np.zeros_like(X_test_csi_2d)
                for c in range(X_train_csi.shape[1]):
                    mean = X_train_csi[:, c, :].mean()
                    std = X_train_csi[:, c, :].std() + 1e-8
                    X_test_csi_norm[:, c, :] = (X_test_csi_2d[:, c, :] - mean) / std

                test_dataset = CSIDataset2D(X_test_meta_norm, X_test_csi_norm)
                model = CSI1DCNN(meta_dim=5, num_classes=10).to(device)
                pred_fn = get_cnn_predictions

            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

            model.load_state_dict(torch.load(f'{model_name}.pth', map_location=device))
            test_probs = pred_fn(model, test_loader, device)
            model_test_preds.append(test_probs)

        # Average across folds
        avg_preds = np.mean(model_test_preds, axis=0)
        all_test_preds.append(avg_preds)

    # Final ensemble: average all model predictions
    final_test_probs = np.mean(all_test_preds, axis=0)
    final_test_preds = np.argmax(final_test_probs, axis=1)

    # Save submission
    submission = pd.DataFrame({
        'id': range(len(final_test_preds)),
        'position': final_test_preds
    })

    submission.to_csv('submission_v5_advanced.csv', index=False)
    print(f"\nFinal predictions saved to 'submission_v5_advanced.csv'")
    print(f"Prediction distribution:\n{pd.Series(final_test_preds).value_counts().sort_index()}")

    # Save results
    results = {
        'oof_accuracy': float(oof_accuracy),
        'mean_fold_acc': float(np.mean(fold_accuracies)),
        'std_fold_acc': float(np.std(fold_accuracies)),
        'n_models': len(configs) * n_folds,
        'improvement_vs_baseline': float(oof_accuracy - 95.71),
        'improvement_vs_previous': float(oof_accuracy - 96.59)
    }

    with open('v5_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("ADVANCED ENSEMBLE COMPLETE!")
    print(f"{'='*70}")
    print(f"OOF Accuracy: {oof_accuracy:.2f}%")
    print(f"Total models trained: {len(configs) * n_folds}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
