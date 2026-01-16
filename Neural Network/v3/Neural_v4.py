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
from typing import List, Tuple

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ===========================
# Feature Engineering Functions
# ===========================

def extract_csi_features(csi_data):
    """
    Extract enhanced features from raw CSI I/Q data.

    Args:
        csi_data: numpy array of shape (n_samples, 256) containing I/Q pairs
                  64 subcarriers × 2 antennas × 2 (I/Q)

    Returns:
        numpy array with engineered features
    """
    n_samples = csi_data.shape[0]

    # Inactive subcarrier indices (guard bands and DC)
    # For 64-subcarrier OFDM: indices 0, 27-37 are typically null/pilot
    inactive_indices = [0] + list(range(27, 38))
    active_indices = [i for i in range(64) if i not in inactive_indices]

    # Reshape to (n_samples, 64 subcarriers, 2 antennas, 2 I/Q)
    csi_reshaped = csi_data.reshape(n_samples, 64, 2, 2)

    # Extract I and Q components
    I = csi_reshaped[:, :, :, 0]  # (n_samples, 64, 2)
    Q = csi_reshaped[:, :, :, 1]  # (n_samples, 64, 2)

    # Compute amplitude and phase
    amplitude = np.sqrt(I**2 + Q**2)  # (n_samples, 64, 2)
    phase = np.arctan2(Q, I)  # (n_samples, 64, 2)

    # Filter to active subcarriers only
    amplitude_active = amplitude[:, active_indices, :]  # (n_samples, 52, 2)
    phase_active = phase[:, active_indices, :]  # (n_samples, 52, 2)

    # Flatten amplitude and phase for active subcarriers
    amp_features = amplitude_active.reshape(n_samples, -1)  # (n_samples, 104)
    phase_features = phase_active.reshape(n_samples, -1)  # (n_samples, 104)

    # Cross-antenna features (amplitude and phase differences)
    amp_diff = np.abs(amplitude_active[:, :, 0] - amplitude_active[:, :, 1])  # (n_samples, 52)

    # Phase difference (handling wraparound)
    phase_diff = phase_active[:, :, 0] - phase_active[:, :, 1]
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))  # Normalize to [-π, π]

    # Subcarrier band statistics
    # Divide 52 active subcarriers into low/mid/high bands
    n_active = len(active_indices)
    band_size = n_active // 3

    low_band = amplitude_active[:, :band_size, :]
    mid_band = amplitude_active[:, band_size:2*band_size, :]
    high_band = amplitude_active[:, 2*band_size:, :]

    # Statistics for each band and antenna
    band_stats = []
    for band in [low_band, mid_band, high_band]:
        for antenna_idx in range(2):
            band_stats.append(np.mean(band[:, :, antenna_idx], axis=1))  # mean
            band_stats.append(np.std(band[:, :, antenna_idx], axis=1))   # std
            band_stats.append(np.max(band[:, :, antenna_idx], axis=1))   # max

    band_stats = np.column_stack(band_stats)  # (n_samples, 18)

    # Combine all features
    features = np.concatenate([
        amp_features,      # 104 features
        phase_features,    # 104 features
        amp_diff,          # 52 features
        phase_diff,        # 52 features
        band_stats         # 18 features
    ], axis=1)

    return features  # (n_samples, 330)


# ===========================
# Dataset Class
# ===========================

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
# Model Architecture
# ===========================

class ResidualBlock(nn.Module):
    """Residual block with two linear layers, batch norm, and skip connection."""

    def __init__(self, in_features, out_features, dropout=0.3):
        super(ResidualBlock, self).__init__()

        # Main path
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # Skip connection (projection if dimensions don't match)
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
    """
    Residual network for CSI classification with optional attention.

    Architecture: input → ResBlock(1024, 512) → ResBlock(512, 256) →
                  ResBlock(256, 128) → Linear(128, 64) → Linear(64, 10)
    """

    def __init__(self, input_dim, num_classes=10, use_attention=False):
        super(CSIResNetClassifier, self).__init__()

        self.use_attention = use_attention

        # Input projection to wider dimension
        self.input_proj = nn.Linear(input_dim, 1024)
        self.input_bn = nn.BatchNorm1d(1024)
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(0.4)

        # Residual blocks with decreasing dimensions
        self.res1 = ResidualBlock(1024, 512, dropout=0.35)
        self.res2 = ResidualBlock(512, 256, dropout=0.3)
        self.res3 = ResidualBlock(256, 128, dropout=0.25)

        # Optional attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Final classification layers
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        x = self.input_dropout(x)

        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        # Optional attention
        if self.use_attention:
            attention_weights = torch.softmax(self.attention(x), dim=1)
            x = x * attention_weights

        # Final layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)

        x = self.fc_out(x)
        return x


# ===========================
# Training Utilities
# ===========================

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
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
    """Compute loss for mixup samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = torch.log_softmax(preds, dim=-1)

        # Smooth labels
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
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + (self.base_lr - self.lr_min) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.2):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Apply mixup augmentation
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
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


# ===========================
# Data Loading and Preprocessing
# ===========================

print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_nolabels.csv')

# Separate features and labels
X_train_full = train_df.drop(['position'], axis=1).values
y_train_full = train_df['position'].values

# Remove row ID if it exists
if X_train_full.shape[1] > 260:
    X_train_full = X_train_full[:, 1:]

X_test = test_df.values
if X_test.shape[1] > 260:
    X_test = X_test[:, 1:]

print(f"Training data shape: {X_train_full.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y_train_full))}")

# Split into metadata and CSI features
# First 5 columns are metadata: timestamp, seq_ctrl, aoa, rssi1, rssi2
X_train_meta = X_train_full[:, :5]
X_train_csi = X_train_full[:, 5:]

X_test_meta = X_test[:, :5]
X_test_csi = X_test[:, 5:]

print("\nApplying feature engineering to CSI data...")
X_train_csi_features = extract_csi_features(X_train_csi)
X_test_csi_features = extract_csi_features(X_test_csi)

print(f"Engineered CSI features shape: {X_train_csi_features.shape}")

# Split training data for validation
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

# Normalize metadata features separately
scaler_meta = StandardScaler()
X_train_meta_split = scaler_meta.fit_transform(X_train_meta_split)
X_val_meta_split = scaler_meta.transform(X_val_meta_split)
X_test_meta_norm = scaler_meta.transform(X_test_meta)

# Normalize CSI features
scaler_csi = StandardScaler()
X_train_csi_split = scaler_csi.fit_transform(X_train_csi_split)
X_val_csi_split = scaler_csi.transform(X_val_csi_split)
X_test_csi_norm = scaler_csi.transform(X_test_csi_features)

# Create datasets and dataloaders
batch_size = 256
train_dataset = CSIDataset(X_train_meta_split, X_train_csi_split, y_train_split)
val_dataset = CSIDataset(X_val_meta_split, X_val_csi_split, y_val_split)
test_dataset = CSIDataset(X_test_meta_norm, X_test_csi_norm)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Input dimension
input_dim = 5 + X_train_csi_features.shape[1]  # metadata + CSI features
print(f"Total input features: {input_dim}")

# ===========================
# Single Model Training
# ===========================

print("\n" + "="*60)
print("TRAINING SINGLE MODEL WITH ALL IMPROVEMENTS")
print("="*60)

# Model configuration
model_config = {
    'input_dim': input_dim,
    'num_classes': 10,
    'use_attention': False
}

model = CSIResNetClassifier(**model_config).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training configuration
lr = 0.001
weight_decay = 1e-5
num_epochs = 200
warmup_epochs = 10
patience = 40

criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs)

# Training loop
best_val_acc = 0
patience_counter = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"\nTraining configuration:")
print(f"  Learning rate: {lr}")
print(f"  Weight decay: {weight_decay}")
print(f"  Batch size: {batch_size}")
print(f"  Max epochs: {num_epochs}")
print(f"  Warmup epochs: {warmup_epochs}")
print(f"  Early stopping patience: {patience}")
print(f"  Mixup alpha: 0.2")
print(f"  Label smoothing: 0.1")

print("\nStarting training...")
for epoch in range(num_epochs):
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                       use_mixup=True, mixup_alpha=0.2)

    # Validation
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Learning rate scheduling
    current_lr = scheduler.step()

    # Record history
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch < 10:
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  LR: {current_lr:.6f}')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Early stopping and model saving
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model_v4_single.pth')
        patience_counter = 0
        print(f'  ✓ New best model! Val Acc: {val_acc:.2f}%')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

print(f'\n{"="*60}')
print(f'Single model training complete!')
print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
print(f'Improvement over baseline (95.71%): {best_val_acc - 95.71:.2f}%')
print(f'{"="*60}')

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc', alpha=0.8)
plt.plot(val_accs, label='Val Acc', alpha=0.8)
plt.axhline(y=95.71, color='r', linestyle='--', label='Baseline (95.71%)', alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_v4_single.png', dpi=150)
print('\nTraining curves saved to training_history_v4_single.png')

# Save single model predictions for later ensemble
model.load_state_dict(torch.load('best_model_v4_single.pth'))
model.eval()

single_model_val_preds = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        single_model_val_preds.append(probs.cpu().numpy())

single_model_val_preds = np.vstack(single_model_val_preds)
np.save('single_model_val_probs.npy', single_model_val_preds)

print(f'\nSingle model complete. Proceeding to ensemble training if accuracy target not met...')
