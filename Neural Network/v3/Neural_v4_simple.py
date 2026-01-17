"""
Simplified Neural Network for Wi-Fi CSI Localization.

This version uses:
- Amplitude and phase extraction from I/Q values
- Inactive subcarrier removal
- ResNet architecture with skip connections
- Standard cross-entropy loss
- No mixup, no label smoothing, no ensemble
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
    Extract features from raw CSI I/Q data.

    Only includes:
    1) Amplitude and phase extraction
    2) Inactive subcarrier removal

    Args:
        csi_data: numpy array of shape (n_samples, 256) containing I/Q pairs
                  64 subcarriers x 2 antennas x 2 (I/Q)

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

    # Combine features (amplitude + phase only)
    features = np.concatenate([
        amp_features,      # 104 features
        phase_features,    # 104 features
    ], axis=1)

    return features  # (n_samples, 208)


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
    Residual network for CSI classification.

    Architecture: input -> ResBlock(1024, 512) -> ResBlock(512, 256) ->
                  ResBlock(256, 128) -> Linear(128, 64) -> Linear(64, 10)
    """

    def __init__(self, input_dim, num_classes=10):
        super(CSIResNetClassifier, self).__init__()

        # Input projection to wider dimension
        self.input_proj = nn.Linear(input_dim, 1024)
        self.input_bn = nn.BatchNorm1d(1024)
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(0.4)

        # Residual blocks with decreasing dimensions
        self.res1 = ResidualBlock(1024, 512, dropout=0.35)
        self.res2 = ResidualBlock(512, 256, dropout=0.3)
        self.res3 = ResidualBlock(256, 128, dropout=0.25)

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

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch with standard cross-entropy (no mixup)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

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
print("  - Amplitude and phase extraction")
print("  - Inactive subcarrier removal (12 subcarriers)")
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Input dimension
input_dim = 5 + X_train_csi_features.shape[1]  # metadata + CSI features
print(f"Total input features: {input_dim}")

# ===========================
# Model Training
# ===========================

print("\n" + "="*60)
print("TRAINING SIMPLIFIED MODEL")
print("="*60)
print("Features: Amplitude + Phase extraction, Inactive subcarrier removal")
print("No: Cross-antenna features, Band statistics, Mixup, Label smoothing")

# Model configuration
model_config = {
    'input_dim': input_dim,
    'num_classes': 10,
}

model = CSIResNetClassifier(**model_config).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training configuration
lr = 0.001
weight_decay = 1e-5
num_epochs = 200
patience = 40

# Standard cross-entropy loss (no label smoothing)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
print(f"  Early stopping patience: {patience}")
print(f"  Loss: CrossEntropyLoss (standard)")

print("\nStarting training...")
for epoch in range(num_epochs):
    # Training (no mixup)
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # Validation
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Record history
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Print progress
    if (epoch + 1) % 5 == 0 or epoch < 10:
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Early stopping and model saving
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model_v4_simple.pth')
        patience_counter = 0
        print(f'  New best model! Val Acc: {val_acc:.2f}%')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

print(f'\n{"="*60}')
print(f'Training complete!')
print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
print(f'Comparison to baseline (95.71%): {best_val_acc - 95.71:+.2f}%')
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
plt.savefig('training_history_v4_simple.png', dpi=150)
print('\nTraining curves saved to training_history_v4_simple.png')

# Load best model and make predictions
model.load_state_dict(torch.load('best_model_v4_simple.pth'))
model.eval()

print("\nMaking predictions on test set...")
predictions = []

with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = outputs.max(1)
        predictions.extend(predicted.cpu().numpy())

# Create submission file
submission = pd.DataFrame({
    'id': range(len(predictions)),
    'position': predictions
})

submission.to_csv('submission_v4_simple.csv', index=False)
print(f"Predictions saved to 'submission_v4_simple.csv'")
print(f"Total predictions: {len(predictions)}")
print(f"Prediction distribution:\n{pd.Series(predictions).value_counts().sort_index()}")
