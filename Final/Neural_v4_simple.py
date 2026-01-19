import json
import itertools
import os
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

# Feature Engineering Functions

def extract_csi_features(csi_data):
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

# Dataset Class

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


# Model Architecture

class ResidualBlock(nn.Module):
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
    def __init__(self, input_dim, num_classes=10, hidden_dims=None, dropout_schedule=None):
        super(CSIResNetClassifier, self).__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256, 128, 64]
        if dropout_schedule is None:
            dropout_schedule = [0.45, 0.40, 0.35, 0.30, 0.25, 0.2]

        self.hidden_dims = hidden_dims

        # Input projection to wider dimension
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_schedule[0])

        # Residual blocks with decreasing dimensions
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            dropout = dropout_schedule[min(i + 1, len(dropout_schedule) - 1)]
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout=dropout)
            )

        # Final classification layers
        self.fc1 = nn.Linear(hidden_dims[-1], 64)
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
        for res_block in self.res_blocks:
            x = res_block(x)

        # Final layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)

        x = self.fc_out(x)
        return x

# Training Utilities

def train_epoch(model, loader, criterion, optimizer, device):
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

    accuracy = 100. * correct / total
    return total_loss / len(loader), accuracy

# Data Loading and Preprocessing

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


# Grid Search Training Function


def train_with_config(config, train_loader, val_loader, input_dim, device, num_epochs=150, patience=30):
    set_seed(42)

    # Build dropout schedule
    dropout_base = config['dropout_base']
    hidden_dims = config['hidden_dims']
    dropout_schedule = [dropout_base - i * 0.05 for i in range(len(hidden_dims) + 1)]
    dropout_schedule = [max(0.15, d) for d in dropout_schedule]

    # Create model
    model = CSIResNetClassifier(
        input_dim=input_dim,
        num_classes=10,
        hidden_dims=hidden_dims,
        dropout_schedule=dropout_schedule
    ).to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Training loop
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        _, _ = train_epoch(model, train_loader, criterion, optimizer, device)
        _, val_acc = validate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_acc



# Check if best_hyperparams.json exists
if os.path.exists('best_hyperparams.json'):
    print("Found best_hyperparams.json, skipping grid search...")
    with open('best_hyperparams.json', 'r') as f:
        best_config_dict = json.load(f)
    best_lr = best_config_dict['lr']
    best_wd = best_config_dict['weight_decay']
    best_dropout = best_config_dict['dropout_base']
    best_hidden_dims = best_config_dict['hidden_dims']
    print(f"Loaded hyperparameters:")
    print(f"  Learning rate: {best_lr}")
    print(f"  Weight decay: {best_wd}")
    print(f"  Dropout base: {best_dropout}")
    print(f"  Architecture: {best_hidden_dims}")
else:
    # Grid Search
    print("HYPERPARAMETER GRID SEARCH")

    # Define parameter grid
    param_grid = {
        'lr': [0.0005, 0.001, 0.002],
        'weight_decay': [1e-6, 5e-6, 1e-5],
        'dropout_base': [0.35, 0.40, 0.45],
        'hidden_dims': [
            [512, 256, 128],
            [1024, 512, 256, 128],
            [1024, 512, 256, 128, 64],
        ]
    }

    print("\nParameter grid:")
    print(f"  lr: {param_grid['lr']}")
    print(f"  weight_decay: {param_grid['weight_decay']}")
    print(f"  dropout_base: {param_grid['dropout_base']}")
    print(f"  architectures: {len(param_grid['hidden_dims'])} configurations")

    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    print(f"\nTotal combinations to test: {total_combinations}")

    # Run grid search
    results = []
    config_id = 0

    for lr, wd, dropout, hidden_dims in itertools.product(
        param_grid['lr'],
        param_grid['weight_decay'],
        param_grid['dropout_base'],
        param_grid['hidden_dims']
    ):
        config_id += 1
        config = {
            'lr': lr,
            'weight_decay': wd,
            'dropout_base': dropout,
            'hidden_dims': hidden_dims
        }

        print(f"\n[{config_id}/{total_combinations}] lr={lr}, wd={wd}, dropout={dropout}, arch={hidden_dims}")

        val_acc = train_with_config(config, train_loader, val_loader, input_dim, device)

        results.append({
            'config_id': config_id,
            'lr': lr,
            'weight_decay': wd,
            'dropout_base': dropout,
            'hidden_dims': str(hidden_dims),
            'num_layers': len(hidden_dims),
            'val_acc': val_acc
        })

        print(f"  Val Accuracy: {val_acc:.2f}%")

    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_acc', ascending=False)
    results_df.to_csv('grid_search_results.csv', index=False)

    print("GRID SEARCH COMPLETE")
    print("\nTop 10 configurations:")
    print(results_df.head(10).to_string(index=False))

    best_config = results_df.iloc[0]
    print(f"BEST CONFIGURATION:")
    print(f"  Learning rate: {best_config['lr']}")
    print(f"  Weight decay: {best_config['weight_decay']}")
    print(f"  Dropout base: {best_config['dropout_base']}")
    print(f"  Architecture: {best_config['hidden_dims']}")
    print(f"  Validation Accuracy: {best_config['val_acc']:.2f}%")

    # Save best config
    best_config_dict = {
        'lr': float(best_config['lr']),
        'weight_decay': float(best_config['weight_decay']),
        'dropout_base': float(best_config['dropout_base']),
        'hidden_dims': eval(best_config['hidden_dims']),
        'val_acc': float(best_config['val_acc'])
    }
    with open('best_hyperparams.json', 'w') as f:
        json.dump(best_config_dict, f, indent=2)
    print("\nBest hyperparameters saved to best_hyperparams.json")

    # Plot Grid Search Results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # LR vs Accuracy
    axes[0, 0].scatter(results_df['lr'], results_df['val_acc'], alpha=0.7)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Learning Rate vs Accuracy')
    axes[0, 0].grid(True, alpha=0.3)

    # Weight Decay vs Accuracy
    axes[0, 1].scatter(results_df['weight_decay'], results_df['val_acc'], alpha=0.7)
    axes[0, 1].set_xlabel('Weight Decay')
    axes[0, 1].set_ylabel('Validation Accuracy (%)')
    axes[0, 1].set_title('Weight Decay vs Accuracy')
    axes[0, 1].grid(True, alpha=0.3)

    # Dropout vs Accuracy
    axes[1, 0].scatter(results_df['dropout_base'], results_df['val_acc'], alpha=0.7)
    axes[1, 0].set_xlabel('Dropout Base')
    axes[1, 0].set_ylabel('Validation Accuracy (%)')
    axes[1, 0].set_title('Dropout vs Accuracy')
    axes[1, 0].grid(True, alpha=0.3)

    # Top configs bar chart
    top_10 = results_df.head(10)
    axes[1, 1].barh(range(len(top_10)), top_10['val_acc'])
    axes[1, 1].set_yticks(range(len(top_10)))
    axes[1, 1].set_yticklabels([f"Config {int(x)}" for x in top_10['config_id']])
    axes[1, 1].set_xlabel('Validation Accuracy (%)')
    axes[1, 1].set_title('Top 10 Configurations')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grid_search_analysis.png', dpi=150)
    print('\nGrid search analysis saved to grid_search_analysis.png')

    # Extract best config values for final training
    best_lr = float(best_config['lr'])
    best_wd = float(best_config['weight_decay'])
    best_dropout = float(best_config['dropout_base'])
    best_hidden_dims = eval(best_config['hidden_dims'])

# Train Final Model with Best Config
print("TRAINING FINAL MODEL WITH BEST CONFIGURATION")

dropout_schedule = [best_dropout - i * 0.05 for i in range(len(best_hidden_dims) + 1)]
dropout_schedule = [max(0.15, d) for d in dropout_schedule]

set_seed(42)
model = CSIResNetClassifier(
    input_dim=input_dim,
    num_classes=10,
    hidden_dims=best_hidden_dims,
    dropout_schedule=dropout_schedule
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_wd)

print(f"Architecture: {best_hidden_dims}")
print(f"lr={best_lr}, weight_decay={best_wd}, dropout_base={best_dropout}")

# Training loop for final model
best_val_acc = 0
patience_counter = 0
num_epochs = 200
patience = 40

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model_final.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

print(f'\nFinal model best validation accuracy: {best_val_acc:.2f}%')

# Load best model for predictions
model.load_state_dict(torch.load('best_model_final.pth'))
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
    'ID': range(len(predictions)),
    'position': predictions
})

submission.to_csv('submission_v4_simple.csv', index=False)
print(f"Predictions saved to 'submission_v4.csv'")
print(f"Total predictions: {len(predictions)}")
print(f"Prediction distribution:\n{pd.Series(predictions).value_counts().sort_index()}")
