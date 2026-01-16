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
import copy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------
# Dataset Class
# -----------------------------
class CSIDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# -----------------------------
# Model Definition
# -----------------------------
class CSIClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(CSIClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_nolabels.csv')

X_train_full = train_df.drop(['position'], axis=1).values
y_train_full = train_df['position'].values

if X_train_full.shape[1] > 260:
    X_train_full = X_train_full[:, 1:]

X_test = test_df.values
if X_test.shape[1] > 260:
    X_test = X_test[:, 1:]

print(f"Training data shape: {X_train_full.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y_train_full))}")

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Create datasets and dataloaders
batch_size = 128
train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CSIDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(CSIDataset(X_test), batch_size=batch_size, shuffle=False)

# -----------------------------
# Device
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

input_dim = X_train.shape[1]
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Training and Validation Functions
# -----------------------------
def train_epoch(model, loader, criterion, optimizer):
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
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
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

# -----------------------------
# Grid Search
# -----------------------------
param_grid = {
    'lr': [0.0005, 0.0008, 0.001],
    'weight_decay': [0, 1e-5, 1e-4]
}

best_val_acc_overall = 0
best_params_overall = None
best_model_state = None

for lr, wd in itertools.product(param_grid['lr'], param_grid['weight_decay']):
    print(f"\n=== Testing lr={lr}, weight_decay={wd} ===")
    model = CSIClassifier(input_dim, num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_acc = 0
    patience_counter = 0
    max_patience = 20
    num_epochs = 100

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break

    print(f"Validation Accuracy for lr={lr}, wd={wd}: {best_val_acc:.2f}%")

    if best_val_acc > best_val_acc_overall:
        best_val_acc_overall = best_val_acc
        best_params_overall = {'lr': lr, 'weight_decay': wd}
        best_model_state = copy.deepcopy(model.state_dict())

print("\n=== Grid Search Complete ===")
print(f"Best Validation Accuracy: {best_val_acc_overall:.2f}%")
print(f"Best Hyperparameters: {best_params_overall}")

# -----------------------------
# Load Best Model
# -----------------------------
model.load_state_dict(best_model_state)
torch.save(best_model_state, 'best_model_gridsearch.pth')

# -----------------------------
# Plot Training History
# -----------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# -----------------------------
# Predictions on Test Set
# -----------------------------
print("\nMaking predictions on test set...")
model.eval()
predictions = []

with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = outputs.max(1)
        predictions.extend(predicted.cpu().numpy())

submission = pd.DataFrame({
    'id': range(len(predictions)),
    'position': predictions
})

submission.to_csv('submission.csv', index=False)
print("Predictions saved to 'submission.csv'")
print(f"Total predictions: {len(predictions)}")
print(f"Prediction distribution:\n{pd.Series(predictions).value_counts().sort_index()}")
