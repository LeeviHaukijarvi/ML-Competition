"""
Final optimized ensemble using best models from hyperparameter search.
Selects diverse top performers for maximum ensemble benefit.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
from scipy.stats import spearmanr
from itertools import combinations

# ===========================
# Core Components
# ===========================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


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
    def __init__(self, input_dim, num_classes=10, hidden_dims=[1024, 512, 256, 128, 64],
                 dropout_schedule=[0.4, 0.35, 0.3, 0.25, 0.2]):
        super(CSIResNetClassifier, self).__init__()

        self.hidden_dims = hidden_dims
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


def get_predictions(model, loader, device):
    """Get softmax probabilities for a dataset."""
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


def calculate_diversity(predictions_list, labels):
    """
    Calculate diversity metrics for ensemble members.
    Returns pairwise prediction correlation and disagreement rates.
    """
    n_models = len(predictions_list)

    # Convert probabilities to predictions
    preds_list = [np.argmax(p, axis=1) for p in predictions_list]

    # Pairwise correlation matrix
    correlation_matrix = np.zeros((n_models, n_models))
    disagreement_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                correlation_matrix[i, j] = 1.0
                disagreement_matrix[i, j] = 0.0
            else:
                # Correlation of predictions
                corr, _ = spearmanr(preds_list[i], preds_list[j])
                correlation_matrix[i, j] = corr

                # Disagreement rate
                disagreement = np.mean(preds_list[i] != preds_list[j])
                disagreement_matrix[i, j] = disagreement

    return correlation_matrix, disagreement_matrix, preds_list


def select_diverse_models(results_df, predictions_list, labels, top_n=20, select_n=12):
    """
    Select diverse models from top performers using greedy selection.

    Strategy:
    1. Start with the best model
    2. Iteratively add models that maximize diversity while maintaining high accuracy
    """

    # Get top N models by accuracy
    top_models_df = results_df.head(top_n).copy()
    top_indices = top_models_df.index.tolist()

    print(f"\nSelecting {select_n} diverse models from top {top_n} performers...")
    print(f"Accuracy range: {top_models_df['val_acc'].min():.2f}% - {top_models_df['val_acc'].max():.2f}%")

    # Calculate diversity metrics
    top_predictions = [predictions_list[i] for i in top_indices]
    corr_matrix, disagree_matrix, _ = calculate_diversity(top_predictions, labels)

    # Greedy selection
    selected_indices = [0]  # Start with best model

    for _ in range(select_n - 1):
        best_score = -np.inf
        best_idx = None

        for idx in range(len(top_indices)):
            if idx in selected_indices:
                continue

            # Calculate average diversity with selected models
            avg_disagreement = np.mean([disagree_matrix[idx, s] for s in selected_indices])

            # Score combines accuracy and diversity
            accuracy_score = top_models_df.iloc[idx]['val_acc']
            diversity_score = avg_disagreement * 100  # Scale to match accuracy

            # Weighted combination (70% accuracy, 30% diversity)
            combined_score = 0.7 * accuracy_score + 0.3 * diversity_score

            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx

        selected_indices.append(best_idx)

    selected_configs = [top_indices[i] for i in selected_indices]

    print(f"\nSelected {len(selected_configs)} models:")
    for i, idx in enumerate(selected_configs):
        config_id = top_models_df.iloc[idx]['config_id']
        acc = top_models_df.iloc[idx]['val_acc']
        print(f"  {i+1}. Config {int(config_id)}: {acc:.2f}%")

    return selected_configs, selected_indices


def main():
    print("="*70)
    print("FINAL OPTIMIZED ENSEMBLE FROM HYPERPARAMETER SEARCH")
    print("="*70)

    # Load hyperparameter search results
    results_df = pd.read_csv('hyperopt_results.csv')
    print(f"\nLoaded {len(results_df)} hyperopt configurations")
    print(f"Best accuracy: {results_df['val_acc'].max():.2f}%")
    print(f"Top 10 range: {results_df.head(10)['val_acc'].min():.2f}% - {results_df.head(10)['val_acc'].max():.2f}%")

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

    # Train/val split (same as hyperopt)
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
    val_dataset = CSIDataset(X_val_meta_split, X_val_csi_split, y_val_split)
    test_dataset = CSIDataset(X_test_meta_norm, X_test_csi_norm)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 5 + X_train_csi_features.shape[1]

    # Load all models and get predictions
    print("\nLoading models and generating predictions...")
    all_val_predictions = []
    architecture = [1024, 512, 256, 128, 64]

    for idx, row in results_df.iterrows():
        config_id = int(row['config_id'])
        model_path = f'hyperopt_model_{config_id}.pth'

        # Build dropout schedule
        dropout_base = row['dropout_base']
        dropout_schedule = [dropout_base, dropout_base-0.05, dropout_base-0.1,
                           dropout_base-0.15, dropout_base-0.2, 0.2]
        dropout_schedule = [max(0.15, d) for d in dropout_schedule]

        # Create and load model
        model = CSIResNetClassifier(
            input_dim=input_dim,
            num_classes=10,
            hidden_dims=architecture,
            dropout_schedule=dropout_schedule
        ).to(device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            val_preds = get_predictions(model, val_loader, device)
            all_val_predictions.append(val_preds)

            if (idx + 1) % 10 == 0:
                print(f"  Loaded {idx + 1}/{len(results_df)} models...")
        except FileNotFoundError:
            print(f"  Warning: Model {model_path} not found, skipping...")
            all_val_predictions.append(None)

    print(f"Successfully loaded {sum(p is not None for p in all_val_predictions)} models")

    # Remove None entries
    valid_indices = [i for i, p in enumerate(all_val_predictions) if p is not None]
    all_val_predictions = [all_val_predictions[i] for i in valid_indices]
    results_df = results_df.iloc[valid_indices].reset_index(drop=True)

    # Select diverse models
    selected_configs, selected_indices = select_diverse_models(
        results_df, all_val_predictions, y_val_split,
        top_n=20, select_n=12
    )

    # Create ensemble from selected models
    print(f"\n{'='*70}")
    print("CREATING ENSEMBLE FROM SELECTED MODELS")
    print(f"{'='*70}")

    selected_val_preds = [all_val_predictions[i] for i in selected_indices]

    # Average predictions
    ensemble_val_probs = np.mean(selected_val_preds, axis=0)
    ensemble_val_preds = np.argmax(ensemble_val_probs, axis=1)

    # Calculate ensemble accuracy
    ensemble_val_acc = 100.0 * accuracy_score(y_val_split, ensemble_val_preds)

    print(f"\nEnsemble Validation Accuracy: {ensemble_val_acc:.2f}%")
    print(f"Improvement over best single model: {ensemble_val_acc - results_df['val_acc'].max():.2f}%")
    print(f"Improvement over baseline (95.71%): {ensemble_val_acc - 95.71:.2f}%")
    print(f"Improvement over previous ensemble (96.48%): {ensemble_val_acc - 96.48:.2f}%")

    # Compare with simple averaging of top N
    for n in [5, 10, 15, 20]:
        if n <= len(all_val_predictions):
            top_n_probs = np.mean(all_val_predictions[:n], axis=0)
            top_n_preds = np.argmax(top_n_probs, axis=1)
            top_n_acc = 100.0 * accuracy_score(y_val_split, top_n_preds)
            print(f"  Top-{n} simple average: {top_n_acc:.2f}%")

    # Generate test predictions
    print(f"\n{'='*70}")
    print("GENERATING TEST PREDICTIONS")
    print(f"{'='*70}")

    test_predictions = []

    for idx in selected_configs:
        row = results_df.iloc[idx]
        config_id = int(row['config_id'])
        model_path = f'hyperopt_model_{config_id}.pth'

        dropout_base = row['dropout_base']
        dropout_schedule = [dropout_base, dropout_base-0.05, dropout_base-0.1,
                           dropout_base-0.15, dropout_base-0.2, 0.2]
        dropout_schedule = [max(0.15, d) for d in dropout_schedule]

        model = CSIResNetClassifier(
            input_dim=input_dim,
            num_classes=10,
            hidden_dims=architecture,
            dropout_schedule=dropout_schedule
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        test_preds = get_predictions(model, test_loader, device)
        test_predictions.append(test_preds)

    # Average test predictions
    ensemble_test_probs = np.mean(test_predictions, axis=0)
    ensemble_test_preds = np.argmax(ensemble_test_probs, axis=1)

    # Create submission
    submission = pd.DataFrame({
        'id': range(len(ensemble_test_preds)),
        'position': ensemble_test_preds
    })

    submission.to_csv('submission_v4_final.csv', index=False)
    print(f"\nFinal ensemble predictions saved to 'submission_v4_final.csv'")
    print(f"Prediction distribution:\n{pd.Series(ensemble_test_preds).value_counts().sort_index()}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Model selection
    selected_df = results_df.iloc[selected_indices]
    axes[0, 0].barh(range(len(selected_df)), selected_df['val_acc'])
    axes[0, 0].set_yticks(range(len(selected_df)))
    axes[0, 0].set_yticklabels([f"Config {int(x)}" for x in selected_df['config_id']])
    axes[0, 0].set_xlabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Selected Models for Ensemble')
    axes[0, 0].axvline(x=96.48, color='r', linestyle='--', alpha=0.5, label='Previous Best')
    axes[0, 0].axvline(x=ensemble_val_acc, color='g', linestyle='--', label='Ensemble')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Diversity heatmap
    selected_val_preds = [all_val_predictions[i] for i in selected_indices]
    _, disagree_matrix, _ = calculate_diversity(selected_val_preds, y_val_split)

    im = axes[0, 1].imshow(disagree_matrix, cmap='YlOrRd', aspect='auto')
    axes[0, 1].set_xticks(range(len(selected_indices)))
    axes[0, 1].set_yticks(range(len(selected_indices)))
    axes[0, 1].set_xticklabels([f"M{i+1}" for i in range(len(selected_indices))])
    axes[0, 1].set_yticklabels([f"M{i+1}" for i in range(len(selected_indices))])
    # Add text annotations
    for i in range(len(selected_indices)):
        for j in range(len(selected_indices)):
            axes[0, 1].text(j, i, f'{disagree_matrix[i, j]:.3f}', ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_title('Model Disagreement Matrix\n(Higher = More Diverse)')

    # Ensemble size vs accuracy
    ensemble_sizes = range(1, min(21, len(all_val_predictions) + 1))
    ensemble_accs = []

    for size in ensemble_sizes:
        probs = np.mean(all_val_predictions[:size], axis=0)
        preds = np.argmax(probs, axis=1)
        acc = 100.0 * accuracy_score(y_val_split, preds)
        ensemble_accs.append(acc)

    axes[1, 0].plot(ensemble_sizes, ensemble_accs, marker='o')
    axes[1, 0].axhline(y=96.48, color='r', linestyle='--', alpha=0.5, label='Previous Best')
    axes[1, 0].axvline(x=len(selected_indices), color='g', linestyle='--', alpha=0.5, label='Selected Size')
    axes[1, 0].set_xlabel('Ensemble Size')
    axes[1, 0].set_ylabel('Validation Accuracy (%)')
    axes[1, 0].set_title('Ensemble Size vs Accuracy (Top-N Simple Average)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Confusion analysis - compare best single vs ensemble
    best_single_preds = np.argmax(all_val_predictions[0], axis=1)

    # Find where ensemble corrects single model errors
    single_correct = (best_single_preds == y_val_split)
    ensemble_correct = (ensemble_val_preds == y_val_split)

    corrected = (~single_correct) & ensemble_correct
    new_errors = single_correct & (~ensemble_correct)

    improvement_data = {
        'Both Correct': np.sum(single_correct & ensemble_correct),
        'Ensemble Fixed': np.sum(corrected),
        'Ensemble Broke': np.sum(new_errors),
        'Both Wrong': np.sum((~single_correct) & (~ensemble_correct))
    }

    axes[1, 1].bar(improvement_data.keys(), improvement_data.values())
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_title('Ensemble vs Best Single Model')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add numbers on bars
    for i, (k, v) in enumerate(improvement_data.items()):
        axes[1, 1].text(i, v + 5, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('final_ensemble_analysis.png', dpi=150)
    print('\nFinal ensemble analysis saved to final_ensemble_analysis.png')

    # Save ensemble details
    ensemble_info = {
        'n_models': len(selected_indices),
        'selected_configs': [int(x) for x in selected_df['config_id'].tolist()],
        'individual_accs': selected_df['val_acc'].tolist(),
        'ensemble_val_acc': float(ensemble_val_acc),
        'best_single_acc': float(results_df['val_acc'].max()),
        'improvement_vs_baseline': float(ensemble_val_acc - 95.71),
        'improvement_vs_previous': float(ensemble_val_acc - 96.48),
        'samples_corrected': int(np.sum(corrected)),
        'samples_broken': int(np.sum(new_errors))
    }

    with open('final_ensemble_info.json', 'w') as f:
        json.dump(ensemble_info, f, indent=2)

    print('\nEnsemble details saved to final_ensemble_info.json')

    print(f"\n{'='*70}")
    print("FINAL ENSEMBLE COMPLETE!")
    print(f"{'='*70}")
    print(f"Validation Accuracy: {ensemble_val_acc:.2f}%")
    print(f"Models in ensemble: {len(selected_indices)}")
    print(f"Samples corrected vs best single: {np.sum(corrected)}")
    print(f"Samples broken vs best single: {np.sum(new_errors)}")
    print(f"Net improvement: {np.sum(corrected) - np.sum(new_errors)} samples")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
