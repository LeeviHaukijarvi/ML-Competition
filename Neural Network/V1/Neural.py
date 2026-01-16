import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("Wi-Fi CSI Position Classification with Neural Networks")
print("="*60)

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1] Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_nolabels.csv')

# Separate features and labels
X_train_full = train_df.drop(['position'], axis=1).values
y_train_full = train_df['position'].values

# Remove row ID (first column) if it exists
if X_train_full.shape[1] > 260:
    X_train_full = X_train_full[:, 1:]

X_test = test_df.values
if X_test.shape[1] > 260:
    X_test = X_test[:, 1:]

print(f"Training data shape: {X_train_full.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of classes: {len(np.unique(y_train_full))}")
print(f"Class distribution:\n{pd.Series(y_train_full).value_counts().sort_index()}")

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)

print(f"\nSplit sizes:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")

# ============================================================================
# PART 2: BASELINE MODEL - SIMPLE MLP
# ============================================================================

print("\n" + "="*60)
print("PART 2: Training Baseline MLP Model")
print("="*60)

# Create a pipeline with scaling and MLP classifier
baseline_model = Pipeline([
    ('scaling', StandardScaler()),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        verbose=False
    ))
])

print("\nTraining baseline model...")
baseline_model.fit(X_train, y_train)

# Predictions
pred_train = baseline_model.predict(X_train)
pred_val = baseline_model.predict(X_val)

# Evaluation
train_acc = accuracy_score(y_train, pred_train)
val_acc = accuracy_score(y_val, pred_val)

print("\n--- Baseline Model Results ---")
print(f"Training accuracy: {train_acc:.4f}")
print(f"Training error:    {1 - train_acc:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")
print(f"Validation error:    {1 - val_acc:.4f}")

# Confusion matrices
train_cmat = confusion_matrix(y_train, pred_train)
val_cmat = confusion_matrix(y_val, pred_val)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(train_cmat, square=True, annot=True, cbar=False, fmt="d", ax=axes[0])
axes[0].set_title('Confusion Matrix - Training')
axes[0].set_xlabel('Predicted value')
axes[0].set_ylabel('True value')

sns.heatmap(val_cmat, square=True, annot=True, cbar=False, fmt="d", ax=axes[1])
axes[1].set_title('Confusion Matrix - Validation')
axes[1].set_xlabel('Predicted value')
axes[1].set_ylabel('True value')

plt.tight_layout()
plt.savefig('baseline_confusion_matrices.png', dpi=150)
plt.show()

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(baseline_model.named_steps['clf'].loss_curve_)
plt.title('Baseline Model - Training Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.savefig('baseline_loss_curve.png', dpi=150)
plt.show()

print(f"\nTraining set loss: {baseline_model.named_steps['clf'].loss_:.4f}")
print(f"Best loss: {baseline_model.named_steps['clf'].best_loss_:.4f}")

# ============================================================================
# PART 3: HYPERPARAMETER SEARCH
# ============================================================================

print("\n" + "="*60)
print("PART 3: Hyperparameter Search with GridSearchCV")
print("="*60)

# Define model2 pipeline
model2 = Pipeline([
    ('scaling', StandardScaler()),
    ('clf', MLPClassifier(max_iter=200, random_state=42, verbose=False))
])

# Define parameter grid
param_grid = {
    'clf__hidden_layer_sizes': [(100,), (200,), (100, 100), (200, 100)],
    'clf__activation': ['relu', 'tanh'],
    'clf__solver': ['adam', 'sgd'],
    'clf__alpha': [0.0001, 0.001],
    'clf__learning_rate_init': [0.001, 0.01],
    'clf__learning_rate': ['constant', 'adaptive']
}

print("\nParameter grid:")
for key, value in param_grid.items():
    print(f"  {key}: {value}")

total_combinations = 1
for value in param_grid.values():
    total_combinations *= len(value)
print(f"\nTotal combinations to test: {total_combinations}")

# Create ShuffleSplit for validation
cv = ShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(
    model2,
    param_grid,
    cv=cv,
    verbose=2,
    n_jobs=-1,
    scoring='accuracy'
)

# Fit GridSearchCV
print("\nStarting GridSearchCV (this may take a while)...")
grid_search.fit(X_train, y_train)

print("\n" + "="*60)
print("GridSearchCV Results")
print("="*60)
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Convert results to DataFrame
cv_results = pd.DataFrame(grid_search.cv_results_)

# Sort by rank and display top 10 configurations
top_results = cv_results.sort_values('rank_test_score').head(10)[
    ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
]

print("\nTop 10 Configurations:")
print(top_results.to_string(index=False))

# Save full results
cv_results.to_csv('grid_search_results.csv', index=False)
print("\nFull grid search results saved to 'grid_search_results.csv'")

# ============================================================================
# PART 4: EVALUATE BEST MODEL
# ============================================================================

print("\n" + "="*60)
print("PART 4: Evaluating Best Model")
print("="*60)

best_model = grid_search.best_estimator_

# Predictions on train and validation sets
pred_train_best = best_model.predict(X_train)
pred_val_best = best_model.predict(X_val)

train_acc_best = accuracy_score(y_train, pred_train_best)
val_acc_best = accuracy_score(y_val, pred_val_best)

print("\n--- Best Model Performance ---")
print(f"Training accuracy:   {train_acc_best:.4f}")
print(f"Training error:      {1 - train_acc_best:.4f}")
print(f"Validation accuracy: {val_acc_best:.4f}")
print(f"Validation error:    {1 - val_acc_best:.4f}")

print("\nTRAINING SET - Classification Report:")
print(classification_report(y_train, pred_train_best))

print("\nVALIDATION SET - Classification Report:")
print(classification_report(y_val, pred_val_best))

# Confusion matrices for best model
train_cmat_best = confusion_matrix(y_train, pred_train_best)
val_cmat_best = confusion_matrix(y_val, pred_val_best)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(train_cmat_best, square=True, annot=True, cbar=False, fmt="d", ax=axes[0])
axes[0].set_title('Best Model - Confusion Matrix (Training)')
axes[0].set_xlabel('Predicted value')
axes[0].set_ylabel('True value')

sns.heatmap(val_cmat_best, square=True, annot=True, cbar=False, fmt="d", ax=axes[1])
axes[1].set_title('Best Model - Confusion Matrix (Validation)')
axes[1].set_xlabel('Predicted value')
axes[1].set_ylabel('True value')

plt.tight_layout()
plt.savefig('best_model_confusion_matrices.png', dpi=150)
plt.show()

# Plot loss curve for best model
plt.figure(figsize=(10, 6))
plt.plot(best_model.named_steps['clf'].loss_curve_)
plt.title('Best Model - Training Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.savefig('best_model_loss_curve.png', dpi=150)
plt.show()

# ============================================================================
# PART 5: PREDICTIONS ON TEST SET
# ============================================================================

print("\n" + "="*60)
print("PART 5: Making Predictions on Test Set")
print("="*60)

# Train final model on full training set
print("\nRetraining best model on full training set...")

# Remove 'clf__' prefix from parameter names
best_params_clean = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}

final_model = Pipeline([
    ('scaling', StandardScaler()),
    ('clf', MLPClassifier(**best_params_clean, max_iter=200, random_state=42))
])

final_model.fit(X_train_full, y_train_full)

# Make predictions on test set
predictions = final_model.predict(X_test)

print(f"\nTest set predictions completed: {len(predictions)} samples")
print(f"\nPrediction distribution:")
print(pd.Series(predictions).value_counts().sort_index())

# Create submission file
submission = pd.DataFrame({
    'id': range(len(predictions)),
    'position': predictions
})

submission.to_csv('submission.csv', index=False)
print("\nPredictions saved to 'submission.csv'")
print("Ready for Kaggle submission!")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Baseline Model Validation Accuracy: {val_acc:.4f}")
print(f"Best Model Validation Accuracy:     {val_acc_best:.4f}")
print(f"Improvement:                         {val_acc_best - val_acc:.4f}")
print(f"\nBest hyperparameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print("\nFiles generated:")
print("  - submission.csv (for Kaggle)")
print("  - grid_search_results.csv (hyperparameter search results)")
print("  - baseline_confusion_matrices.png")
print("  - best_model_confusion_matrices.png")
print("  - baseline_loss_curve.png")
print("  - best_model_loss_curve.png")
print("="*60)