# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based neural network classification project for Wi-Fi device localization using Channel State Information (CSI).

### Problem Domain

**Channel State Information (CSI)** describes how Wi-Fi transmitted signals are affected by the wireless channel before reaching the receiver. Modern Wi-Fi uses OFDM (Orthogonal Frequency Division Multiplexing), which splits transmissions into many closely spaced subcarriers. As signals travel through the environment, each subcarrier experiences different effects from multipath propagation, reflections, shadowing, and fading. CSI captures these effects by reporting amplitude and phase for every OFDM subcarrier.

**Why Machine Learning**: Wi-Fi CSI generates high-dimensional data (64 subcarriers for 20 MHz channels, up to 4096 for 320 MHz in Wi-Fi 7). With hundreds or thousands of features per CSI sample, ML techniques can learn complex patterns that analytical models struggle to handle. CSI-based sensing is increasingly central in Wi-Fi standards evolution (IEEE 802.11bf) for applications like motion detection, presence sensing, occupancy tracking, localization, and even vital signs monitoring.

**This Task**: Train a system to locate and track a Wi-Fi device by analyzing CSI from its transmissions. Formulated as a 10-class classification problem where the target device occupies one of ten predefined positions (classes 0-9) relative to the tracking unit. The input features are 260-dimensional CSI measurements.

## Performance Summary

| Version | Validation Accuracy | Description |
|---------|---------------------|-------------|
| v3 (Neural.py) | 95.71% | Base 4-layer feedforward |
| v4 Ensemble | 96.48% | 7-model ensemble with feature engineering |
| v4 Final Ensemble | **96.59%** | 12-model diversity-optimized ensemble |

## Directory Structure

- Root directory (`/v3`): Contains all implementations
- `v3.1/`: Earlier grid search version (`Neural3.1.py`)
- Data files: `train.csv` (12,888 samples), `test_nolabels.csv` (3,223 samples)

**Key Files**:
- `Neural.py` - Base model (95.71%)
- `Neural_v4.py` - Single improved model with feature engineering
- `Neural_v4_ensemble.py` - 7-model ensemble training
- `Neural_v4_hyperopt.py` - 43-configuration hyperparameter search
- `Neural_v4_final_ensemble.py` - Diversity-optimized final ensemble (96.59%)

**Model Checkpoints**:
- `best_model.pth` - Base v3 model
- `hyperopt_model_*.pth` - 43 models from hyperparameter search
- `model*_base_seed*.pth` - Original ensemble models

**Outputs**:
- `submission_v4_final.csv` - Best predictions (96.59% val acc)
- `final_ensemble_analysis.png` - Ensemble visualization
- `hyperopt_results.csv` - All hyperparameter search results

## Commands

### Running with Docker (ROCm/AMD GPU)

```bash
cd "/home/leevi/Desktop/Competition/Neural Network/v3"

sudo docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$PWD:/workspace" \
  -w /workspace \
  rocm/pytorch:latest \
  bash -c "pip install -r requirements.txt && python3 <script_name>.py"
```

### Training Scripts

```bash
# Base model (95.71%)
python Neural.py

# v4 single model with all improvements
python Neural_v4.py

# v4 7-model ensemble (96.48%)
python Neural_v4_ensemble.py

# Hyperparameter optimization (43 configs)
python Neural_v4_hyperopt.py

# Final diversity-optimized ensemble (96.59%)
python Neural_v4_final_ensemble.py
```

### Dependencies

```bash
pip install -r requirements.txt
```

Required: pandas, numpy, torch, scikit-learn, matplotlib, scipy

### GPU Usage

Code auto-detects GPU via `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.

## Architecture

### v3 Base Model: CSIClassifier

4-layer feedforward network:
- Input: 260 features → 512 → 256 → 128 → 64 → 10 classes
- BatchNorm1d + ReLU + Dropout after each layer
- Dropout: 0.3 (first two layers), 0.2 (next two layers)

### v4 Improved Model: CSIResNetClassifier

5-layer residual network with skip connections:
- Input: 335 engineered features → 1024 → 512 → 256 → 128 → 64 → 10 classes
- ResidualBlocks with skip connections for better gradient flow
- Dropout schedule: 0.45 → 0.40 → 0.35 → 0.30 → 0.25

### v4 Feature Engineering

Transforms 260 raw features into 335 engineered features:

1. **Amplitude/Phase Extraction**: Convert I/Q pairs to polar representation
   - Amplitude: `sqrt(I² + Q²)` for each subcarrier
   - Phase: `arctan2(Q, I)` for each subcarrier

2. **Inactive Subcarrier Removal**: Remove 24 WiFi guard band subcarriers (indices 0, 27-37)
   - Reduces noise from null/pilot carriers

3. **Cross-Antenna Features**: Capture angle-of-arrival information
   - Amplitude difference between antennas
   - Phase difference (normalized to [-π, π])

4. **Band Statistics**: Aggregate features for low/mid/high frequency bands
   - Mean, std, max amplitude per band per antenna

### Data Pipeline

1. **Preprocessing**: Separate metadata (5 features) from CSI (255 features)
2. **Feature Engineering**: Apply CSI transforms (v4 only)
3. **Normalization**: StandardScaler on metadata and CSI separately
4. **Train/Val Split**: 85/15 with stratification
5. **DataLoader**: Batch size 256 (v4), 128 (v3)

### Training Configuration

**v4 Best Hyperparameters** (from 43-config search):
- Learning rate: 0.001
- Weight decay: 5e-6
- Dropout base: 0.45
- Label smoothing: 0.1
- Mixup alpha: 0.2
- Warmup epochs: 10
- Scheduler: Warmup + Cosine Annealing

**v4 Training Enhancements**:
- Mixup data augmentation
- Label smoothing cross-entropy
- Gradient clipping (max_norm=1.0)
- Early stopping (patience=30-40)

### Ensemble Strategy

**v4 Final Ensemble** (96.59%):
- 12 models selected from 43 hyperopt configurations
- Diversity-based greedy selection: `0.7 × accuracy + 0.3 × diversity`
- Predictions averaged (softmax probabilities)

## Key Implementation Details

- Input data: 260 features (5 metadata + 255 CSI I/Q values)
- Labels: 'position' column with values 0-9
- CSI structure: 64 subcarriers × 2 antennas × 2 (I/Q) = 256 values
- Metadata: timestamp, seq_ctrl, aoa, rssi1, rssi2
- All scripts are self-contained with inline execution
- Models output raw logits (CrossEntropyLoss handles softmax internally)

## Future Improvements

Potential strategies to push beyond 96.59%:
- **5-fold cross-validation ensemble**: Use all training data more effectively
- **Architectural diversity**: Add 1D CNN or Transformer variants to ensemble
- **Advanced stacking**: Train meta-learner to weight ensemble members
- **Test-time augmentation**: Average predictions with slight perturbations
