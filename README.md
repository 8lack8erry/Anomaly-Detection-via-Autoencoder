# Anomaly Detection via Autoencoder for LHC Physics

This repository contains deep learning implementations for anomaly detection in high-energy physics, specifically designed for detecting Beyond Standard Model (BSM) physics signals in Large Hadron Collider (LHC) data. 

## Overview

The goal of this project is to develop algorithms for detecting New Physics by reformulating the problem as an out-of-distribution detection task. Using four-vectors of the highest-momentum jets, electrons, and muons produced in LHC collision events, together with the missing transverse energy (missing ET), we aim to find a-priori unknown and rare New Physics hidden in data samples dominated by ordinary Standard Model processes.

## Dataset

The datasets used in this project come from the **Anomaly Detection Data Challenge 2021**, available at: https://mpp-hep.github.io/ADC2021/

### Data Description
- **Input features**: 19 particles × 4 features (pT, η, φ, mass) per event
- **Training data**: Standard Model background events for unsupervised learning
- **Test signals**: Four different Beyond Standard Model scenarios:
  - A to 4 leptons
  - Leptoquark to bτ  
  - h⁰ to ττ
  - h⁺ to τν

### Data Preprocessing
All models implement standardized preprocessing:
- **Standard Scaler normalization**: Applied feature-wise across the dataset
- **Train/Validation/Test split**: 70%/20%/10% respectively
- **Input shape**: (batch_size, 19, 4) representing particle four-vectors

## Implemented Architectures

### 1. Feed Forward Network (FFN)
**File**: `networks/FFN.ipynb`

A fully connected autoencoder with symmetric encoder-decoder architecture:
- **Encoder**: 76 → 128 → 64 → 32 → 16 (bottleneck)
- **Decoder**: 16 → 32 → 64 → 128 → 76
- **Regularization**: L2 regularization, batch normalization, dropout
- **Activation**: LeakyReLU throughout the network

### 2. Convolutional Neural Network (CNN)  
**File**: `networks/CNN.ipynb`

A convolutional autoencoder designed for spatial feature extraction:
- **Encoder**: Conv2D layers with (3,1) kernels, max pooling
- **Decoder**: Transpose convolutions for upsampling
- **Architecture**: Processes particle sequences as 2D feature maps
- **Channels**: 16 → 32 → 64 → 128 in encoder path

### 3. Transformer Architecture
**File**: `networks/Trasformer.ipynb`

A transformer-based autoencoder leveraging self-attention mechanisms:
- **Embedding**: Dense projection to 32-dimensional space
- **Positional encoding**: Learned positional embeddings for particle sequences  
- **Self-attention**: Multi-head attention with 4 heads
- **Architecture**: Transformer encoder-decoder with attention mechanisms

## Model Performance

Each architecture is optimized for:
- **Reconstruction loss**: Mean Squared Error between input and reconstructed events
- **Anomaly detection**: Events with high reconstruction error flagged as potential BSM signals
- **Efficiency metrics**: Evaluated on true positive rate at fixed false positive rate

## Training Configuration

All models share common training parameters:
- **Regularization**: L2 weight decay (1e-4)
- **Dropout**: 0.01-0.3 depending on architecture
- **Batch normalization**: Applied after dense/conv layers
- **Loss function**: Mean Squared Error (reconstruction loss)
- **Optimization**: Adam optimizer with learning rate scheduling

## Key Features

- **Unsupervised Learning**: No labeled anomaly data required for training
- **Multiple Architectures**: Comparative study of FFN, CNN, and Transformer approaches
- **Physics-Informed**: Designed specifically for particle physics four-vector data
- **Trigger-Ready**: Optimized for real-time LHC trigger system constraints
- **Reproducible**: Complete implementation with saved model weights

## References

- [Anomaly Detection Data Challenge 2021](https://mpp-hep.github.io/ADC2021/)
- [Challenge Paper: arXiv:2107.02157](https://arxiv.org/abs/2107.02157)


