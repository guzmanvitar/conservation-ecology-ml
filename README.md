# Predicting Missing Plant–Pollinator Interactions with PU Learning + GNN

This project uses **Positive-Unlabeled Learning (PU Learning)** with **Graph Neural Networks (GNNs)** to predict undocumented plant-pollinator interactions from the EuPPollNet dataset. This approach is specifically designed for ecological networks where we have confirmed positive interactions but no confirmed negative examples.

## 🎯 Project Overview

Pollinators are essential for global biodiversity and food security, yet many of their interactions with plants remain undocumented due to incomplete sampling, temporal gaps, or geographic limitations. This project addresses this gap using **PU Learning + GNNs** to predict likely but unrecorded plant-pollinator interactions.

### 🧠 Why PU Learning Fits Ecological Networks

**The Core Problem:**
- ✅ **Confirmed positive interactions** (plants & pollinators that do interact)
- ❓ **No confirmed negatives** (you don't know if others can't interact, or if you just didn't observe them)

**PU Learning's Insight:**
- You have only **Positive (P)** and **Unlabeled (U)** examples
- You suspect some of U are actually Positive, but don't know which
- → **Exactly your ecological scenario**

### Research Question
Can PU Learning with Graph Neural Networks identify additional, undocumented plant-pollinator interactions with high confidence while accounting for the inherent uncertainty in ecological sampling?

### Expected Contributions
- A validated **PU Learning + GNN** model for predicting undocumented mutualistic interactions
- A ranked list of high-probability missing links to guide field verification
- Insights into which species traits and network patterns most influence interaction likelihood
- A practical demonstration of how advanced ML enhances ecological inference and conservation decision-making

## 📊 Dataset

This project uses the **EuPPollNet dataset**—a large, open-access compilation of over 600,000 plant-pollinator interactions recorded across Europe, including:

- 1,864 networks across 23 countries
- 1,400+ plant and 2,200+ pollinator species
- Habitat metadata, geographic information, and year of observation
- Taxonomic information for both plants and pollinators

## 🏗️ Project Structure

```
conservation-ecology-ml/
├── src/
│   ├── constants.py              # Project path constants
│   ├── logger_definition.py      # Logging utilities
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py             # EuPPollNet data loading utilities
│   ├── features/
│   │   ├── __init__.py
│   │   └── pu_features.py        # PU Learning specific features
│   └── models/
│       ├── __init__.py
│       ├── pu_gnn.py             # PU Learning + GNN implementation
│       └── evaluation.py         # Ecological evaluation metrics
├── data/
│   ├── raw/                      # Raw EuPPollNet data
│   ├── processed/                # Processed data files
│   └── interim/                  # Intermediate data
├── models/                       # Trained model files
├── results/                      # Model evaluation results
│   ├── figures/                  # Generated plots
│   └── tables/                   # Results tables
├── logs/                         # Application logs
├── train_pu_gnn.py              # Main training pipeline
├── example_pu_gnn.py            # Usage examples
└── pyproject.toml               # Project configuration
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/guzmanvitar/conservation-ecology-ml.git
cd conservation-ecology-ml

# Install dependencies using uv
uv sync
```

### 2. Install PyTorch Geometric (if needed)

```bash
# For CUDA support
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# For CPU only
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### 3. Run Training

```bash
# Basic training
python train_pu_gnn.py

# With custom parameters
python train_pu_gnn.py \
    --hidden_dim 128 \
    --num_layers 3 \
    --conv_type gat \
    --learning_rate 0.001 \
    --num_epochs 200 \
    --num_unlabeled_samples 15000
```

### 4. Run Examples

```bash
# Run comprehensive examples
python example_pu_gnn.py
```

## 🔬 Methodology

### 🧠 PU Learning Framework

**Step 1: Data Preparation**
- **Positive Examples**: Observed plant-pollinator interactions
- **Unlabeled Examples**: Random plant-pollinator pairs without observed edges
- **No Negative Examples**: We don't assume unobserved pairs are negative

**Step 2: Bipartite Graph Construction**
```python
# Plants and pollinators as nodes
# Observed interactions as edges
# Species traits as node features
```

**Step 3: PU Loss Function**
```python
# Non-negative PU Loss (nnPU)
# Accounts for uncertainty in unlabeled class
# Prevents overfitting to false negatives
```

### 🕸️ Graph Neural Network Architecture

**Bipartite GNN Design:**
- **Separate encoders** for plants and pollinators
- **Graph convolutions** to capture network structure
- **Interaction predictor** for pair-wise predictions

**Supported GNN Types:**
- **GCN** (Graph Convolutional Network)
- **GAT** (Graph Attention Network)
- **GraphSAGE** (Graph SAmple and aggreGatE)

### 🎯 Feature Engineering

**Plant Features:**
- Taxonomic information (Family, Genus, Order)
- Network topology (degree, specialization, diversity)
- Temporal features (flowering season, duration)
- Spatial features (geographic range, network presence)
- Trait features (family dummies, growth form)

**Pollinator Features:**
- Taxonomic information (Family, Genus, Order)
- Network topology (degree, specialization, diversity)
- Temporal features (activity season, duration)
- Spatial features (geographic range, network presence)
- Trait features (family dummies, functional groups)

**Pair Features:**
- Temporal overlap between species
- Spatial overlap between species
- Network overlap (shared partners)
- Ecological compatibility metrics

### 📊 Evaluation Metrics

**Ranking Metrics:**
- **AUC-ROC**: Overall ranking performance
- **Average Precision**: Precision-recall trade-off
- **Precision@k**: Top-k prediction accuracy
- **Recall@k**: Coverage of positive examples
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank

**Ecological Metrics:**
- **Taxonomic Consistency**: Predictions respect taxonomic patterns
- **Temporal Consistency**: Predictions respect seasonal patterns
- **Spatial Consistency**: Predictions respect geographic patterns
- **Network Structure Preservation**: Predictions maintain network properties


## 📊 Results and Outputs

The project generates several outputs:

1. **Trained Models**: Saved in `results/pu_gnn/` directory
2. **Training History**: JSON files with loss curves and metrics
3. **Evaluation Results**: Comprehensive metrics for ranking and ecological validation
4. **Feature Importance**: Analysis of which features drive predictions
5. **Predicted Missing Links**: Ranked list of high-probability interactions
6. **Model Interpretability**: SHAP explanations and feature analysis

## 📚 References

- **PU Learning**: Kiryo et al. (2017). "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
- **Graph Neural Networks**: Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks"
- **Ecological Networks**: Bascompte & Jordano (2007). "Plant-Animal Mutualistic Networks"
