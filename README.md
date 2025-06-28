# Predicting Missing Plant–Pollinator Interactions with Machine Learning

This project uses machine learning to predict undocumented plant-pollinator interactions from the EuPPollNet dataset, helping to identify missing links in ecological networks and inform conservation strategies.

## 🎯 Project Overview

Pollinators are essential for global biodiversity and food security, yet many of their interactions with plants remain undocumented due to incomplete sampling, temporal gaps, or geographic limitations. This project addresses this gap by using machine learning to predict likely but unrecorded plant-pollinator interactions.

### Research Question
Can machine learning models trained on known plant-pollinator interactions identify additional, undocumented interactions with high confidence?

### Expected Contributions
- A validated ML model for predicting undocumented mutualistic interactions
- A ranked list of high-probability missing links to guide field verification
- Insights into which species traits and co-occurrence patterns most influence interaction likelihood
- A practical demonstration of how ML enhances ecological inference and conservation decision-making

## 📊 Dataset

This project uses the **EuPPollNet dataset**—a large, open-access compilation of over 600,000 plant-pollinator interactions recorded across Europe, including:

- 1,864 networks across 23 countries
- 1,400+ plant and 2,200+ pollinator species
- Habitat metadata, geographic information, and year of observation

## 🏗️ Project Structure

```
conservation-ecology-ml/
├── src/
│   ├── constants.py              # Project path constants
│   ├── logger_definition.py      # Logging utilities
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py             # Data loading utilities
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py        # Feature engineering
│   └── models/
│       ├── __init__.py
│       └── trainer.py            # Model training and evaluation
├── data/
│   ├── raw/                      # Raw EuPPollNet data
│   ├── processed/                # Processed data files
│   └── interim/                  # Intermediate data
├── models/                       # Trained model files
├── results/                      # Model evaluation results
│   ├── figures/                  # Generated plots
│   └── tables/                   # Results tables
├── logs/                         # Application logs
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

## 🔬 Methodology

### Feature Engineering
TBD
### Machine Learning Models
TBD
### Evaluation Metrics
TBD
## 📈 Usage Examples
TBD

## 🛠️ Development

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting
- **pre-commit**: Git hooks for automatic formatting


## 📊 Results and Outputs

The project generates several outputs:

1. **Trained Models**: Saved in `models/` directory
2. **Evaluation Results**: CSV files with performance metrics
3. **Feature Importance**: Analysis of which features drive predictions
4. **Predicted Missing Links**: Ranked list of high-probability interactions
5. **SHAP Explanations**: Model interpretability analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
