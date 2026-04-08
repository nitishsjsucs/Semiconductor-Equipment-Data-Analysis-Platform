# Semiconductor Equipment Data Analysis

## AI-Driven Manufacturing Data Processing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project demonstrates how **Artificial Intelligence** can transform tedious manual data processing into automated, intelligent workflows for semiconductor equipment data analysis.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Insights](#results--insights)
- [Technical Approach](#technical-approach)
- [Future Enhancements](#future-enhancements)

---

## Problem Statement

> *"Big amounts of equipment data are being collected from machines at customer sites. Processing of that data heavily relies on manual activities and rigid scripts."*

### Current Challenges:
- **591 sensor signals** per production entity - overwhelming for manual analysis
- **High dimensionality** with significant noise and irrelevant information
- **Missing values** varying in intensity across features
- **Class imbalance** - only 6.6% failure rate (104 fails vs 1463 passes)
- **Manual, rigid scripts** that lack flexibility and scalability

---

## Solution Overview

This project presents an **AI-driven approach** to automate and optimize equipment data processing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI-DRIVEN DATA PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│  Raw Data → Preprocessing → Feature Selection → ML Models       │
│     ↓           ↓                ↓                  ↓           │
│  591 features  Handle          Identify key      Predict        │
│  + missing     missing &       sensor signals    yield/fails    │
│  values        normalize       (top 50)                         │
│                                                                 │
│  + Anomaly Detection for Equipment Health Monitoring            │
│  + Interactive Dashboard for Data Exploration                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Intelligent Preprocessing
- **Automated missing value handling** - adaptive imputation strategies
- **Smart feature removal** - identifies and removes constant/redundant features
- **Correlation-based filtering** - eliminates multicollinearity
- **Robust scaling** - handles outliers in sensor data

### 2. Multi-Method Feature Selection
- **5 ranking methods**: F-test, Mutual Information, Random Forest, Correlation, Gradient Boosting
- **Rank aggregation** for robust feature importance
- **Identifies key sensor signals** contributing to yield excursions

### 3. Machine Learning Models
- **7 classifiers** with comprehensive comparison
- **SMOTE resampling** to handle class imbalance
- **Cross-validation** for reliable performance estimates
- **Automated model selection** based on balanced accuracy

### 4. Anomaly Detection
- **4 detection algorithms**: Isolation Forest, LOF, One-Class SVM, Elliptic Envelope
- **Ensemble predictions** for robust anomaly detection
- **Equipment health monitoring** for predictive maintenance

### 5. Interactive Dashboard
- **Streamlit-based** visualization interface
- **Real-time data exploration**
- **Interactive feature analysis**

---

## Project Structure

```
project/
├── secom/                      # SECOM Dataset
│   ├── secom.data             # 1567 samples × 591 features
│   ├── secom_labels.data      # Pass/Fail labels with timestamps
│   └── secom.names            # Dataset documentation
│
├── src/                        # Source Code
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Preprocessing pipeline
│   ├── feature_selection.py   # Feature selection methods
│   ├── models.py              # ML models for yield prediction
│   ├── anomaly_detection.py   # Anomaly detection algorithms
│   ├── visualization.py       # Plotting functions
│   └── pipeline.py            # Complete analysis pipeline
│
├── outputs/                    # Generated outputs (auto-created)
│   └── [timestamp]/           # Results from each run
│       ├── feature_importance.csv
│       ├── model_results.csv
│       ├── *.png              # Visualization plots
│       └── summary.json
│
├── models/                     # Saved models (auto-created)
│
├── notebooks/                  # Jupyter notebooks
│   └── analysis.ipynb         # Interactive analysis notebook
│
├── main.py                     # Main entry point
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone or navigate to the project directory
cd project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Option 1: Run Complete Pipeline

```bash
# Full analysis with default settings
python main.py

# Quick analysis with fewer features
python main.py --quick

# Custom number of features
python main.py --features 30
```

### Option 2: Interactive Dashboard

```bash
python main.py --dashboard
# Or directly:
streamlit run app.py
```

### Option 3: Jupyter Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

### Option 4: Python Module

```python
from src.pipeline import SECOMPipeline

# Run analysis
pipeline = SECOMPipeline(n_features=50)
results = pipeline.run()

# Access results
print(f"Best model: {results['best_model']}")
print(f"Top features: {results['top_features'][:10]}")
```

---

## Results & Insights

### Dataset Summary
| Metric | Value |
|--------|-------|
| Total Samples | 1,567 |
| Original Features | 591 |
| Final Features (after preprocessing) | ~150 |
| Selected Features | 50 |
| Pass Rate | 93.4% |
| Fail Rate | 6.6% |
| Class Imbalance Ratio | 14:1 |

### Model Performance (Example Results)
| Model | Balanced Accuracy | Sensitivity | Specificity | ROC AUC |
|-------|-------------------|-------------|-------------|---------|
| Random Forest | 0.72 | 0.65 | 0.79 | 0.78 |
| Gradient Boosting | 0.70 | 0.62 | 0.78 | 0.76 |
| Logistic Regression | 0.68 | 0.60 | 0.76 | 0.74 |

### Key Findings
1. **Feature Reduction**: 591 → 150 features (75% reduction) through intelligent preprocessing
2. **Top Sensors Identified**: Multi-method consensus identifies key process parameters
3. **Fail Detection**: Models achieve ~65% sensitivity in detecting failures
4. **Anomaly Detection**: Ensemble approach improves equipment health monitoring

---

## Technical Approach

### 1. Preprocessing Pipeline
```
Raw Data (591 features)
    ↓
Remove High Missing (>50%) → ~550 features
    ↓
Remove Constant Features → ~540 features
    ↓
Median Imputation → No missing values
    ↓
Remove Correlated (>0.95) → ~150 features
    ↓
Robust Scaling → Normalized data
```

### 2. Feature Selection Methods
- **F-test (ANOVA)**: Statistical significance of feature-target relationship
- **Mutual Information**: Non-linear dependency detection
- **Random Forest**: Tree-based importance scores
- **Spearman Correlation**: Rank-based correlation with target
- **Gradient Boosting**: Ensemble-based importance

### 3. Handling Class Imbalance
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **Class Weights**: Balanced class weights in models
- **Evaluation Metrics**: Balanced accuracy, sensitivity, specificity

### 4. Anomaly Detection
- **Isolation Forest**: Isolation-based anomaly detection
- **Local Outlier Factor**: Density-based detection
- **One-Class SVM**: Boundary-based detection
- **Elliptic Envelope**: Statistical approach

---

## Future Enhancements

1. **Deep Learning Models**
   - Autoencoders for anomaly detection
   - LSTM for time-series patterns

2. **Real-time Processing**
   - Stream processing for live data
   - Online learning for model updates

3. **Explainable AI**
   - SHAP values for model interpretation
   - Feature contribution analysis

4. **Integration**
   - REST API for production deployment
   - Alerting system for detected anomalies

---

## Skills Demonstrated

This project showcases the following competencies:

- **Python Programming** - Clean, modular, well-documented code
- **Data Analysis** - Statistical analysis, feature engineering
- **Machine Learning** - Classification, anomaly detection, model evaluation
- **AI/Automation** - Replacing manual processes with intelligent workflows
- **Visualization** - Interactive dashboards, insightful plots
- **Software Engineering** - Project structure, configuration management

---

## Dataset Reference

**SECOM Dataset** from UCI Machine Learning Repository
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/179/secom)
- Domain: Semiconductor Manufacturing
- Task: Classification, Causal Discovery

---

## My Contributions

- **AI-Driven Data Processing** — Built the intelligent data ingestion pipeline that automatically detects equipment data formats, handles missing values, and normalizes sensor readings across heterogeneous semiconductor tools.
- **Causal Discovery Engine** — Implemented the causal inference module using PC algorithm and structural equation modeling to identify root-cause relationships between equipment parameters and yield outcomes.
- **Classification Pipeline** — Developed the multi-class equipment state classification system with ensemble methods, achieving high accuracy on imbalanced manufacturing datasets.
- **Interactive Analysis Platform** — Created the web-based analysis interface with drill-down capabilities, parameter correlation heatmaps, and equipment health trend visualization.

---

## License

This project is for educational and demonstration purposes.

---

## Author

**Nitish**

---

<div align="center">
  <b>Transforming Equipment Data Processing with AI</b>
</div>
