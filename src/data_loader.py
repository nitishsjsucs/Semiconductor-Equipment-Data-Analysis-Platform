"""
Data Loading Module for SECOM Dataset
Handles loading and initial parsing of semiconductor manufacturing data
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

from .config import SECOM_DATA_FILE, SECOM_LABELS_FILE, PASS_LABEL, FAIL_LABEL


def load_secom_data(data_path: Optional[Path] = None, 
                    labels_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the SECOM dataset from raw files.
    
    Args:
        data_path: Path to secom.data file
        labels_path: Path to secom_labels.data file
    
    Returns:
        Tuple of (features DataFrame, labels Series, timestamps Series)
    """
    data_path = data_path or SECOM_DATA_FILE
    labels_path = labels_path or SECOM_LABELS_FILE
    
    # Load feature data (591 sensor measurements per sample)
    # NaN values are represented as 'NaN' in the file
    print(f"Loading feature data from {data_path}...")
    features = pd.read_csv(
        data_path, 
        sep=r'\s+',  # Space-separated
        header=None,
        na_values=['NaN', 'nan', 'NA'],
        dtype=np.float64
    )
    
    # Generate feature names (sensor signals)
    features.columns = [f'sensor_{i:03d}' for i in range(features.shape[1])]
    
    # Load labels and timestamps
    print(f"Loading labels from {labels_path}...")
    labels_df = pd.read_csv(
        labels_path,
        sep=r'\s+',
        header=None,
        names=['label', 'timestamp'],
        parse_dates=['timestamp'],
        dayfirst=True  # Format: DD/MM/YYYY HH:MM:SS
    )
    
    labels = labels_df['label']
    timestamps = labels_df['timestamp']
    
    print(f"Loaded {len(features)} samples with {features.shape[1]} features")
    print(f"Class distribution: Pass={sum(labels == PASS_LABEL)}, Fail={sum(labels == FAIL_LABEL)}")
    
    return features, labels, timestamps


def get_data_summary(features: pd.DataFrame, labels: pd.Series) -> dict:
    """
    Generate summary statistics for the dataset.
    
    Args:
        features: Feature DataFrame
        labels: Labels Series
    
    Returns:
        Dictionary with summary statistics
    """
    n_samples, n_features = features.shape
    n_pass = sum(labels == PASS_LABEL)
    n_fail = sum(labels == FAIL_LABEL)
    
    # Missing value analysis
    missing_per_feature = features.isnull().sum()
    missing_per_sample = features.isnull().sum(axis=1)
    total_missing = features.isnull().sum().sum()
    total_cells = n_samples * n_features
    
    # Features with any missing values
    features_with_missing = (missing_per_feature > 0).sum()
    
    # Constant features (zero variance)
    constant_features = (features.std() == 0).sum()
    
    summary = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_pass': n_pass,
        'n_fail': n_fail,
        'imbalance_ratio': n_pass / n_fail if n_fail > 0 else float('inf'),
        'fail_rate': n_fail / n_samples * 100,
        'total_missing_values': total_missing,
        'missing_percentage': total_missing / total_cells * 100,
        'features_with_missing': features_with_missing,
        'constant_features': constant_features,
        'avg_missing_per_sample': missing_per_sample.mean(),
        'max_missing_per_sample': missing_per_sample.max(),
        'avg_missing_per_feature': missing_per_feature.mean(),
        'max_missing_per_feature': missing_per_feature.max(),
    }
    
    return summary


def print_data_summary(summary: dict) -> None:
    """Print formatted data summary."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"\nDataset Size:")
    print(f"   - Samples: {summary['n_samples']:,}")
    print(f"   - Features: {summary['n_features']:,}")
    
    print(f"\nClass Distribution:")
    print(f"   - Pass (yield): {summary['n_pass']:,} ({100 - summary['fail_rate']:.1f}%)")
    print(f"   - Fail (defect): {summary['n_fail']:,} ({summary['fail_rate']:.1f}%)")
    print(f"   - Imbalance ratio: {summary['imbalance_ratio']:.1f}:1")
    
    print(f"\nMissing Values:")
    print(f"   - Total missing: {summary['total_missing_values']:,} ({summary['missing_percentage']:.1f}%)")
    print(f"   - Features with missing: {summary['features_with_missing']:,}")
    print(f"   - Avg missing per sample: {summary['avg_missing_per_sample']:.1f}")
    print(f"   - Max missing per sample: {summary['max_missing_per_sample']:,}")
    
    print(f"\nData Quality Issues:")
    print(f"   - Constant features: {summary['constant_features']:,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test data loading
    features, labels, timestamps = load_secom_data()
    summary = get_data_summary(features, labels)
    print_data_summary(summary)
