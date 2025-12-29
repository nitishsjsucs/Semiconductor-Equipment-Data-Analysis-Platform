"""
Data Preprocessing Module for SECOM Dataset
Handles missing values, normalization, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from typing import Tuple, Optional, List
import warnings

from .config import (
    MISSING_VALUE_THRESHOLD, 
    CORRELATION_THRESHOLD,
    VARIANCE_THRESHOLD,
    RANDOM_STATE
)


class SECOMPreprocessor:
    """
    Comprehensive preprocessing pipeline for SECOM semiconductor data.
    
    This class demonstrates AI-driven automation of data preprocessing,
    replacing manual, rigid scripts with intelligent, adaptive processing.
    """
    
    def __init__(self, 
                 missing_threshold: float = MISSING_VALUE_THRESHOLD,
                 correlation_threshold: float = CORRELATION_THRESHOLD,
                 variance_threshold: float = VARIANCE_THRESHOLD,
                 imputation_strategy: str = 'median',
                 scaling_method: str = 'robust'):
        """
        Initialize the preprocessor.
        
        Args:
            missing_threshold: Remove features with missing ratio above this
            correlation_threshold: Remove features with correlation above this
            variance_threshold: Remove features with variance below this
            imputation_strategy: 'mean', 'median', 'knn', or 'most_frequent'
            scaling_method: 'standard', 'robust', or 'minmax'
        """
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.imputation_strategy = imputation_strategy
        self.scaling_method = scaling_method
        
        # These will be set during fit
        self.features_to_keep_: List[str] = []
        self.removed_features_: dict = {}
        self.imputer_ = None
        self.scaler_ = None
        self.is_fitted_ = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'SECOMPreprocessor':
        """
        Fit the preprocessor to the training data.
        
        Args:
            X: Feature DataFrame
            y: Labels (optional, for supervised feature selection)
        
        Returns:
            self
        """
        print("\n[PREPROCESS] Fitting preprocessor...")
        X_processed = X.copy()
        
        # Step 1: Remove features with too many missing values
        X_processed, removed_missing = self._remove_high_missing_features(X_processed)
        self.removed_features_['high_missing'] = removed_missing
        print(f"   Removed {len(removed_missing)} features with >{self.missing_threshold*100}% missing values")
        
        # Step 2: Remove constant features (zero variance)
        X_processed, removed_constant = self._remove_constant_features(X_processed)
        self.removed_features_['constant'] = removed_constant
        print(f"   Removed {len(removed_constant)} constant features")
        
        # Step 3: Impute missing values temporarily to find correlated features
        temp_imputer = SimpleImputer(strategy='median')
        X_temp_imputed = pd.DataFrame(
            temp_imputer.fit_transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        # Step 4: Remove highly correlated features BEFORE fitting final imputer
        X_temp_imputed, removed_corr = self._remove_correlated_features(X_temp_imputed)
        self.removed_features_['correlated'] = removed_corr
        print(f"   Removed {len(removed_corr)} highly correlated features")
        
        # Store features to keep (after all removals)
        self.features_to_keep_ = list(X_temp_imputed.columns)
        
        # Step 5: Now fit the actual imputer on the final feature set
        X_final = X_processed[self.features_to_keep_]
        self.imputer_ = self._create_imputer()
        X_imputed = pd.DataFrame(
            self.imputer_.fit_transform(X_final),
            columns=X_final.columns,
            index=X_final.index
        )
        print(f"   Fitted {self.imputation_strategy} imputer")
        
        # Step 6: Fit scaler
        self.scaler_ = self._create_scaler()
        self.scaler_.fit(X_imputed)
        print(f"   Fitted {self.scaling_method} scaler")
        
        self.is_fitted_ = True
        
        total_removed = sum(len(v) for v in self.removed_features_.values())
        print(f"   [OK] Preprocessing fitted: {len(self.features_to_keep_)} features retained "
              f"({total_removed} removed)")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted preprocessor.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Preprocessed DataFrame
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Select only the features we're keeping
        available_features = [f for f in self.features_to_keep_ if f in X.columns]
        X_subset = X[available_features].copy()
        
        # Impute missing values
        X_imputed = pd.DataFrame(
            self.imputer_.transform(X_subset),
            columns=X_subset.columns,
            index=X_subset.index
        )
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler_.transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def _remove_high_missing_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with missing values above threshold."""
        missing_ratio = X.isnull().sum() / len(X)
        features_to_remove = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
        X_clean = X.drop(columns=features_to_remove)
        return X_clean, features_to_remove
    
    def _remove_constant_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with zero or near-zero variance."""
        # First, impute temporarily to calculate variance
        temp_imputer = SimpleImputer(strategy='median')
        X_temp = pd.DataFrame(
            temp_imputer.fit_transform(X),
            columns=X.columns
        )
        
        variances = X_temp.var()
        features_to_remove = variances[variances < self.variance_threshold].index.tolist()
        X_clean = X.drop(columns=features_to_remove)
        return X_clean, features_to_remove
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features, keeping one from each correlated pair."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_matrix = X.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        features_to_remove = [
            column for column in upper.columns 
            if any(upper[column] > self.correlation_threshold)
        ]
        
        X_clean = X.drop(columns=features_to_remove)
        return X_clean, features_to_remove
    
    def _create_imputer(self):
        """Create the appropriate imputer based on strategy."""
        if self.imputation_strategy == 'knn':
            return KNNImputer(n_neighbors=5)
        else:
            return SimpleImputer(strategy=self.imputation_strategy)
    
    def _create_scaler(self):
        """Create the appropriate scaler based on method."""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        return scalers.get(self.scaling_method, RobustScaler())
    
    def get_preprocessing_report(self) -> dict:
        """Generate a report of preprocessing steps applied."""
        if not self.is_fitted_:
            return {"status": "Not fitted"}
        
        return {
            "original_features": (
                len(self.features_to_keep_) + 
                sum(len(v) for v in self.removed_features_.values())
            ),
            "final_features": len(self.features_to_keep_),
            "removed_high_missing": len(self.removed_features_.get('high_missing', [])),
            "removed_constant": len(self.removed_features_.get('constant', [])),
            "removed_correlated": len(self.removed_features_.get('correlated', [])),
            "imputation_strategy": self.imputation_strategy,
            "scaling_method": self.scaling_method,
            "missing_threshold": self.missing_threshold,
            "correlation_threshold": self.correlation_threshold,
        }


def preprocess_secom_data(features: pd.DataFrame, 
                          labels: pd.Series,
                          test_size: float = 0.2) -> dict:
    """
    Complete preprocessing pipeline for SECOM data.
    
    Args:
        features: Raw feature DataFrame
        labels: Labels Series
        test_size: Fraction of data for testing
    
    Returns:
        Dictionary with preprocessed train/test splits and preprocessor
    """
    from sklearn.model_selection import train_test_split
    from .config import RANDOM_STATE
    
    # Split data first (to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=labels  # Maintain class balance in splits
    )
    
    print(f"\nData split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Create and fit preprocessor
    preprocessor = SECOMPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'report': preprocessor.get_preprocessing_report()
    }


if __name__ == "__main__":
    from .data_loader import load_secom_data
    
    # Test preprocessing
    features, labels, _ = load_secom_data()
    result = preprocess_secom_data(features, labels)
    
    print("\n📋 Preprocessing Report:")
    for key, value in result['report'].items():
        print(f"   {key}: {value}")
