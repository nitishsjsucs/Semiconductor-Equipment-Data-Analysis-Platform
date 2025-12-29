"""
Anomaly Detection Module for SECOM Equipment Monitoring
Implements multiple anomaly detection algorithms for predictive maintenance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

from .config import RANDOM_STATE


class AnomalyDetector:
    """
    Multi-method anomaly detection for semiconductor equipment monitoring.
    
    This class demonstrates how AI can automate equipment health monitoring,
    enabling predictive maintenance and early fault detection.
    """
    
    def __init__(self, 
                 contamination: float = 0.1,
                 random_state: int = RANDOM_STATE):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in the data
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.detectors_: Dict = {}
        self.results_: Dict = {}
        self.ensemble_predictions_: Optional[np.ndarray] = None
        
    def _get_detectors(self) -> Dict:
        """Get dictionary of anomaly detection algorithms."""
        return {
            'Isolation Forest': IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=200,
                n_jobs=-1
            ),
            'Local Outlier Factor': LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=20
            ),
            'One-Class SVM': OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            ),
            'Elliptic Envelope': EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state
            )
        }
    
    def fit(self, X: pd.DataFrame) -> 'AnomalyDetector':
        """
        Fit all anomaly detection methods.
        
        Args:
            X: Feature DataFrame (should be from normal/healthy operation)
        
        Returns:
            self
        """
        print("\n[ANOMALY] Fitting anomaly detectors...")
        
        detectors = self._get_detectors()
        
        for name, detector in detectors.items():
            print(f"   Training {name}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                detector.fit(X)
            self.detectors_[name] = detector
        
        print(f"   [OK] {len(self.detectors_)} anomaly detectors trained")
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using all fitted detectors.
        
        Args:
            X: Feature DataFrame to analyze
        
        Returns:
            DataFrame with anomaly predictions from each method
        """
        if not self.detectors_:
            raise RuntimeError("Must fit detectors first")
        
        predictions = pd.DataFrame(index=X.index)
        
        for name, detector in self.detectors_.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Predictions: 1 = normal, -1 = anomaly
                pred = detector.predict(X)
                # Convert to: 0 = normal, 1 = anomaly
                predictions[name] = (pred == -1).astype(int)
        
        # Ensemble prediction (majority voting)
        predictions['ensemble'] = (predictions.mean(axis=1) >= 0.5).astype(int)
        predictions['anomaly_score'] = predictions.drop('ensemble', axis=1).mean(axis=1)
        
        self.ensemble_predictions_ = predictions['ensemble'].values
        
        return predictions
    
    def get_anomaly_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get anomaly scores from each detector.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            DataFrame with anomaly scores
        """
        scores = pd.DataFrame(index=X.index)
        
        for name, detector in self.detectors_.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if hasattr(detector, 'decision_function'):
                    # Negative scores = more anomalous
                    score = -detector.decision_function(X)
                elif hasattr(detector, 'score_samples'):
                    score = -detector.score_samples(X)
                else:
                    score = np.zeros(len(X))
                scores[name] = score
        
        # Normalize scores to [0, 1]
        for col in scores.columns:
            min_val, max_val = scores[col].min(), scores[col].max()
            if max_val > min_val:
                scores[col] = (scores[col] - min_val) / (max_val - min_val)
        
        scores['ensemble_score'] = scores.mean(axis=1)
        
        return scores
    
    def analyze_anomalies(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Analyze how well anomaly detection correlates with actual failures.
        
        Args:
            X: Feature DataFrame
            y: Actual labels
        
        Returns:
            Dictionary with analysis results
        """
        predictions = self.predict(X)
        y_binary = (y == 1).astype(int)  # 1 = fail
        
        results = {}
        
        for col in predictions.columns:
            if col in ['anomaly_score']:
                continue
                
            pred = predictions[col]
            
            # Calculate agreement with actual failures
            tp = ((pred == 1) & (y_binary == 1)).sum()
            fp = ((pred == 1) & (y_binary == 0)).sum()
            fn = ((pred == 0) & (y_binary == 1)).sum()
            tn = ((pred == 0) & (y_binary == 0)).sum()
            
            results[col] = {
                'true_positive': tp,
                'false_positive': fp,
                'false_negative': fn,
                'true_negative': tn,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'detected_failures': tp,
                'total_failures': tp + fn,
                'false_alarms': fp
            }
        
        self.results_ = results
        return results
    
    def get_analysis_report(self) -> pd.DataFrame:
        """Get formatted analysis report."""
        if not self.results_:
            raise RuntimeError("Must run analyze_anomalies first")
        
        report = pd.DataFrame(self.results_).T
        return report.sort_values('f1', ascending=False)


class StatisticalAnomalyDetector:
    """
    Statistical methods for anomaly detection using control charts and Z-scores.
    More interpretable approach for process engineers.
    """
    
    def __init__(self, z_threshold: float = 3.0):
        """
        Initialize statistical detector.
        
        Args:
            z_threshold: Z-score threshold for anomaly detection
        """
        self.z_threshold = z_threshold
        self.feature_stats_: Dict = {}
        
    def fit(self, X: pd.DataFrame) -> 'StatisticalAnomalyDetector':
        """Compute feature statistics from normal operation data."""
        for col in X.columns:
            self.feature_stats_[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'q1': X[col].quantile(0.25),
                'q3': X[col].quantile(0.75),
                'median': X[col].median()
            }
        return self
    
    def detect_anomalies(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using statistical methods.
        
        Returns DataFrame with:
        - z_score_anomaly: Features with |Z| > threshold
        - iqr_anomaly: Features outside IQR bounds
        """
        results = pd.DataFrame(index=X.index)
        
        z_scores = pd.DataFrame(index=X.index)
        iqr_anomalies = pd.DataFrame(index=X.index)
        
        for col in X.columns:
            if col not in self.feature_stats_:
                continue
                
            stats = self.feature_stats_[col]
            
            # Z-score
            if stats['std'] > 0:
                z = (X[col] - stats['mean']) / stats['std']
                z_scores[col] = abs(z) > self.z_threshold
            else:
                z_scores[col] = False
            
            # IQR method
            iqr = stats['q3'] - stats['q1']
            lower_bound = stats['q1'] - 1.5 * iqr
            upper_bound = stats['q3'] + 1.5 * iqr
            iqr_anomalies[col] = (X[col] < lower_bound) | (X[col] > upper_bound)
        
        # Count anomalous features per sample
        results['z_score_anomaly_count'] = z_scores.sum(axis=1)
        results['iqr_anomaly_count'] = iqr_anomalies.sum(axis=1)
        
        # Flag as anomaly if many features are anomalous
        n_features = len(X.columns)
        results['is_z_anomaly'] = results['z_score_anomaly_count'] > n_features * 0.1
        results['is_iqr_anomaly'] = results['iqr_anomaly_count'] > n_features * 0.1
        
        return results
    
    def get_anomalous_features(self, sample: pd.Series) -> List[str]:
        """Identify which features are anomalous for a given sample."""
        anomalous = []
        for col in sample.index:
            if col not in self.feature_stats_:
                continue
            stats = self.feature_stats_[col]
            if stats['std'] > 0:
                z = abs(sample[col] - stats['mean']) / stats['std']
                if z > self.z_threshold:
                    anomalous.append((col, z))
        
        return sorted(anomalous, key=lambda x: x[1], reverse=True)


def detect_equipment_anomalies(X: pd.DataFrame, 
                               y: pd.Series,
                               contamination: float = None) -> Dict:
    """
    Complete anomaly detection analysis for equipment data.
    
    Args:
        X: Feature DataFrame
        y: Labels (for validation)
        contamination: Expected anomaly rate (if None, use actual fail rate)
    
    Returns:
        Dictionary with detector, predictions, and analysis
    """
    # Use actual fail rate if contamination not specified
    if contamination is None:
        contamination = (y == 1).mean()
        print(f"   Using actual fail rate as contamination: {contamination:.3f}")
    
    # Train on samples labeled as "pass" for better anomaly detection
    X_normal = X[y == -1]
    print(f"   Training on {len(X_normal)} normal (pass) samples")
    
    # Fit detectors
    detector = AnomalyDetector(contamination=contamination)
    detector.fit(X_normal)
    
    # Analyze all data
    predictions = detector.predict(X)
    analysis = detector.analyze_anomalies(X, y)
    
    print("\nAnomaly Detection Results:")
    print("-" * 50)
    report = detector.get_analysis_report()
    print(report[['precision', 'recall', 'f1', 'detected_failures', 'false_alarms']].to_string())
    
    return {
        'detector': detector,
        'predictions': predictions,
        'analysis': analysis,
        'report': report
    }


if __name__ == "__main__":
    from .data_loader import load_secom_data
    from .preprocessing import preprocess_secom_data
    
    # Load and preprocess
    features, labels, _ = load_secom_data()
    data = preprocess_secom_data(features, labels)
    
    # Run anomaly detection
    results = detect_equipment_anomalies(
        data['X_train'], 
        data['y_train']
    )
    
    print("\n📋 Anomaly Detection Summary:")
    print(f"   Total samples: {len(data['X_train'])}")
    print(f"   Detected anomalies: {results['predictions']['ensemble'].sum()}")
