"""
Machine Learning Models for SECOM Yield Prediction
Handles class imbalance and provides comprehensive model comparison
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from typing import Dict, List, Tuple, Optional
import warnings

from .config import RANDOM_STATE, CV_FOLDS, MODELS_DIR


class YieldPredictor:
    """
    AI-driven yield prediction for semiconductor manufacturing.
    
    This class demonstrates how ML can automate defect prediction,
    replacing rigid rule-based systems with adaptive learning.
    """
    
    def __init__(self, 
                 handle_imbalance: str = 'smote',
                 random_state: int = RANDOM_STATE):
        """
        Initialize yield predictor.
        
        Args:
            handle_imbalance: Strategy for handling class imbalance
                            ('smote', 'adasyn', 'undersample', 'smote_tomek', 'class_weight')
            random_state: Random seed for reproducibility
        """
        self.handle_imbalance = handle_imbalance
        self.random_state = random_state
        self.models_: Dict = {}
        self.results_: Dict = {}
        self.best_model_name_: str = ""
        self.best_model_ = None
        
    def _get_base_models(self) -> Dict:
        """Get dictionary of base models to evaluate."""
        return {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        }
    
    def _get_sampler(self):
        """Get the appropriate sampler for handling imbalance."""
        samplers = {
            'smote': SMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'undersample': RandomUnderSampler(random_state=self.random_state),
            'smote_tomek': SMOTETomek(random_state=self.random_state),
            'class_weight': None  # Handled by model's class_weight parameter
        }
        return samplers.get(self.handle_imbalance)
    
    def train_and_evaluate(self, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           X_test: pd.DataFrame,
                           y_test: pd.Series,
                           cv_folds: int = CV_FOLDS) -> pd.DataFrame:
        """
        Train and evaluate multiple models with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            cv_folds: Number of cross-validation folds
        
        Returns:
            DataFrame with model comparison results
        """
        print("\n[MODEL] Training and evaluating models...")
        print(f"   Imbalance handling: {self.handle_imbalance}")
        print(f"   Cross-validation folds: {cv_folds}")
        
        # Convert labels to binary
        y_train_binary = (y_train == 1).astype(int)
        y_test_binary = (y_test == 1).astype(int)
        
        # Apply resampling if needed
        sampler = self._get_sampler()
        if sampler is not None:
            print(f"   Applying {self.handle_imbalance} resampling...")
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train_binary)
            print(f"   Resampled: {len(X_train)} → {len(X_train_resampled)} samples")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train_binary
        
        base_models = self._get_base_models()
        results = []
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in base_models.items():
            print(f"\n   Training {name}...")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_resampled, y_train_resampled,
                    cv=cv, scoring='balanced_accuracy'
                )
                
                # Train on full training set
                model.fit(X_train_resampled, y_train_resampled)
                
                # Predict on test set
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test_binary, y_pred, y_proba)
                metrics['cv_balanced_acc_mean'] = cv_scores.mean()
                metrics['cv_balanced_acc_std'] = cv_scores.std()
                metrics['model_name'] = name
                
                results.append(metrics)
                self.models_[name] = model
                
                print(f"      CV Balanced Acc: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                print(f"      Test Balanced Acc: {metrics['balanced_accuracy']:.3f}")
                print(f"      Test F1 (fail class): {metrics['f1_fail']:.3f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('balanced_accuracy', ascending=False)
        self.results_ = results_df
        
        # Select best model based on balanced accuracy
        self.best_model_name_ = results_df.iloc[0]['model_name']
        self.best_model_ = self.models_[self.best_model_name_]
        
        print(f"\n   [OK] Best model: {self.best_model_name_}")
        print(f"      Balanced Accuracy: {results_df.iloc[0]['balanced_accuracy']:.3f}")
        
        return results_df
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_fail': f1_score(y_true, y_pred, pos_label=1),
            'f1_pass': f1_score(y_true, y_pred, pos_label=0),
        }
        
        # Confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positive'] = tp
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for fail class
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for pass class
        
        # AUC if probabilities available
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_proba)
        else:
            metrics['roc_auc'] = None
            metrics['avg_precision'] = None
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the best model."""
        if self.best_model_ is None:
            raise RuntimeError("Must train models first")
        return self.best_model_.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using the best model."""
        if self.best_model_ is None:
            raise RuntimeError("Must train models first")
        return self.best_model_.predict_proba(X)
    
    def save_best_model(self, filename: str = "best_yield_model.joblib"):
        """Save the best model to disk."""
        filepath = MODELS_DIR / filename
        joblib.dump({
            'model': self.best_model_,
            'model_name': self.best_model_name_,
            'results': self.results_
        }, filepath)
        print(f"   [OK] Model saved to {filepath}")
        return filepath
    
    def get_classification_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """Get detailed classification report."""
        y_test_binary = (y_test == 1).astype(int)
        y_pred = self.predict(X_test)
        return classification_report(
            y_test_binary, y_pred,
            target_names=['Pass', 'Fail']
        )


def create_ensemble_model(models: Dict, voting: str = 'soft') -> VotingClassifier:
    """
    Create an ensemble model from multiple trained models.
    
    Args:
        models: Dictionary of trained models
        voting: 'hard' or 'soft' voting
    
    Returns:
        VotingClassifier ensemble
    """
    estimators = [(name, model) for name, model in models.items()]
    return VotingClassifier(estimators=estimators, voting=voting)


if __name__ == "__main__":
    from .data_loader import load_secom_data
    from .preprocessing import preprocess_secom_data
    from .feature_selection import FeatureSelector
    
    # Load and preprocess
    features, labels, _ = load_secom_data()
    data = preprocess_secom_data(features, labels)
    
    # Feature selection
    selector = FeatureSelector(n_features=50)
    X_train_selected = selector.fit_transform(data['X_train'], data['y_train'])
    X_test_selected = selector.transform(data['X_test'])
    
    # Train and evaluate models
    predictor = YieldPredictor(handle_imbalance='smote')
    results = predictor.train_and_evaluate(
        X_train_selected, data['y_train'],
        X_test_selected, data['y_test']
    )
    
    print("\nModel Comparison Results:")
    print(results[['model_name', 'balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc']].to_string())
    
    print("\nClassification Report:")
    print(predictor.get_classification_report(X_test_selected, data['y_test']))
