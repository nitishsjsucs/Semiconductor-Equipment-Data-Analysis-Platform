"""
Feature Selection Module for SECOM Dataset
Implements multiple feature selection techniques to identify key sensor signals
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, 
    f_classif, 
    mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, pearsonr
from typing import List, Tuple, Optional, Dict
import warnings

from .config import RANDOM_STATE, N_TOP_FEATURES


class FeatureSelector:
    """
    Multi-method feature selection for semiconductor manufacturing data.
    
    Implements various techniques to identify the most relevant sensor signals
    that impact yield, enabling engineers to focus on key process parameters.
    """
    
    def __init__(self, n_features: int = N_TOP_FEATURES, random_state: int = RANDOM_STATE):
        """
        Initialize feature selector.
        
        Args:
            n_features: Number of top features to select
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.random_state = random_state
        self.feature_rankings_: Dict[str, pd.DataFrame] = {}
        self.combined_ranking_: Optional[pd.DataFrame] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit all feature selection methods and compute rankings.
        
        Args:
            X: Preprocessed feature DataFrame
            y: Labels Series
        
        Returns:
            self
        """
        print("\n[FEATURE] Running feature selection methods...")
        
        # Convert labels to binary (0/1)
        y_binary = (y == 1).astype(int)
        
        # Method 1: F-test (ANOVA)
        self.feature_rankings_['f_test'] = self._rank_by_ftest(X, y_binary)
        print("   [OK] F-test ranking complete")
        
        # Method 2: Mutual Information
        self.feature_rankings_['mutual_info'] = self._rank_by_mutual_info(X, y_binary)
        print("   [OK] Mutual Information ranking complete")
        
        # Method 3: Random Forest Importance
        self.feature_rankings_['random_forest'] = self._rank_by_random_forest(X, y_binary)
        print("   [OK] Random Forest importance complete")
        
        # Method 4: Correlation with target
        self.feature_rankings_['correlation'] = self._rank_by_correlation(X, y_binary)
        print("   [OK] Correlation ranking complete")
        
        # Method 5: Gradient Boosting Importance
        self.feature_rankings_['gradient_boosting'] = self._rank_by_gradient_boosting(X, y_binary)
        print("   [OK] Gradient Boosting importance complete")
        
        # Combine rankings using rank aggregation
        self.combined_ranking_ = self._combine_rankings()
        print(f"   [OK] Combined ranking computed for {len(X.columns)} features")
        
        return self
    
    def _rank_by_ftest(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Rank features using F-test (ANOVA)."""
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_,
            'pvalue': selector.pvalues_
        })
        scores['rank'] = scores['score'].rank(ascending=False)
        return scores.sort_values('rank')
    
    def _rank_by_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Rank features using Mutual Information."""
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': mi_scores
        })
        scores['rank'] = scores['score'].rank(ascending=False)
        return scores.sort_values('rank')
    
    def _rank_by_random_forest(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Rank features using Random Forest feature importance."""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X, y)
        
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': rf.feature_importances_
        })
        scores['rank'] = scores['score'].rank(ascending=False)
        return scores.sort_values('rank')
    
    def _rank_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Rank features by absolute correlation with target."""
        correlations = []
        for col in X.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = spearmanr(X[col], y)
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': correlations
        })
        scores['rank'] = scores['score'].rank(ascending=False)
        return scores.sort_values('rank')
    
    def _rank_by_gradient_boosting(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Rank features using Gradient Boosting feature importance."""
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.random_state
        )
        gb.fit(X, y)
        
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': gb.feature_importances_
        })
        scores['rank'] = scores['score'].rank(ascending=False)
        return scores.sort_values('rank')
    
    def _combine_rankings(self) -> pd.DataFrame:
        """Combine rankings from all methods using average rank."""
        all_ranks = pd.DataFrame({'feature': self.feature_rankings_['f_test']['feature']})
        
        for method, ranking in self.feature_rankings_.items():
            all_ranks[f'{method}_rank'] = ranking.set_index('feature').loc[all_ranks['feature'], 'rank'].values
        
        # Calculate average rank
        rank_columns = [col for col in all_ranks.columns if col.endswith('_rank')]
        all_ranks['avg_rank'] = all_ranks[rank_columns].mean(axis=1)
        all_ranks['final_rank'] = all_ranks['avg_rank'].rank()
        
        return all_ranks.sort_values('final_rank')
    
    def get_top_features(self, n: Optional[int] = None) -> List[str]:
        """Get top n features based on combined ranking."""
        n = n or self.n_features
        if self.combined_ranking_ is None:
            raise RuntimeError("Must call fit() before getting top features")
        return self.combined_ranking_.head(n)['feature'].tolist()
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Generate detailed feature importance report."""
        if self.combined_ranking_ is None:
            raise RuntimeError("Must call fit() before getting report")
        
        report = self.combined_ranking_.copy()
        
        # Add individual method scores
        for method, ranking in self.feature_rankings_.items():
            ranking_indexed = ranking.set_index('feature')
            report[f'{method}_score'] = report['feature'].map(ranking_indexed['score'])
        
        return report
    
    def transform(self, X: pd.DataFrame, n: Optional[int] = None) -> pd.DataFrame:
        """Select top features from the data."""
        top_features = self.get_top_features(n)
        return X[top_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, n: Optional[int] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X, n)


def analyze_feature_importance(X: pd.DataFrame, y: pd.Series, 
                               n_top: int = 20) -> Tuple[List[str], pd.DataFrame]:
    """
    Analyze and rank feature importance for the SECOM dataset.
    
    This function demonstrates how AI can automate the identification of
    key sensor signals, replacing tedious manual analysis.
    
    Args:
        X: Preprocessed feature DataFrame
        y: Labels Series
        n_top: Number of top features to return
    
    Returns:
        Tuple of (top feature names, full importance report)
    """
    selector = FeatureSelector(n_features=n_top)
    selector.fit(X, y)
    
    top_features = selector.get_top_features(n_top)
    report = selector.get_feature_importance_report()
    
    print(f"\nTop {n_top} Most Important Sensor Signals:")
    print("-" * 50)
    for i, feat in enumerate(top_features[:10], 1):
        avg_rank = report[report['feature'] == feat]['avg_rank'].values[0]
        print(f"   {i:2d}. {feat} (avg rank: {avg_rank:.1f})")
    
    return top_features, report


if __name__ == "__main__":
    from .data_loader import load_secom_data
    from .preprocessing import preprocess_secom_data
    
    # Load and preprocess data
    features, labels, _ = load_secom_data()
    data = preprocess_secom_data(features, labels)
    
    # Analyze feature importance
    top_features, report = analyze_feature_importance(
        data['X_train'], 
        data['y_train'],
        n_top=30
    )
    
    print("\n📋 Full Feature Importance Report (top 10):")
    print(report.head(10).to_string())
