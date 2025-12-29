"""
Visualization Module for SECOM Data Analysis
Creates insightful visualizations for equipment data analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple
import warnings

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_class_distribution(labels: pd.Series, 
                           save_path: Optional[str] = None) -> plt.Figure:
    """Plot class distribution showing imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    counts = labels.value_counts()
    colors = ['#2ecc71', '#e74c3c']
    labels_map = {-1: 'Pass', 1: 'Fail'}
    
    ax1 = axes[0]
    bars = ax1.bar([labels_map[i] for i in counts.index], counts.values, color=colors)
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count}', ha='center', fontsize=12)
    
    # Pie chart
    ax2 = axes[1]
    ax2.pie(counts.values, labels=[labels_map[i] for i in counts.index],
            autopct='%1.1f%%', colors=colors, explode=(0, 0.1),
            shadow=True, startangle=90)
    ax2.set_title('Class Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_missing_values(features: pd.DataFrame,
                       save_path: Optional[str] = None) -> plt.Figure:
    """Visualize missing value patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Missing values per feature
    missing_per_feature = features.isnull().sum().sort_values(ascending=False)
    missing_per_feature = missing_per_feature[missing_per_feature > 0]
    
    ax1 = axes[0]
    if len(missing_per_feature) > 30:
        missing_per_feature[:30].plot(kind='bar', ax=ax1, color='coral')
        ax1.set_title('Top 30 Features with Missing Values', fontsize=12, fontweight='bold')
    else:
        missing_per_feature.plot(kind='bar', ax=ax1, color='coral')
        ax1.set_title('Features with Missing Values', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Missing Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Missing values per sample
    missing_per_sample = features.isnull().sum(axis=1)
    ax2 = axes[1]
    ax2.hist(missing_per_sample, bins=50, color='steelblue', edgecolor='white')
    ax2.axvline(missing_per_sample.mean(), color='red', linestyle='--', 
                label=f'Mean: {missing_per_sample.mean():.0f}')
    ax2.set_title('Distribution of Missing Values per Sample', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Missing Values')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 20,
                           save_path: Optional[str] = None) -> plt.Figure:
    """Plot feature importance from multiple methods."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    top_features = importance_df.head(top_n)
    
    # Combined ranking
    ax1 = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))[::-1]
    bars = ax1.barh(range(top_n), top_features['avg_rank'].values[::-1], color=colors)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_features['feature'].values[::-1])
    ax1.set_xlabel('Average Rank (lower is better)')
    ax1.set_title(f'Top {top_n} Features by Combined Ranking', fontsize=12, fontweight='bold')
    
    # Rank comparison across methods
    ax2 = axes[1]
    rank_cols = [col for col in importance_df.columns if col.endswith('_rank') and col != 'final_rank']
    
    for col in rank_cols:
        method_name = col.replace('_rank', '').replace('_', ' ').title()
        ax2.scatter(range(top_n), 
                   top_features[col].values,
                   label=method_name, alpha=0.7, s=50)
    
    ax2.set_xticks(range(top_n))
    ax2.set_xticklabels(top_features['feature'].values, rotation=45, ha='right')
    ax2.set_ylabel('Rank')
    ax2.set_title('Feature Rankings Across Methods', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_model_comparison(results_df: pd.DataFrame,
                         save_path: Optional[str] = None) -> plt.Figure:
    """Compare model performance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = results_df['model_name'].values
    x = np.arange(len(models))
    
    # Balanced Accuracy
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax1.bar(x, results_df['balanced_accuracy'], color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Balanced Accuracy')
    ax1.set_title('Model Comparison: Balanced Accuracy', fontweight='bold')
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars, results_df['balanced_accuracy']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)
    
    # Sensitivity vs Specificity
    ax2 = axes[0, 1]
    width = 0.35
    ax2.bar(x - width/2, results_df['sensitivity'], width, label='Sensitivity (Fail Recall)', color='#e74c3c')
    ax2.bar(x + width/2, results_df['specificity'], width, label='Specificity (Pass Recall)', color='#2ecc71')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylabel('Score')
    ax2.set_title('Sensitivity vs Specificity', fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # ROC AUC
    ax3 = axes[1, 0]
    roc_auc = results_df['roc_auc'].fillna(0)
    bars = ax3.bar(x, roc_auc, color='steelblue')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_ylabel('ROC AUC')
    ax3.set_title('ROC AUC Score', fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax3.legend()
    
    # F1 Scores
    ax4 = axes[1, 1]
    ax4.bar(x - width/2, results_df['f1_fail'], width, label='F1 (Fail)', color='#e74c3c')
    ax4.bar(x + width/2, results_df['f1_pass'], width, label='F1 (Pass)', color='#2ecc71')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Scores by Class', fontweight='bold')
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_pca_visualization(X: pd.DataFrame, 
                          y: pd.Series,
                          save_path: Optional[str] = None) -> plt.Figure:
    """Visualize data in 2D using PCA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    ax1 = axes[0]
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=(y == 1).astype(int), 
                         cmap='RdYlGn_r', alpha=0.6, s=30)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('PCA Visualization', fontsize=12, fontweight='bold')
    ax1.legend(*scatter.legend_elements(), title="Class", labels=['Pass', 'Fail'])
    
    # Explained variance
    ax2 = axes[1]
    pca_full = PCA(n_components=min(50, X.shape[1]))
    pca_full.fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    
    ax2.bar(range(1, len(cumsum)+1), pca_full.explained_variance_ratio_, 
            alpha=0.6, label='Individual')
    ax2.plot(range(1, len(cumsum)+1), cumsum, 'r-', marker='o', 
             markersize=3, label='Cumulative')
    ax2.axhline(0.95, color='green', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
    ax2.legend()
    
    n_95 = np.argmax(cumsum >= 0.95) + 1
    ax2.axvline(n_95, color='green', linestyle=':', alpha=0.5)
    ax2.text(n_95 + 1, 0.5, f'{n_95} PCs for 95%', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_anomaly_detection_results(predictions: pd.DataFrame,
                                  y: pd.Series,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """Visualize anomaly detection results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    y_binary = (y == 1).astype(int)
    
    # Anomaly score distribution
    ax1 = axes[0]
    pass_scores = predictions[y_binary == 0]['anomaly_score']
    fail_scores = predictions[y_binary == 1]['anomaly_score']
    
    ax1.hist(pass_scores, bins=30, alpha=0.7, label='Pass', color='#2ecc71', density=True)
    ax1.hist(fail_scores, bins=30, alpha=0.7, label='Fail', color='#e74c3c', density=True)
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Anomaly Score Distribution by Class', fontweight='bold')
    ax1.legend()
    
    # Detection agreement across methods
    ax2 = axes[1]
    methods = [col for col in predictions.columns if col not in ['ensemble', 'anomaly_score']]
    agreement = predictions[methods].sum(axis=1)
    
    ax2.hist(agreement[y_binary == 0], bins=range(len(methods)+2), 
             alpha=0.7, label='Pass', color='#2ecc71', align='left')
    ax2.hist(agreement[y_binary == 1], bins=range(len(methods)+2), 
             alpha=0.7, label='Fail', color='#e74c3c', align='left')
    ax2.set_xlabel('Number of Methods Detecting Anomaly')
    ax2.set_ylabel('Count')
    ax2.set_title('Detection Agreement Across Methods', fontweight='bold')
    ax2.legend()
    ax2.set_xticks(range(len(methods)+1))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def create_analysis_report_figures(data: dict, 
                                  output_dir: str = None) -> List[plt.Figure]:
    """Create all analysis figures for the report."""
    figures = []
    
    # This would create all figures and optionally save them
    # Called from the main pipeline
    
    return figures


if __name__ == "__main__":
    print("Visualization module loaded. Use functions to create plots.")
