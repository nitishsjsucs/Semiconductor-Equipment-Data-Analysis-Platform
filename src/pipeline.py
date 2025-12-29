"""
Main Pipeline Module for Semiconductor Data Analysis
Orchestrates the complete AI-driven data processing workflow
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings

from .config import OUTPUT_DIR, MODELS_DIR
from .data_loader import load_secom_data, get_data_summary, print_data_summary
from .preprocessing import SECOMPreprocessor, preprocess_secom_data
from .feature_selection import FeatureSelector, analyze_feature_importance
from .models import YieldPredictor
from .anomaly_detection import AnomalyDetector, detect_equipment_anomalies
from . import visualization as viz
from .ai_insights import AIInsightsGenerator, AnalysisContext, generate_ai_report


class SECOMPipeline:
    """
    Complete AI-driven pipeline for semiconductor equipment data analysis.
    
    This pipeline demonstrates how AI can transform manual, rigid data 
    processing into automated, intelligent workflows.
    """
    
    def __init__(self, 
                 n_features: int = 50,
                 imbalance_strategy: str = 'smote',
                 save_outputs: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            n_features: Number of top features to select
            imbalance_strategy: Strategy for handling class imbalance
            save_outputs: Whether to save outputs to disk
        """
        self.n_features = n_features
        self.imbalance_strategy = imbalance_strategy
        self.save_outputs = save_outputs
        
        # Pipeline components
        self.preprocessor = None
        self.feature_selector = None
        self.predictor = None
        self.anomaly_detector = None
        
        # Results storage
        self.data_summary = None
        self.preprocessing_report = None
        self.feature_importance = None
        self.model_results = None
        self.anomaly_results = None
        
        # Timestamps
        self.run_timestamp = None
        
    def run(self, data_path=None, labels_path=None) -> dict:
        """
        Execute the complete analysis pipeline.
        
        Args:
            data_path: Path to secom.data (optional)
            labels_path: Path to secom_labels.data (optional)
        
        Returns:
            Dictionary with all results
        """
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*70)
        print("SEMICONDUCTOR EQUIPMENT DATA ANALYSIS PIPELINE")
        print("   AI-Driven Manufacturing Data Processing")
        print("="*70)
        
        # Step 1: Load Data
        print("\n[STEP 1] Loading Dataset")
        print("-" * 50)
        features, labels, timestamps = load_secom_data(data_path, labels_path)
        self.data_summary = get_data_summary(features, labels)
        print_data_summary(self.data_summary)
        
        # Step 2: Preprocess Data
        print("\n[STEP 2] Preprocessing Data")
        print("-" * 50)
        data = preprocess_secom_data(features, labels)
        self.preprocessor = data['preprocessor']
        self.preprocessing_report = data['report']
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        print(f"\nPreprocessing Summary:")
        print(f"   Original features: {self.preprocessing_report['original_features']}")
        print(f"   Final features: {self.preprocessing_report['final_features']}")
        print(f"   Removed (high missing): {self.preprocessing_report['removed_high_missing']}")
        print(f"   Removed (constant): {self.preprocessing_report['removed_constant']}")
        print(f"   Removed (correlated): {self.preprocessing_report['removed_correlated']}")
        
        # Step 3: Feature Selection
        print("\n[STEP 3] Feature Selection")
        print("-" * 50)
        self.feature_selector = FeatureSelector(n_features=self.n_features)
        self.feature_selector.fit(X_train, y_train)
        
        top_features = self.feature_selector.get_top_features()
        self.feature_importance = self.feature_selector.get_feature_importance_report()
        
        X_train_selected = self.feature_selector.transform(X_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        print(f"\nTop 10 Most Important Sensors:")
        for i, feat in enumerate(top_features[:10], 1):
            rank = self.feature_importance[
                self.feature_importance['feature'] == feat
            ]['avg_rank'].values[0]
            print(f"   {i:2d}. {feat} (avg rank: {rank:.1f})")
        
        # Step 4: Model Training
        print("\n[STEP 4] Training Predictive Models")
        print("-" * 50)
        self.predictor = YieldPredictor(handle_imbalance=self.imbalance_strategy)
        self.model_results = self.predictor.train_and_evaluate(
            X_train_selected, y_train,
            X_test_selected, y_test
        )
        
        print(f"\nModel Performance Summary:")
        print(self.model_results[[
            'model_name', 'balanced_accuracy', 'sensitivity', 
            'specificity', 'roc_auc'
        ]].to_string(index=False))
        
        # Step 5: Anomaly Detection
        print("\n[STEP 5] Anomaly Detection Analysis")
        print("-" * 50)
        self.anomaly_results = detect_equipment_anomalies(
            X_train_selected, y_train
        )
        
        # Step 6: AI-Powered Insights
        print("\n[STEP 6] AI-Powered Insights Generation")
        print("-" * 50)
        self.ai_report = self._generate_ai_insights()
        
        # Step 7: Generate Visualizations
        if self.save_outputs:
            print("\n[STEP 7] Generating Visualizations")
            print("-" * 50)
            self._generate_visualizations(features, labels, X_train_selected, y_train)
        
        # Step 8: Save Results
        if self.save_outputs:
            print("\n[STEP 8] Saving Results")
            print("-" * 50)
            self._save_results()
        
        # Final Summary
        self._print_final_summary()
        
        return self._compile_results()
    
    def _generate_visualizations(self, features, labels, X_selected, y):
        """Generate and save all visualizations."""
        output_dir = OUTPUT_DIR / self.run_timestamp
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Class distribution
            fig = viz.plot_class_distribution(labels, str(output_dir / 'class_distribution.png'))
            print(f"   [OK] Saved class_distribution.png")
            
            # Missing values
            fig = viz.plot_missing_values(features, str(output_dir / 'missing_values.png'))
            print(f"   [OK] Saved missing_values.png")
            
            # Feature importance
            fig = viz.plot_feature_importance(
                self.feature_importance, 
                top_n=20,
                save_path=str(output_dir / 'feature_importance.png')
            )
            print(f"   [OK] Saved feature_importance.png")
            
            # Model comparison
            fig = viz.plot_model_comparison(
                self.model_results,
                save_path=str(output_dir / 'model_comparison.png')
            )
            print(f"   [OK] Saved model_comparison.png")
            
            # PCA visualization
            fig = viz.plot_pca_visualization(
                X_selected, y,
                save_path=str(output_dir / 'pca_visualization.png')
            )
            print(f"   [OK] Saved pca_visualization.png")
            
            # Anomaly detection
            fig = viz.plot_anomaly_detection_results(
                self.anomaly_results['predictions'], y,
                save_path=str(output_dir / 'anomaly_detection.png')
            )
            print(f"   [OK] Saved anomaly_detection.png")
            
            import matplotlib.pyplot as plt
            plt.close('all')
            
        except Exception as e:
            print(f"   [WARN] Could not generate some visualizations: {e}")
    
    def _generate_ai_insights(self) -> dict:
        """Generate AI-powered insights from analysis results."""
        context = AnalysisContext(
            data_summary=self.data_summary,
            preprocessing_report=self.preprocessing_report,
            top_features=self.feature_selector.get_top_features(),
            feature_importance=self.feature_importance,
            model_results=self.model_results,
            anomaly_results=self.anomaly_results,
            best_model=self.predictor.best_model_name_
        )
        
        report = generate_ai_report(context)
        
        # Print recommendations
        if report.get('recommendations'):
            print("\nAI-Generated Recommendations:")
            for i, rec in enumerate(report['recommendations'][:3], 1):
                priority = rec.get('priority', 'medium').upper()
                print(f"   {i}. [{priority}] {rec.get('recommendation', '')}")
        
        return report
    
    def _save_results(self):
        """Save all results to disk."""
        output_dir = OUTPUT_DIR / self.run_timestamp
        output_dir.mkdir(exist_ok=True)
        
        # Save feature importance
        self.feature_importance.to_csv(
            output_dir / 'feature_importance.csv', index=False
        )
        print(f"   [OK] Saved feature_importance.csv")
        
        # Save model results
        self.model_results.to_csv(
            output_dir / 'model_results.csv', index=False
        )
        print(f"   [OK] Saved model_results.csv")
        
        # Save anomaly detection report
        self.anomaly_results['report'].to_csv(
            output_dir / 'anomaly_detection_report.csv'
        )
        print(f"   [OK] Saved anomaly_detection_report.csv")
        
        # Save summary report
        summary = {
            'run_timestamp': self.run_timestamp,
            'data_summary': self.data_summary,
            'preprocessing_report': self.preprocessing_report,
            'best_model': self.predictor.best_model_name_,
            'best_model_accuracy': float(self.model_results.iloc[0]['balanced_accuracy']),
            'top_10_features': self.feature_selector.get_top_features(10)
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   [OK] Saved summary.json")
        
        # Save AI insights
        if hasattr(self, 'ai_report') and self.ai_report:
            ai_summary = {
                'ai_available': self.ai_report.get('ai_available', False),
                'recommendations': self.ai_report.get('recommendations', [])
            }
            with open(output_dir / 'ai_insights.json', 'w') as f:
                json.dump(ai_summary, f, indent=2, default=str)
            print(f"   [OK] Saved ai_insights.json")
            
            # Save executive summary as markdown
            if self.ai_report.get('executive_summary'):
                with open(output_dir / 'executive_summary.md', 'w') as f:
                    f.write(self.ai_report['executive_summary'])
                print(f"   [OK] Saved executive_summary.md")
        
        # Save best model
        self.predictor.save_best_model()
    
    def _print_final_summary(self):
        """Print final analysis summary."""
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE - FINAL SUMMARY")
        print("="*70)
        
        print(f"\nDataset Analysis:")
        print(f"   - Total samples: {self.data_summary['n_samples']:,}")
        print(f"   - Original features: {self.data_summary['n_features']}")
        print(f"   - Features after preprocessing: {self.preprocessing_report['final_features']}")
        print(f"   - Selected features: {self.n_features}")
        
        print(f"\nClass Imbalance:")
        print(f"   - Pass samples: {self.data_summary['n_pass']:,} ({100-self.data_summary['fail_rate']:.1f}%)")
        print(f"   - Fail samples: {self.data_summary['n_fail']:,} ({self.data_summary['fail_rate']:.1f}%)")
        print(f"   - Imbalance ratio: {self.data_summary['imbalance_ratio']:.1f}:1")
        
        print(f"\nBest Model: {self.predictor.best_model_name_}")
        best_result = self.model_results.iloc[0]
        print(f"   - Balanced Accuracy: {best_result['balanced_accuracy']:.3f}")
        print(f"   - Sensitivity (Fail Detection): {best_result['sensitivity']:.3f}")
        print(f"   - Specificity (Pass Detection): {best_result['specificity']:.3f}")
        if best_result['roc_auc']:
            print(f"   - ROC AUC: {best_result['roc_auc']:.3f}")
        
        print(f"\nTop 5 Key Sensor Signals:")
        for i, feat in enumerate(self.feature_selector.get_top_features(5), 1):
            print(f"   {i}. {feat}")
        
        if self.save_outputs:
            print(f"\nResults saved to: {OUTPUT_DIR / self.run_timestamp}")
        
        print("\n" + "="*70)
    
    def _compile_results(self) -> dict:
        """Compile all results into a single dictionary."""
        results = {
            'data_summary': self.data_summary,
            'preprocessing_report': self.preprocessing_report,
            'feature_importance': self.feature_importance,
            'model_results': self.model_results,
            'anomaly_results': self.anomaly_results,
            'best_model': self.predictor.best_model_name_,
            'top_features': self.feature_selector.get_top_features(),
            'run_timestamp': self.run_timestamp
        }
        
        # Include AI insights if available
        if hasattr(self, 'ai_report') and self.ai_report:
            results['ai_report'] = self.ai_report
        
        return results


def run_analysis():
    """Run the complete SECOM analysis pipeline."""
    pipeline = SECOMPipeline(
        n_features=50,
        imbalance_strategy='smote',
        save_outputs=True
    )
    results = pipeline.run()
    return results


if __name__ == "__main__":
    results = run_analysis()
