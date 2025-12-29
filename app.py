"""
Streamlit Dashboard for SECOM Equipment Data Analysis
Interactive visualization and exploration of semiconductor manufacturing data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import SECOM_DATA_FILE, SECOM_LABELS_FILE, OUTPUT_DIR
from src.data_loader import load_secom_data, get_data_summary
from src.preprocessing import SECOMPreprocessor
from src.feature_selection import FeatureSelector
from src.models import YieldPredictor
from src.anomaly_detection import AnomalyDetector
from src.ai_insights import AIInsightsGenerator, AnalysisContext

# Page config
st.set_page_config(
    page_title="Equipment Data Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the SECOM dataset."""
    features, labels, timestamps = load_secom_data()
    summary = get_data_summary(features, labels)
    return features, labels, timestamps, summary


@st.cache_data
def preprocess_data(_features, _labels):
    """Preprocess and cache the data."""
    preprocessor = SECOMPreprocessor()
    X_processed = preprocessor.fit_transform(_features, _labels)
    return X_processed, preprocessor


@st.cache_data
def run_feature_selection(_X, _y, n_features=50):
    """Run feature selection and cache results."""
    selector = FeatureSelector(n_features=n_features)
    selector.fit(_X, _y)
    importance = selector.get_feature_importance_report()
    return selector, importance


def main():
    # Header
    st.markdown('<p class="main-header">Equipment Data Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Driven Semiconductor Manufacturing Data Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Load data
    with st.spinner("Loading SECOM dataset..."):
        try:
            features, labels, timestamps, summary = load_data()
            data_loaded = True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please ensure the SECOM dataset is in the 'secom' directory.")
            data_loaded = False
            return
    
    # Sidebar options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Options")
    n_features = st.sidebar.slider("Number of features to select", 10, 100, 50)
    show_raw_data = st.sidebar.checkbox("Show raw data sample", False)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", 
        "Preprocessing", 
        "Feature Analysis",
        "Model Performance",
        "Anomaly Detection",
        "AI Insights"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Dataset Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", f"{summary['n_samples']:,}")
        with col2:
            st.metric("Total Features", f"{summary['n_features']:,}")
        with col3:
            st.metric("Fail Rate", f"{summary['fail_rate']:.1f}%")
        with col4:
            st.metric("Missing Values", f"{summary['missing_percentage']:.1f}%")
        
        st.markdown("---")
        
        # Class distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Class Distribution")
            class_data = pd.DataFrame({
                'Class': ['Pass', 'Fail'],
                'Count': [summary['n_pass'], summary['n_fail']],
                'Percentage': [100 - summary['fail_rate'], summary['fail_rate']]
            })
            
            fig = px.pie(class_data, values='Count', names='Class',
                        color='Class',
                        color_discrete_map={'Pass': '#2ecc71', 'Fail': '#e74c3c'},
                        hole=0.4)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Data Quality Metrics")
            quality_data = pd.DataFrame({
                'Metric': ['Features with Missing', 'Constant Features', 
                          'Avg Missing/Sample', 'Max Missing/Sample'],
                'Value': [
                    summary['features_with_missing'],
                    summary['constant_features'],
                    f"{summary['avg_missing_per_sample']:.1f}",
                    summary['max_missing_per_sample']
                ]
            })
            st.dataframe(quality_data, use_container_width=True, hide_index=True)
            
            # Imbalance warning
            st.warning(f"**Class Imbalance**: {summary['imbalance_ratio']:.0f} to 1 ratio (Pass to Fail)")
        
        # Timeline
        st.subheader("Data Collection Timeline")
        timeline_df = pd.DataFrame({
            'timestamp': timestamps,
            'label': labels.map({-1: 'Pass', 1: 'Fail'})
        })
        timeline_df['date'] = pd.to_datetime(timeline_df['timestamp']).dt.date
        daily_counts = timeline_df.groupby(['date', 'label']).size().reset_index(name='count')
        
        fig = px.bar(daily_counts, x='date', y='count', color='label',
                    color_discrete_map={'Pass': '#2ecc71', 'Fail': '#e74c3c'},
                    title='Daily Sample Collection')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        if show_raw_data:
            st.subheader("Raw Data Sample")
            st.dataframe(features.head(10))
    
    # Tab 2: Preprocessing
    with tab2:
        st.header("Data Preprocessing")
        
        with st.spinner("Preprocessing data..."):
            X_processed, preprocessor = preprocess_data(features, labels)
        
        report = preprocessor.get_preprocessing_report()
        
        # Preprocessing summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Reduction")
            reduction_data = {
                'Stage': ['Original', 'After High Missing Removal', 
                         'After Constant Removal', 'After Correlation Removal'],
                'Features': [
                    report['original_features'],
                    report['original_features'] - report['removed_high_missing'],
                    report['original_features'] - report['removed_high_missing'] - report['removed_constant'],
                    report['final_features']
                ]
            }
            fig = px.funnel(pd.DataFrame(reduction_data), x='Features', y='Stage')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Preprocessing Steps")
            st.markdown(f"""
            | Step | Action | Features Removed |
            |------|--------|-----------------|
            | 1 | Remove high missing (>{report['missing_threshold']*100:.0f}%) | {report['removed_high_missing']} |
            | 2 | Remove constant features | {report['removed_constant']} |
            | 3 | Imputation ({report['imputation_strategy']}) | 0 |
            | 4 | Remove correlated (>{report['correlation_threshold']*100:.0f}%) | {report['removed_correlated']} |
            | 5 | Scaling ({report['scaling_method']}) | 0 |
            """)
            
            st.success(f"✅ Final feature count: **{report['final_features']}** (from {report['original_features']})")
        
        # Missing value visualization
        st.subheader("Missing Value Analysis")
        missing_per_feature = features.isnull().sum().sort_values(ascending=False)
        missing_per_feature = missing_per_feature[missing_per_feature > 0].head(30)
        
        fig = px.bar(x=missing_per_feature.index, y=missing_per_feature.values,
                    labels={'x': 'Feature', 'y': 'Missing Count'},
                    title='Top 30 Features by Missing Values')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Feature Analysis
    with tab3:
        st.header("Feature Importance Analysis")
        
        with st.spinner("Running feature selection..."):
            X_processed, _ = preprocess_data(features, labels)
            selector, importance = run_feature_selection(X_processed, labels, n_features)
        
        top_n_display = st.slider("Number of top features to display", 10, 50, 20)
        
        # Top features bar chart
        top_features = importance.head(top_n_display)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Features by Combined Ranking")
            fig = px.bar(top_features, x='avg_rank', y='feature',
                        orientation='h',
                        color='avg_rank',
                        color_continuous_scale='RdYlGn_r')
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Ranking Comparison Across Methods")
            rank_cols = [col for col in importance.columns if col.endswith('_rank') and 'final' not in col and 'avg' not in col]
            
            fig = go.Figure()
            for col in rank_cols:
                method_name = col.replace('_rank', '').replace('_', ' ').title()
                fig.add_trace(go.Scatter(
                    x=top_features['feature'],
                    y=top_features[col],
                    mode='markers',
                    name=method_name,
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                height=600,
                yaxis_title='Rank (lower is better)',
                xaxis_tickangle=45,
                yaxis_autorange='reversed'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.subheader("Detailed Feature Rankings")
        display_cols = ['feature', 'avg_rank', 'final_rank'] + rank_cols
        st.dataframe(
            importance[display_cols].head(top_n_display).style.background_gradient(
                subset=['avg_rank'], cmap='RdYlGn_r'
            ),
            use_container_width=True,
            hide_index=True
        )
    
    # Tab 4: Model Performance (placeholder - would need training)
    with tab4:
        st.header("Model Performance")
        
        st.info("Model training requires more computational resources. Click below to see sample results.")
        
        if st.button("Show Sample Model Results"):
            # Sample results for demonstration
            sample_results = pd.DataFrame({
                'Model': ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 
                         'SVM', 'Neural Network'],
                'Balanced Accuracy': [0.72, 0.70, 0.68, 0.65, 0.69],
                'Sensitivity': [0.65, 0.62, 0.60, 0.55, 0.63],
                'Specificity': [0.79, 0.78, 0.76, 0.75, 0.75],
                'ROC AUC': [0.78, 0.76, 0.74, 0.72, 0.75]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(sample_results, x='Model', y='Balanced Accuracy',
                            color='Balanced Accuracy',
                            color_continuous_scale='Viridis',
                            title='Model Comparison: Balanced Accuracy')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Sensitivity', x=sample_results['Model'], 
                                    y=sample_results['Sensitivity'], marker_color='#e74c3c'))
                fig.add_trace(go.Bar(name='Specificity', x=sample_results['Model'], 
                                    y=sample_results['Specificity'], marker_color='#2ecc71'))
                fig.update_layout(barmode='group', title='Sensitivity vs Specificity')
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(sample_results, use_container_width=True, hide_index=True)
    
    # Tab 5: Anomaly Detection
    with tab5:
        st.header("Anomaly Detection for Equipment Monitoring")
        
        st.markdown("""
        Anomaly detection helps identify unusual equipment behavior that may indicate:
        - **Potential failures** before they occur
        - **Process drift** requiring calibration
        - **Quality issues** in production
        """)
        
        if st.button("Run Anomaly Detection Analysis"):
            with st.spinner("Running anomaly detection..."):
                X_processed, _ = preprocess_data(features, labels)
                
                # Simple anomaly detection demo
                from sklearn.ensemble import IsolationForest
                
                detector = IsolationForest(contamination=0.1, random_state=42)
                predictions = detector.fit_predict(X_processed)
                
                # Results
                n_anomalies = (predictions == -1).sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(predictions))
                with col2:
                    st.metric("Detected Anomalies", n_anomalies)
                with col3:
                    st.metric("Anomaly Rate", f"{n_anomalies/len(predictions)*100:.1f}%")
                
                # Comparison with actual failures
                y_binary = (labels == 1).astype(int)
                pred_binary = (predictions == -1).astype(int)
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_binary, pred_binary)
                
                fig = px.imshow(cm, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Normal', 'Anomaly'],
                               y=['Pass', 'Fail'],
                               text_auto=True,
                               color_continuous_scale='Blues')
                fig.update_layout(title='Anomaly Detection vs Actual Failures')
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 6: AI Insights
    with tab6:
        st.header("AI-Powered Insights")
        
        st.markdown("""
        This module uses AI to generate actionable insights from your analysis results.
        It can provide executive summaries, feature interpretations, and recommendations.
        """)
        
        # Check for OpenAI API key
        api_key = st.text_input("OpenAI API Key (optional)", type="password", 
                                help="Enter your OpenAI API key for GPT-powered insights. Leave empty for rule-based analysis.")
        
        if st.button("Generate AI Insights"):
            with st.spinner("Generating insights..."):
                # Preprocess data for context
                X_processed, preprocessor = preprocess_data(features, labels)
                selector, importance = run_feature_selection(X_processed, labels, n_features)
                
                # Create context
                context = AnalysisContext(
                    data_summary=summary,
                    preprocessing_report=preprocessor.get_preprocessing_report(),
                    top_features=selector.get_top_features(),
                    feature_importance=importance,
                    model_results=pd.DataFrame({
                        'model_name': ['Logistic Regression', 'Random Forest'],
                        'balanced_accuracy': [0.70, 0.62],
                        'sensitivity': [0.62, 0.29],
                        'specificity': [0.78, 0.95],
                        'roc_auc': [0.74, 0.75]
                    }),
                    anomaly_results={},
                    best_model='Logistic Regression'
                )
                
                # Generate insights
                generator = AIInsightsGenerator(api_key=api_key if api_key else None)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Executive Summary")
                    summary_text = generator.generate_executive_summary(context)
                    st.markdown(summary_text)
                
                with col2:
                    st.subheader("AI Recommendations")
                    recommendations = generator.generate_recommendations(context)
                    for rec in recommendations:
                        priority = rec.get('priority', 'medium')
                        color = {'high': '#ff4b4b', 'medium': '#ffa500', 'low': '#21c354'}.get(priority, 'gray')
                        st.markdown(f"""
                        <div style='padding: 12px; margin: 8px 0; border-left: 4px solid {color}; 
                                    background: rgba(255,255,255,0.05); border-radius: 4px;'>
                            <span style='color: {color}; font-weight: bold;'>[{priority.upper()}]</span> 
                            <span style='font-weight: 600;'>{rec.get('category', 'General')}</span><br>
                            <span style='opacity: 0.9;'>{rec.get('recommendation', '')}</span><br>
                            <span style='opacity: 0.7; font-style: italic;'>Expected Impact: {rec.get('expected_impact', 'N/A')}</span>
                        </div>
                        """, unsafe_allow_html=True)
        
        # AI Chat Interface
        st.markdown("---")
        st.subheader("Ask AI About Your Data")
        
        user_question = st.text_input("Ask a question about the analysis:", 
                                      placeholder="e.g., Which sensors should we monitor most closely?")
        
        if user_question and st.button("Get Answer"):
            with st.spinner("Thinking..."):
                X_processed, preprocessor = preprocess_data(features, labels)
                selector, importance = run_feature_selection(X_processed, labels, n_features)
                
                context = AnalysisContext(
                    data_summary=summary,
                    preprocessing_report=preprocessor.get_preprocessing_report(),
                    top_features=selector.get_top_features(),
                    feature_importance=importance,
                    model_results=pd.DataFrame({
                        'model_name': ['Logistic Regression'],
                        'balanced_accuracy': [0.70],
                        'sensitivity': [0.62],
                        'specificity': [0.78],
                        'roc_auc': [0.74]
                    }),
                    anomaly_results={},
                    best_model='Logistic Regression'
                )
                
                generator = AIInsightsGenerator(api_key=api_key if api_key else None)
                answer = generator.answer_question(user_question, context)
                st.info(answer)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Equipment Data Analysis Project | AI-Driven Semiconductor Manufacturing</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
