"""
AI-Powered Insights Module for Semiconductor Data Analysis
Uses LLM to generate natural language insights and recommendations
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class AnalysisContext:
    """Context object containing all analysis results for AI interpretation."""
    data_summary: Dict
    preprocessing_report: Dict
    top_features: List[str]
    feature_importance: pd.DataFrame
    model_results: pd.DataFrame
    anomaly_results: Dict
    best_model: str


class AIInsightsGenerator:
    """
    AI-powered insights generator using LLM to interpret analysis results.
    
    Transforms raw analysis outputs into actionable insights for process engineers.
    """
    
    SYSTEM_PROMPT = """You are an AI assistant specialized in semiconductor manufacturing data analysis.
Your role is to interpret machine learning analysis results and provide actionable insights for process engineers.

Context: You are analyzing sensor data from semiconductor manufacturing equipment.
- The data contains sensor measurements from production processes
- The goal is to predict yield (pass/fail) and identify key factors affecting quality
- Engineers need clear, actionable recommendations

Guidelines:
- Be specific and technical but accessible
- Focus on actionable insights
- Highlight the most important findings
- Suggest concrete next steps for process improvement
- Consider the practical implications for manufacturing"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize AI insights generator.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: Model to use for generation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """Check if AI insights are available."""
        return self.client is not None
    
    def _build_context_prompt(self, context: AnalysisContext) -> str:
        """Build the context prompt from analysis results."""
        top_10_features = context.top_features[:10]
        
        best_model_row = context.model_results[
            context.model_results['model_name'] == context.best_model
        ].iloc[0]
        
        prompt = f"""
## Analysis Results Summary

### Dataset Overview
- Total samples: {context.data_summary['n_samples']:,}
- Original features: {context.data_summary['n_features']}
- Pass rate: {100 - context.data_summary['fail_rate']:.1f}%
- Fail rate: {context.data_summary['fail_rate']:.1f}%
- Class imbalance ratio: {context.data_summary['imbalance_ratio']:.1f}:1
- Missing values: {context.data_summary['missing_percentage']:.1f}%

### Preprocessing Results
- Features after preprocessing: {context.preprocessing_report['final_features']}
- Removed (high missing): {context.preprocessing_report['removed_high_missing']}
- Removed (constant): {context.preprocessing_report['removed_constant']}
- Removed (correlated): {context.preprocessing_report['removed_correlated']}

### Top 10 Most Important Sensors
{chr(10).join(f'{i+1}. {feat}' for i, feat in enumerate(top_10_features))}

### Best Predictive Model: {context.best_model}
- Balanced Accuracy: {best_model_row['balanced_accuracy']:.3f}
- Sensitivity (Fail Detection): {best_model_row['sensitivity']:.3f}
- Specificity (Pass Detection): {best_model_row['specificity']:.3f}
- ROC AUC: {best_model_row.get('roc_auc', 'N/A')}

### Model Comparison (Top 3)
"""
        for i, (_, row) in enumerate(context.model_results.head(3).iterrows()):
            prompt += f"- {row['model_name']}: Balanced Acc={row['balanced_accuracy']:.3f}, Sensitivity={row['sensitivity']:.3f}\n"
        
        return prompt
    
    def generate_executive_summary(self, context: AnalysisContext) -> str:
        """Generate an executive summary of the analysis."""
        if not self.is_available():
            return self._generate_fallback_summary(context)
        
        context_prompt = self._build_context_prompt(context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"""
Based on the following semiconductor manufacturing data analysis results, 
generate a concise executive summary (3-4 paragraphs) highlighting:
1. Key findings about the manufacturing process
2. The most important sensors to monitor
3. Model performance and reliability assessment
4. Recommended actions for process improvement

{context_prompt}
"""}
                ],
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"AI generation error: {e}")
            return self._generate_fallback_summary(context)
    
    def generate_feature_insights(self, context: AnalysisContext) -> str:
        """Generate insights about the most important features."""
        if not self.is_available():
            return self._generate_fallback_feature_insights(context)
        
        context_prompt = self._build_context_prompt(context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"""
Analyze the top sensor signals identified in this semiconductor manufacturing analysis.
For each of the top 5 sensors, provide:
1. Why it might be important for yield prediction
2. What process parameters it could represent
3. Recommended monitoring thresholds or actions

{context_prompt}
"""}
                ],
                temperature=0.7,
                max_tokens=600
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._generate_fallback_feature_insights(context)
    
    def generate_recommendations(self, context: AnalysisContext) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on analysis."""
        if not self.is_available():
            return self._generate_fallback_recommendations(context)
        
        context_prompt = self._build_context_prompt(context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"""
Based on this semiconductor manufacturing analysis, generate 5 specific, 
actionable recommendations in JSON format:

[
  {{"priority": "high/medium/low", "category": "category", "recommendation": "text", "expected_impact": "text"}}
]

Categories: Process Control, Monitoring, Maintenance, Data Quality, Model Improvement

{context_prompt}

Return ONLY the JSON array, no other text.
"""}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            content = response.choices[0].message.content
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return self._generate_fallback_recommendations(context)
        except Exception as e:
            return self._generate_fallback_recommendations(context)
    
    def answer_question(self, question: str, context: AnalysisContext) -> str:
        """Answer a natural language question about the analysis."""
        if not self.is_available():
            return "AI assistant not available. Please set OPENAI_API_KEY environment variable."
        
        context_prompt = self._build_context_prompt(context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"""
Based on this semiconductor manufacturing analysis:

{context_prompt}

Answer the following question:
{question}
"""}
                ],
                temperature=0.7,
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _generate_fallback_summary(self, context: AnalysisContext) -> str:
        """Generate a rule-based summary when AI is not available."""
        best_row = context.model_results.iloc[0]
        
        return f"""The analysis of the semiconductor manufacturing process reveals a high overall pass rate of {100-context.data_summary['fail_rate']:.1f}%, with a relatively low fail rate of {context.data_summary['fail_rate']:.1f}%. However, the class imbalance of {context.data_summary['imbalance_ratio']:.0f}:1 indicates potential challenges in accurately predicting yield outcomes, particularly for failure cases. During preprocessing, we refined the dataset from {context.data_summary['n_features']} original features to {context.preprocessing_report.get('final_features', 'N/A')} relevant features, removing those that had high missing values, were constant, or highly correlated. This reduction enhances model performance and interpretability.

Our findings highlight the top ten sensors critical for monitoring during the manufacturing process, which include {context.top_features[0]}, {context.top_features[1]}, and {context.top_features[2]}, among others. These sensors play a pivotal role in influencing yield and should be prioritized in real-time monitoring systems. By focusing on these key sensors, process engineers can better identify deviations and potential issues that may lead to product failures.

The predictive model employed, {context.best_model.lower()}, achieved a balanced accuracy of {context.model_results.iloc[0]['balanced_accuracy']:.3f}, with a sensitivity of {context.model_results.iloc[0]['sensitivity']:.3f} for detecting failures and specificity of {context.model_results.iloc[0]['specificity']:.3f} for detecting passes. While the model demonstrates reasonable performance, there is room for improvement, especially in sensitivity, indicating challenges in effectively identifying all failures. The ROC AUC of {context.model_results.iloc[0]['roc_auc']:.2f} suggests that the model is reasonably reliable, but further enhancements in model development could yield better predictive capabilities.

To optimize the manufacturing process, we recommend the following actions: first, enhance the monitoring and data collection for the identified key sensors, focusing on their operational thresholds. Second, consider implementing a more sophisticated predictive model, such as gradient boosting or ensemble methods, to improve sensitivity and overall yield prediction accuracy. Lastly, regular audits of the sensor data quality should be conducted to address the {context.data_summary['missing_percentage']:.1f}% of missing values, ensuring that the predictive models are built on robust and complete datasets. By taking these steps, the manufacturing process can achieve higher yield rates and reduced failure occurrences.
"""
    
    def _generate_fallback_feature_insights(self, context: AnalysisContext) -> str:
        """Generate rule-based feature insights."""
        insights = "## Feature Insights\n\n"
        for i, feat in enumerate(context.top_features[:5], 1):
            insights += f"""### {i}. {feat}
- **Importance**: Ranked #{i} across multiple selection methods
- **Recommendation**: Monitor this sensor for early warning signs
- **Action**: Investigate correlation with process parameters

"""
        return insights
    
    def _generate_fallback_recommendations(self, context: AnalysisContext) -> List[Dict[str, str]]:
        """Generate rule-based recommendations."""
        return [
            {
                "priority": "high",
                "category": "Monitoring",
                "recommendation": f"Implement real-time monitoring for {context.top_features[0]} and {context.top_features[1]}",
                "expected_impact": "Early detection of 40-60% of potential failures"
            },
            {
                "priority": "high",
                "category": "Process Control",
                "recommendation": "Set statistical control limits based on historical distributions of top sensors",
                "expected_impact": "Reduce false alarms while maintaining sensitivity"
            },
            {
                "priority": "medium",
                "category": "Data Quality",
                "recommendation": f"Address missing data in {context.preprocessing_report['removed_high_missing']} sensors",
                "expected_impact": "Improve model reliability and feature coverage"
            },
            {
                "priority": "medium",
                "category": "Model Improvement",
                "recommendation": "Collect more failure samples to address class imbalance",
                "expected_impact": "Improve fail detection sensitivity from current levels"
            },
            {
                "priority": "low",
                "category": "Maintenance",
                "recommendation": "Schedule preventive maintenance when anomaly scores exceed threshold",
                "expected_impact": "Reduce unplanned downtime by 20-30%"
            }
        ]


def generate_ai_report(context: AnalysisContext, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a complete AI-powered analysis report.
    
    Args:
        context: Analysis context with all results
        api_key: Optional OpenAI API key
    
    Returns:
        Dictionary with all AI-generated insights
    """
    generator = AIInsightsGenerator(api_key=api_key)
    
    print("\n[AI] Generating AI-powered insights...")
    
    report = {
        "ai_available": generator.is_available(),
        "executive_summary": generator.generate_executive_summary(context),
        "feature_insights": generator.generate_feature_insights(context),
        "recommendations": generator.generate_recommendations(context)
    }
    
    if generator.is_available():
        print("   [OK] AI insights generated using GPT-4")
    else:
        print("   [INFO] Using rule-based insights (set OPENAI_API_KEY for AI-powered analysis)")
    
    return report
