## Executive Summary

### Key Findings
The analysis of 1,567 semiconductor manufacturing samples 
identified 20 key sensor signals that influence yield outcomes. 
The dataset exhibits significant class imbalance (14:1 ratio), 
with only 6.6% failure rate.

### Model Performance
The Logistic Regression achieved the best performance with:
- **Balanced Accuracy**: 70.0%
- **Fail Detection Rate**: 61.9%
- **Pass Detection Rate**: 78.2%

### Top Priority Sensors
The following sensors showed the highest correlation with yield outcomes:
1. **sensor_129** - Highest importance across all methods
2. **sensor_124** - Consistent high ranking
3. **sensor_059** - Strong predictive signal

### Recommendations
1. **Implement real-time monitoring** for top 5 sensors
2. **Set control limits** based on historical pass/fail distributions
3. **Investigate root causes** for sensors showing highest importance
4. **Consider ensemble models** for production deployment
5. **Collect more failure samples** to improve model sensitivity

*Note: For more detailed AI-powered insights, set the OPENAI_API_KEY environment variable.*
