# ChurnAnalyzer README

## Overview
**ChurnAnalyzer** is a Python-based framework designed to analyze customer churn and provide actionable insights for improving retention. The framework leverages advanced machine learning techniques and business logic to detect churn patterns, analyze risk factors, and generate personalized recommendations. It is intended for businesses seeking to minimize churn by identifying at-risk customers and implementing targeted retention strategies.

## Key Features
1. **Feature Engineering**: Prepare advanced features for churn analysis, including engagement, financial metrics, product usage, and customer satisfaction.
2. **Churn Factor Analysis**: Identify key reasons for churn, retention factors, and risk indicators based on customer data.
3. **Customer Segmentation**: Use K-Means clustering to segment customers into groups based on behavior and risk.
4. **Personalized Recommendations**: Generate tailored recommendations for high, medium, and low-risk customers to enhance retention efforts.
5. **Action Plan Creation**: Create an action plan for customer retention with both immediate actions and long-term strategies.
6. **Success Metrics**: Define and track success metrics to measure the impact of retention efforts over time.

## Prerequisites
To use the **ChurnAnalyzer**, the following Python libraries are required:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `seaborn`
- `matplotlib`

## Installation
Ensure you have the required libraries by running the following command:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```

## Usage

### 1. Import Required Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
```

### 2. Initialize ChurnAnalyzer
You can create an instance of the `ChurnAnalyzer` class:

```python
analyzer = ChurnAnalyzer()
```

### 3. Prepare Customer Data
You need to prepare your customer data by feeding it into the `prepare_features` method:

```python
processed_data = analyzer.prepare_features(df)
```

Where `df` is your customer dataset in `pandas` DataFrame format. The data should include features such as `total_session_time`, `session_count`, `active_days`, and others that are used in churn analysis.

### 4. Identify Churn Factors
Once your data is processed, you can analyze churn factors, retention, and risk indicators:

```python
churn_analysis = analyzer.identify_churn_factors(processed_data)
```

This will return a dictionary containing:
- `why_leave`: Reasons customers churn
- `why_stay`: Factors contributing to retention
- `risk_indicators`: Early warning signs of churn
- `customer_segments`: Customer segments based on behavior and risk

### 5. Generate Personalized Recommendations
For a given customer, you can generate personalized recommendations based on their behavior and churn risk:

```python
recommendations = analyzer.generate_recommendations(customer_data)
```

Where `customer_data` is a row from the `processed_data` DataFrame (usually a specific customer).

### 6. Create an Action Plan
You can also generate an action plan for retaining a specific customer:

```python
action_plan = analyzer.create_action_plan(customer_data)
```

This will generate an action plan with both immediate actions and long-term retention strategies.

## Example

```python
# Sample data structure
sample_data = pd.DataFrame({
    'customer_id': range(1000),
    'total_session_time': np.random.normal(1000, 200, 1000),
    'session_count': np.random.normal(50, 10, 1000),
    'active_days': np.random.normal(30, 5, 1000),
    'customer_tenure_days': np.random.normal(365, 30, 1000),
    'support_tickets': np.random.poisson(5, 1000),
    'features_used': np.random.normal(7, 2, 1000),
    'total_features_available': np.full(1000, 10),
    'total_actions': np.random.normal(500, 100, 1000),
    'total_spend': np.random.normal(1000, 200, 1000),
    'transaction_count': np.random.normal(20, 5, 1000),
    'recent_month_spend': np.random.normal(100, 20, 1000),
    'avg_monthly_spend': np.random.normal(100, 20, 1000),
    'recent_satisfaction': np.random.normal(4, 0.5, 1000),
    'avg_satisfaction': np.random.normal(4, 0.3, 1000),
    'total_complaints': np.random.poisson(2, 1000),
    'churned': np.random.binomial(1, 0.2, 1000)
})

# Initialize analyzer
analyzer = ChurnAnalyzer()

# Prepare features
processed_data = analyzer.prepare_features(sample_data)

# Get churn analysis
churn_analysis = analyzer.identify_churn_factors(processed_data)

# Print analysis results
print("\nWhy Customers Leave:")
print(churn_analysis['why_leave'])

print("\nWhy Customers Stay:")
print(churn_analysis['why_stay'])

print("\nRisk Indicators:")
print(churn_analysis['risk_indicators'])

print("\nCustomer Segments:")
print(churn_analysis['customer_segments'])

# Generate personalized recommendations for a specific customer
sample_customer = processed_data.iloc[0]
recommendations = analyzer.generate_recommendations(sample_customer)

print("\nPersonalized Recommendations:")
print(recommendations)

# Create action plan for retention
action_plan = analyzer.create_action_plan(sample_customer)

print("\nAction Plan:")
print(action_plan)
```

