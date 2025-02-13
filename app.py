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

class ChurnAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_importance = None
        self.customer_segments = None
        
    def prepare_features(self, df):
        """
        Create advanced features for churn analysis
        """
        features = df.copy()
        
        # Engagement Metrics
        features['avg_session_duration'] = features['total_session_time'] / features['session_count']
        features['engagement_ratio'] = features['active_days'] / features['customer_tenure_days']
        features['support_interaction_rate'] = features['support_tickets'] / features['customer_tenure_days']
        
        # Product Usage
        features['feature_adoption_rate'] = features['features_used'] / features['total_features_available']
        features['usage_frequency'] = features['total_actions'] / features['active_days']
        
        # Financial Metrics
        features['avg_transaction_value'] = features['total_spend'] / features['transaction_count']
        features['revenue_trend'] = features['recent_month_spend'] / features['avg_monthly_spend']
        
        # Customer Satisfaction
        features['satisfaction_trend'] = features['recent_satisfaction'] / features['avg_satisfaction']
        features['complaint_rate'] = features['total_complaints'] / features['customer_tenure_days']
        
        return features

    def identify_churn_factors(self, df):
        """
        Analyze key factors contributing to churn
        """
        churn_analysis = {
            'why_leave': self._analyze_churn_reasons(df),
            'why_stay': self._analyze_retention_factors(df),
            'risk_indicators': self._identify_risk_indicators(df),
            'customer_segments': self._segment_customers(df)
        }
        return churn_analysis
    
    def _analyze_churn_reasons(self, df):
        """
        Analyze why customers leave
        """
        churned = df[df['churned'] == 1]
        
        churn_reasons = {
            'low_engagement': len(churned[churned['engagement_ratio'] < churned['engagement_ratio'].mean()]),
            'high_complaints': len(churned[churned['complaint_rate'] > churned['complaint_rate'].mean()]),
            'price_sensitivity': len(churned[churned['avg_transaction_value'] < churned['avg_transaction_value'].mean()]),
            'low_satisfaction': len(churned[churned['recent_satisfaction'] < 3]),
            'low_feature_adoption': len(churned[churned['feature_adoption_rate'] < 0.3])
        }
        
        return churn_reasons
    
    def _analyze_retention_factors(self, df):
        """
        Analyze why customers stay
        """
        retained = df[df['churned'] == 0]
        
        retention_factors = {
            'high_engagement': len(retained[retained['engagement_ratio'] > retained['engagement_ratio'].mean()]),
            'feature_utilization': len(retained[retained['feature_adoption_rate'] > 0.7]),
            'satisfaction_level': retained['recent_satisfaction'].mean(),
            'product_stickiness': retained['usage_frequency'].mean(),
            'support_quality': len(retained[retained['support_interaction_rate'] > 0])
        }
        
        return retention_factors

    def _identify_risk_indicators(self, df):
        """
        Identify early warning signs of churn
        """
        risk_indicators = {
            'declining_usage': self._calculate_usage_trend(df),
            'satisfaction_drops': self._analyze_satisfaction_trends(df),
            'support_issues': self._analyze_support_patterns(df),
            'engagement_warning': self._identify_engagement_risks(df)
        }
        
        return risk_indicators

    def _calculate_usage_trend(self, df):
        """
        Calculate trend in usage patterns
        """
        # Calculate average usage trend
        recent_usage = df['usage_frequency'].rolling(window=30).mean()
        previous_usage = df['usage_frequency'].rolling(window=30).mean().shift(30)
        
        usage_trend = {
            'declining_users': len(df[recent_usage < previous_usage]),
            'trend_severity': (recent_usage - previous_usage).mean(),
            'rapid_decline': len(df[recent_usage < previous_usage * 0.7])
        }
        
        return usage_trend

    def _analyze_satisfaction_trends(self, df):
        """
        Analyze trends in customer satisfaction
        """
        satisfaction_analysis = {
            'declining_satisfaction': len(df[df['satisfaction_trend'] < 0.9]),
            'average_satisfaction_change': (df['recent_satisfaction'] - df['avg_satisfaction']).mean(),
            'severe_drops': len(df[df['satisfaction_trend'] < 0.7])
        }
        
        return satisfaction_analysis

    def _analyze_support_patterns(self, df):
        """
        Analyze patterns in support interactions
        """
        support_patterns = {
            'high_ticket_volume': len(df[df['support_interaction_rate'] > df['support_interaction_rate'].mean()]),
            'unresolved_issues': len(df[df['complaint_rate'] > 0.1]),
            'support_satisfaction': df[df['support_tickets'] > 0]['recent_satisfaction'].mean()
        }
        
        return support_patterns

    def _identify_engagement_risks(self, df):
        """
        Identify risks based on engagement patterns
        """
        engagement_risks = {
            'low_engagement': len(df[df['engagement_ratio'] < 0.3]),
            'declining_engagement': len(df[df['engagement_ratio'] < df['engagement_ratio'].rolling(window=30).mean()]),
            'feature_abandonment': len(df[df['feature_adoption_rate'] < 0.4])
        }
        
        return engagement_risks
    
    def generate_recommendations(self, customer_data):
        """
        Generate personalized recommendations for at-risk customers
        """
        recommendations = {
            'high_risk': self._get_high_risk_recommendations(customer_data),
            'medium_risk': self._get_medium_risk_recommendations(customer_data),
            'low_risk': self._get_low_risk_recommendations(customer_data)
        }
        
        return recommendations
    
    def _get_high_risk_recommendations(self, customer_data):
        """
        Recommendations for high-risk customers
        """
        recommendations = []
        
        if customer_data['engagement_ratio'] < 0.3:
            recommendations.append({
                'action': 'Immediate Outreach',
                'method': 'Direct call from account manager',
                'priority': 'High',
                'timeframe': '24 hours',
                'message': 'Schedule urgent review meeting to address concerns'
            })
            
        if customer_data['recent_satisfaction'] < 3:
            recommendations.append({
                'action': 'Satisfaction Recovery',
                'method': 'Executive escalation',
                'priority': 'High',
                'timeframe': '48 hours',
                'message': 'Address specific pain points and provide solution roadmap'
            })
            
        recommendations.append({
            'action': 'Custom Solution Package',
            'method': 'Account review and optimization',
            'priority': 'High',
            'timeframe': '72 hours',
            'message': 'Develop personalized retention package with pricing and feature optimization'
        })
        
        return recommendations

    def _get_medium_risk_recommendations(self, customer_data):
        """
        Recommendations for medium-risk customers
        """
        recommendations = []
        
        if customer_data['feature_adoption_rate'] < 0.5:
            recommendations.append({
                'action': 'Feature Education',
                'method': 'Targeted training session',
                'priority': 'Medium',
                'timeframe': '1 week',
                'message': 'Schedule personalized training on key features'
            })
        
        if customer_data['satisfaction_trend'] < 1:
            recommendations.append({
                'action': 'Satisfaction Check-in',
                'method': 'Customer success call',
                'priority': 'Medium',
                'timeframe': '5 days',
                'message': 'Proactive check-in to gather feedback and address concerns'
            })
        
        if customer_data['support_interaction_rate'] > 0.1:
            recommendations.append({
                'action': 'Support Review',
                'method': 'Support team analysis',
                'priority': 'Medium',
                'timeframe': '1 week',
                'message': 'Review support history and create improvement plan'
            })
            
        return recommendations

    def _get_low_risk_recommendations(self, customer_data):
        """
        Recommendations for low-risk customers
        """
        recommendations = []
        
        if customer_data['feature_adoption_rate'] < 0.7:
            recommendations.append({
                'action': 'Feature Optimization',
                'method': 'Email campaign',
                'priority': 'Low',
                'timeframe': '2 weeks',
                'message': 'Share best practices and feature highlights'
            })
        
        if customer_data['engagement_ratio'] < 0.5:
            recommendations.append({
                'action': 'Engagement Boost',
                'method': 'Newsletter/Content',
                'priority': 'Low',
                'timeframe': '2 weeks',
                'message': 'Share industry insights and success stories'
            })
            
        recommendations.append({
            'action': 'Relationship Building',
            'method': 'Quarterly review',
            'priority': 'Low',
            'timeframe': '4 weeks',
            'message': 'Schedule regular check-in to maintain relationship'
        })
        
        return recommendations
    
    def _segment_customers(self, df):
        """
        Segment customers based on behavior and risk
        """
        features_for_clustering = [
            'engagement_ratio',
            'feature_adoption_rate',
            'avg_transaction_value',
            'satisfaction_trend'
        ]
        
        X = self.scaler.fit_transform(df[features_for_clustering])
        
        # Determine optimal number of clusters
        best_score = -1
        best_k = 2
        
        for k in range(2, 6):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        self.kmeans = KMeans(n_clusters=best_k, random_state=42)
        df['segment'] = self.kmeans.fit_predict(X)
        
        return self._analyze_segments(df)
    
    def _analyze_segments(self, df):
        """
        Analyze characteristics of each customer segment
        """
        segments = {}
        
        for segment in df['segment'].unique():
            segment_data = df[df['segment'] == segment]
            
            segments[f'Segment_{segment}'] = {
                'size': len(segment_data),
                'avg_engagement': segment_data['engagement_ratio'].mean(),
                'avg_satisfaction': segment_data['recent_satisfaction'].mean(),
                'churn_rate': segment_data['churned'].mean(),
                'avg_value': segment_data['avg_transaction_value'].mean()
            }
        
        return segments
    
    def create_action_plan(self, customer_data):
        """
        Create personalized action plan for customer retention
        """
        risk_level = self._calculate_risk_level(customer_data)
        
        action_plan = {
            'risk_level': risk_level,
            'immediate_actions': self._get_immediate_actions(customer_data, risk_level),
            'long_term_strategy': self._get_long_term_strategy(customer_data),
            'success_metrics': self._define_success_metrics(customer_data)
        }
        
        return action_plan

    def _get_immediate_actions(self, customer_data, risk_level):
        """
        Define immediate actions based on risk level and customer data
        """
        actions = []
        
        if risk_level == 'High':
            actions.extend([
                {
                    'action': 'Emergency Account Review',
                    'owner': 'Account Manager',
                    'timeframe': '24 hours',
                    'details': 'Complete account health check and identify critical issues'
                },
                {
                    'action': 'Executive Outreach',
                    'owner': 'Customer Success Manager',
                    'timeframe': '48 hours',
                    'details': 'Schedule executive meeting to address concerns'
                }
            ])
            
        elif risk_level == 'Medium':
            actions.extend([
                {
                    'action': 'Support Review',
                    'owner': 'Support Team Lead',
                    'timeframe': '3 days',
                    'details': 'Review all open support tickets and escalate if needed'
                },
                {
                    'action': 'Usage Analysis',
                    'owner': 'Customer Success',
                    'timeframe': '1 week',
                    'details': 'Analyze usage patterns and identify optimization opportunities'
                }
            ])
            
        else:  # Low risk
            actions.extend([
                {
                    'action': 'Regular Check-in',
                    'owner': 'Account Manager',
                    'timeframe': '2 weeks',
                    'details': 'Schedule routine check-in call'
                }
            ])
            
        return actions

    def _get_long_term_strategy(self, customer_data):
        """
        Define long-term retention strategy
        """
        strategy = {
            'engagement_plan': self._create_engagement_plan(customer_data),
            'feature_adoption': self._create_feature_adoption_plan(customer_data),
            'success_roadmap': self._create_success_roadmap(customer_data)
        }
        
        return strategy

    def _create_engagement_plan(self, customer_data):
        """
        Create engagement improvement plan
        """
        plan = {
            'focus_areas': [],
            'timeline': '3 months',
            'key_milestones': []
        }
        
        if customer_data['engagement_ratio'] < 0.3:
            plan['focus_areas'].append('Increase active usage')
            plan['key_milestones'].append('Achieve 50% increase in weekly active usage')
            
        if customer_data['feature_adoption_rate'] < 0.5:
            plan['focus_areas'].append('Boost feature adoption')
            plan['key_milestones'].append('Increase feature adoption to 70%')
            
        return plan

    def _create_feature_adoption_plan(self, customer_data):
        """
        Create feature adoption improvement plan
        """
        return {
            'target_features': ['Core Feature 1', 'Core Feature 2'],
            'training_needed': customer_data['feature_adoption_rate'] < 0.4,
            'timeline': '2 months',
            'success_criteria': '70% adoption of core features'
        }

    def _create_success_roadmap(self, customer_data):
        """
        Create customer success roadmap
        """
        return {
            'quarterly_goals': [
                'Increase engagement by 30%',
                'Resolve all outstanding support issues',
                'Achieve success criteria for core features'
            ],
            'review_schedule': 'Monthly',
            'success_criteria': 'Improved health score by 20 points'
        }

    def _define_success_metrics(self, customer_data):
        """
        Define metrics to track success of retention efforts
        """
        metrics = {
            'engagement': {
                'current': customer_data['engagement_ratio'],
                'target': max(0.5, customer_data['engagement_ratio'] * 1.3),
                'timeline': '3 months'
            },
            'satisfaction': {
                'current': customer_data['recent_satisfaction'],
                'target': max(4.0, customer_data['recent_satisfaction'] * 1.2),
                'timeline': '2 months'
            },
            'feature_adoption': {
                'current': customer_data['feature_adoption_rate'],
                'target': max(0.7, customer_data['feature_adoption_rate'] * 1.4),
                'timeline': '3 months'
            }
        }
        
        return metrics
    
    def _calculate_risk_level(self, customer_data):
        """
        Calculate customer's risk level
        """
        risk_score = 0
        
        risk_factors = {
            'low_engagement': customer_data['engagement_ratio'] < 0.3,
            'declining_satisfaction': customer_data['satisfaction_trend'] < 0.8,
            'high_complaints': customer_data['complaint_rate'] > 0.1,
            'low_feature_adoption': customer_data['feature_adoption_rate'] < 0.4
        }
        
        risk_score = sum(risk_factors.values())
        
        if risk_score >= 3:
            return 'High'
        elif risk_score >= 1:
            return 'Medium'
        return 'Low'

# Example usage
if __name__ == "__main__":
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
    
    # Get comprehensive analysis
    churn_analysis = analyzer.identify_churn_factors(processed_data)
    
    # Print results
    print("\nWhy Customers Leave:")
    print(churn_analysis['why_leave'])
    
    print("\nWhy Customers Stay:")
    print(churn_analysis['why_stay'])
    
    print("\nRisk Indicators:")
    print(churn_analysis['risk_indicators'])
    
    print("\nCustomer Segments:")
    print(churn_analysis['customer_segments'])
    
    # Generate recommendations for a specific customer
    sample_customer = processed_data.iloc[0]
    recommendations = analyzer.generate_recommendations(sample_customer)
    
    print("\nPersonalized Recommendations:")
    print(recommendations)
    
    # Create action plan
    action_plan = analyzer.create_action_plan(sample_customer)
    
    print("\nAction Plan:")
    print(action_plan)
