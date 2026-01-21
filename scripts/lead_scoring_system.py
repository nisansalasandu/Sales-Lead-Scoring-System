import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Lead Scoring System...")

# Load data
print("\n📊 Loading data...")
df = pd.read_csv('data/sales_leads_realistic.csv')
print(f"   Loaded {len(df)} leads")
print(f"   Conversion rate: {df['converted'].mean()*100:.2f}%")

# Feature engineering
print("\n🔧 Engineering features...")
df_processed = df.copy()

# Encode categorical variables
label_encoders = {}
categorical_cols = ['company_size', 'industry', 'location', 'contact_level', 'referral_source', 'pipeline_stage']

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# Create interaction features
df_processed['engagement_demo'] = df_processed['engagement_score'] * df_processed['demo_requested']
df_processed['budget_revenue_ratio'] = df_processed['budget_indicated_lkr'] / (df_processed['annual_revenue_lkr'] + 1)
df_processed['total_engagement'] = df_processed['website_visits'] + (df_processed['email_opens'] * 2)

# Create contact quality score
contact_scores = {'Employee': 30, 'Manager': 60, 'C-Level': 100}
df_processed['contact_quality'] = df_processed['contact_level'].map(contact_scores)

print("   ✅ Features engineered")

# Prepare data for modeling
feature_cols = [
    'company_size_encoded', 'industry_encoded', 'location_encoded',
    'engagement_score', 'website_visits', 'email_opens', 'demo_requested',
    'days_since_first_contact', 'contact_level_encoded', 'budget_indicated_lkr',
    'annual_revenue_lkr', 'competitor_using', 'referral_source_encoded',
    'engagement_demo', 'budget_revenue_ratio', 'total_engagement', 'contact_quality'
]

X = df_processed[feature_cols]
y = df_processed['converted']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
print("\n⚖️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
print("\n🔄 Balancing classes with SMOTE...")
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"   After SMOTE: {len(X_train_balanced)} samples")

# Train Decision Tree
print("\n🌳 Training Decision Tree model...")
dt_model = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
dt_model.fit(X_train_balanced, y_train_balanced)

y_pred_dt = dt_model.predict(X_test_scaled)
y_pred_proba_dt = dt_model.predict_proba(X_test_scaled)[:, 1]

print("\n📊 Decision Tree Results:")
print(f"   ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_dt):.4f}")
print(f"   Accuracy: {(y_pred_dt == y_test).mean()*100:.2f}%")

# Save models
print("\n💾 Saving models...")
joblib.dump(dt_model, 'models/best_lead_scoring_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')
print("   ✅ Models saved")

# Score all leads
print("\n🎯 Scoring all leads...")

def calculate_composite_score(row):
    features = row[feature_cols].values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    model_prob = dt_model.predict_proba(features_scaled)[0][1] * 100
    
    engagement = row['engagement_score']
    budget_ratio = min(row['budget_revenue_ratio'] * 1000, 100)
    contact_quality = row['contact_quality']
    
    composite_score = (
        model_prob * 0.50 +
        engagement * 0.25 +
        budget_ratio * 0.15 +
        contact_quality * 0.10
    )
    
    return round(composite_score, 2)

df_processed['lead_score'] = df_processed.apply(calculate_composite_score, axis=1)

# Assign priorities based on percentiles
def assign_priority_percentile(scores):
    high_threshold = np.percentile(scores, 80)
    medium_threshold = np.percentile(scores, 50)
    
    priorities = []
    for score in scores:
        if score >= high_threshold:
            priorities.append('HIGH')
        elif score >= medium_threshold:
            priorities.append('MEDIUM')
        else:
            priorities.append('LOW')
    
    return priorities

df_processed['priority'] = assign_priority_percentile(df_processed['lead_score'].values)

# Save scored leads
df_processed.to_csv('data/scored_leads.csv', index=False)
print("   ✅ Scored leads saved to: data/scored_leads.csv")

# Print summary
print("\n" + "="*50)
print("📊 LEAD SCORING SUMMARY")
print("="*50)
print(f"Total Leads: {len(df_processed)}")
print(f"\nPriority Distribution:")
print(df_processed['priority'].value_counts())
print(f"\nPercentages:")
priority_pct = df_processed['priority'].value_counts(normalize=True) * 100
for priority, pct in priority_pct.items():
    print(f"  {priority}: {pct:.1f}%")
print("="*50)
print("\n✅ Lead scoring completed successfully!")
print("   Run 'streamlit run dashboard.py' to view the dashboard")