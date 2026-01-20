# Sales Lead Scoring System 🎯

## 📊 Overview
A data-driven lead prioritization system that uses machine learning to predict conversion likelihood and automatically scores sales leads as High, Medium, or Low priority.

## 🎓 Project Context
Developed as part of a Data Science internship group project at Gamage Recruiters Pvt Ltd, Sri Lanka. This system helps sales teams optimize their workflow by focusing on leads with the highest conversion potential.

## ✨ Features
- 🤖 **Dual Predictive Models**: Logistic Regression & Decision Tree classifiers
- 📊 **Composite Scoring System**: Multi-factor weighted scoring (0-100 scale)
- 🎯 **3-Tier Prioritization**: Automated High/Medium/Low classification
- 📈 **Sales Funnel Tracking**: 5-stage pipeline progression monitoring
- 📑 **Automated Reports**: Daily lead quality reports with action recommendations
- 🇱🇰 **Sri Lankan Context**: Industry and market-specific analysis
- 📊 **Visual Analytics**: Comprehensive dashboards and performance metrics

## 🛠️ Tech Stack
- **Language**: Python 3.13
- **ML Libraries**: Scikit-learn, Imbalanced-learn (SMOTE)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib
- **Development**: Jupyter Notebook

## 📁 Project Structure
```
sales-lead-scoring-system/
├── data/
│   ├── sales_leads_dataset_1000_leads.csv
│   └── cleaned_sales_leads_dataset.csv
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_building.ipynb
│   ├── 03_evaluation_and_scoring.ipynb
│   └── 04_funnel_tracking_and_reports.ipynb
│   
├── models/
│   └── best_lead_scoring_model 1.pkl
│   
├── reports/
│   ├── action_items_2026-01-20.csv
│   ├── comprehensive_lead_report.csv
│   ├── daily_summary_2026-01-20.csv
│   ├── feature_importance.png
│   ├── industry_location_analysis.png
│   └── sales_funnel_visualization.png
│
├── requirements.txt
└── README.md
```
## Prerequisities
- Python 3.8 or higher
- Git
```

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/nisansalasandu/Sales-Lead-Scoring-System.git
cd Sales-Lead-Scoring-System

# Install dependencies
pip install -r requirements.txt

# Run notebooks in sequence
jupyter notebook

# Open and run in order:
# 1. notebooks/01_exploratory_data_analysis.ipynb
# 2. notebooks/02_model_building.ipynb
# 3. notebooks/03_evaluation_and_scoring.ipynb
# 4. notebooks/04_funnel_tracking_and_reports.ipynb
```


## 📊 Dataset Features
- Company size, industry, location
- Engagement metrics (website visits, email opens)
- Contact level and referral source
- Budget indicators
- Conversion status (target variable)

## Dataset Statistics
- Total Leads: 1,000
- Converted: 818 (81.8%)
- Not Converted: 182 (18.2%)
- After SMOTE Balancing: 1,636 samples (50-50 split)


## 🎯 Model Performance

## Comparison Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Decision Tree** ✅ | **78.66%** | **0.79** | **0.79** | **0.79** |
| Logistic Regression | 50.00% | 0.25 | 0.50 | 0.33 |

- **Logistic Regression**: 50% accuracy
- **Decision Tree**: 78.66% accuracy
- **Best Model**: Decision tree

## Selected Model
**Decision Tree Classifier** (max_depth=5, random_state=42)
- Best performance with balanced accuracy
- High precision and recall for both classes
- Saved as `best_lead_scoring_model 1.pkl`
- 
## 🎲 Scoring System

### Composite Score Calculation

Composite Score = 
  Model Prediction (50%) +
  Engagement Score (25%) +
  Budget/Revenue Ratio (15%) +
  Contact Level Score (10%)
````
  
## 📈 Results
- **High Priority Leads**: 583 leads (58.3%)
- **Medium Priority Leads**: 253 leads (25.3%)
- **Low Priority Leads**: 164 leads (16.4%)



## 🔄 Future Enhancements
Potential improvements for scalability:

 Real-time API for instant lead scoring
 Integration with CRM systems (Salesforce, HubSpot)
 A/B testing of priority thresholds
 Advanced models (Random Forest, XGBoost, Neural Networks)
 Automated email alerts for high-priority leads
 Mobile dashboard for on-the-go access
 Historical trend analysis and seasonality detection

## 📄 License
This project is developed for academic purposes as part of an internship program.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📧 Contact
For questions or collaboration: nisansala.ruwanpathirana0@gmail.com


---
**Note**: This is an academic project created for learning purposes.
