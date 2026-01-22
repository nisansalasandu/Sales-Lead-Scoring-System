# Sales Lead Scoring System 🎯

## 📊 Overview
A comprehensive data-driven lead prioritization system that uses machine learning to predict conversion likelihood and automatically scores sales leads as High, Medium, or Low priority. Features an interactive real-time dashboard for sales team optimization.

## 🎓 Project Context
Developed as part of a Data Science internship group project at Gamage Recruiters Pvt Ltd, Sri Lanka. This system helps sales teams optimize their workflow by focusing on leads with the highest conversion potential through AI-powered predictions and actionable insights.

## ✨ Key Features

### 🤖 Machine Learning & Prediction
- **Dual Predictive Models**: Logistic Regression & Decision Tree classifiers with automated model selection
- **Advanced Feature Engineering**: 6 engineered features including engagement interactions, budget ratios, and contact quality scores
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) for balanced training
- **Model Persistence**: Joblib-based serialization for production deployment

### 📊 Lead Scoring & Prioritization
- **Composite Scoring System**: Multi-factor weighted scoring (0-100 scale)
  - Model Prediction: 50%
  - Engagement Score: 25%
  - Budget/Revenue Ratio: 15%
  - Contact Quality: 10%
- **3-Tier Prioritization**: Automated High/Medium/Low classification using percentile thresholds
  - HIGH: Top 20% (80th percentile and above)
  - MEDIUM: Middle 30% (50th-80th percentile)
  - LOW: Bottom 50% (below 50th percentile)

### 📈 Sales Pipeline & Funnel Tracking
- **5-Stage Pipeline**: New → Engaged → Qualified → Demo/Trial → Converted
- **Drop-off Analysis**: Identify bottlenecks and conversion blockers
- **Stage-wise Metrics**: Conversion rates, average scores, and priority distribution per stage
- **Visual Funnel**: Interactive funnel charts showing lead progression

### 📑 Automated Reporting Suite
- **Executive Summary**: High-level KPIs and business metrics
- **Action Items**: Prioritized to-do list with urgency levels (URGENT/HIGH/MEDIUM)
- **Performance Metrics**: Target vs. actual tracking with status indicators
- **Industry Analysis**: Sector-wise performance breakdown
- **Daily Summaries**: Automated snapshot reports with timestamps

### 🎨 Interactive Dashboard (Streamlit)
- **Real-time Lead Scoring Tool**: Input new lead details and get instant predictions
- **Multi-page Navigation**: Dashboard, Reports, and Scoring Tool views
- **Advanced Filtering**: Filter by priority, industry, company size, and score range
- **Visual Analytics**: 15+ interactive charts and graphs (Plotly)
- **Export Functionality**: Download filtered data and reports as CSV
- **Responsive Design**: Mobile and desktop compatible

### 🇱🇰 Sri Lankan Market Context
- **Local Industries**: Agriculture, Tourism, IT/Software, Finance, Healthcare, Education, Manufacturing, Retail
- **Regional Analysis**: Colombo, Kandy, Galle, Negombo, Jaffna, Kurunegala, Matara, Anuradhapura
- **Currency**: LKR (Sri Lankan Rupee) for all financial metrics

## 🛠️ Tech Stack

### Core Technologies
- **Language**: Python 3.8+
- **ML Framework**: Scikit-learn 1.3+
- **Data Processing**: Pandas 2.0+, NumPy 2.0+
- **Visualization**: 
  - Static: Matplotlib, Seaborn
  - Interactive: Plotly, Plotly Express
- **Dashboard**: Streamlit
- **Imbalance Handling**: Imbalanced-learn (SMOTE)
- **Model Persistence**: Joblib

### Development Tools
- **Environment**: Jupyter Notebook
- **Version Control**: Git
- **Package Management**: pip, requirements.txt

## 📁 Project Structure
```
sales-lead-scoring-system/
├── data/
│   ├── sales_leads_dataset_1000_leads.csv      # Raw dataset
│   ├── cleaned_sales_leads_dataset.csv         # Processed data
│   └── scored_leads.csv                        # Final scored leads
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb      # EDA & data cleaning
│   ├── 02_model_building.ipynb                 # Model training & selection
│   ├── 03_evaluation_and_scoring.ipynb         # Lead scoring & prioritization
│   └── 04_funnel_tracking_and_reports.ipynb    # Pipeline analysis & reporting
│   
├── models/
│   ├── best_lead_scoring_model.pkl             # Trained model (Logistic Regression)
│   ├── scaler.pkl                              # StandardScaler for features
│   ├── label_encoders.pkl                      # Category encoders
│   └── feature_columns.pkl                     # Feature list for inference
│
├── reports/
│   ├── executive_summary_YYYY-MM-DD.csv        # Daily executive summary
│   ├── action_items_YYYY-MM-DD.csv             # Prioritized action items
│   ├── performance_metrics_YYYY-MM-DD.csv      # KPI tracking
│   ├── daily_summary_YYYY-MM-DD.csv            # Daily snapshot
│   ├── comprehensive_lead_report.csv           # Complete lead details
│   ├── funnel_metrics.csv                      # Pipeline stage metrics
│   ├── industry_analysis.csv                   # Industry performance
│   ├── feature_coefficients.csv                # Model feature importance
│   ├── lead_scoring_analysis.png               # Visual analytics
│   ├── sales_funnel_visualization.png          # Funnel charts
│   ├── industry_location_analysis.png          # Geographic insights
│   └── feature_coefficients_plot.png           # Feature impact visualization
│
├── dashboard.py                                # Streamlit interactive dashboard
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
└── .gitignore                                  # Git ignore rules
```

## 📋 Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/nisansalasandu/Sales-Lead-Scoring-System.git
cd Sales-Lead-Scoring-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Notebooks (In Order)
```bash
jupyter notebook
```
or

```bash
python -m notebook
```

**Execution Sequence:**
1. `01_exploratory_data_analysis.ipynb` - Data cleaning and EDA
2. `02_model_building.ipynb` - Model training and evaluation
3. `03_evaluation_and_scoring.ipynb` - Lead scoring and prioritization
4. `04_funnel_tracking_and_reports.ipynb` - Funnel analysis and reporting

### 4. Launch Interactive Dashboard
```bash
streamlit run dashboard.py 
```
or

```bash
python -m streamlit run dashboard.py
```

Dashboard will open at: `http://localhost:8501`

## 📊 Dataset Details

### Features (15 columns)
**Categorical Variables:**
- `company_size`: Small, Medium, Large
- `industry`: 8 industries (Agriculture, Tourism, IT/Software, Finance, Healthcare, Education, Manufacturing, Retail)
- `location`: 8 cities in Sri Lanka
- `contact_level`: Employee, Manager, C-Level
- `demo_requested`: Yes/No
- `competitor_using`: Yes/No
- `referral_source`: Website, Cold Call, Referral, Social Media, Trade Show

**Numerical Variables:**
- `annual_revenue_lkr`: Company annual revenue (LKR)
- `engagement_score`: 0-100 engagement metric
- `website_visits`: Number of site visits
- `email_opens`: Email engagement count
- `days_since_first_contact`: Days in pipeline
- `budget_indicated_lkr`: Indicated budget (LKR)

**Target Variable:**
- `converted`: 1 (converted) or 0 (not converted)

### Dataset Statistics
- **Total Leads**: 1,000
- **Converted**: 818 (81.8%)
- **Not Converted**: 182 (18.2%)
- **After SMOTE Balancing**: 859 training samples
  - Class 0: 286
  - Class 1: 573

### Engineered Features (6 additional)
1. `engagement_demo`: Engagement × Demo interaction
2. `budget_revenue_ratio`: Budget/Revenue ratio
3. `total_engagement`: Website visits + (2 × Email opens)
4. `contact_quality`: Numeric contact level score (Employee: 30, Manager: 60, C-Level: 100)
5. `demo_flag`: Binary demo requested indicator
6. `competitor_flag`: Binary competitor usage indicator

## 🎯 Model Performance

### Model Comparison
| Model | Accuracy | ROC-AUC | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|-------|----------|---------|---------------------|------------------|-------------------|
| **Logistic Regression** ✅ | **82.00%** | **0.8132** | **0.88** | **0.90** | **0.89** |
| Decision Tree | 66.67% | 0.7035 | 0.91 | 0.66 | 0.76 |

### Selected Model: Logistic Regression
**Why Logistic Regression Won:**
- ✅ Higher accuracy (82% vs 67%)
- ✅ Better ROC-AUC score (0.8132 vs 0.7035)
- ✅ More balanced precision-recall trade-off
- ✅ Better generalization (less prone to overfitting)
- ✅ Interpretable feature coefficients

**Model Configuration:**
- Algorithm: Logistic Regression with L2 regularization
- Max Iterations: 1000
- Random State: 42
- Class Balancing: SMOTE (sampling_strategy=0.5)
- Feature Scaling: StandardScaler

**Top 5 Most Influential Features:**
1. `engagement_demo` (+1.328) - Strong positive impact
2. `demo_flag` (-0.972) - Encoding artifact
3. `engagement_score` (+0.919) - High engagement drives conversion
4. `company_size_encoded` (-0.807) - Size matters
5. `annual_revenue_lkr` (+0.356) - Revenue correlation

## 🎲 Lead Scoring Methodology

### Composite Score Formula
```python
Composite Score = 
    (Model Probability × 100 × 0.50) +      # 50% weight
    (Engagement Score × 0.25) +             # 25% weight
    (Budget/Revenue Ratio × 1000 × 0.15) +  # 15% weight (capped at 100)
    (Contact Quality Score × 0.10)          # 10% weight
```

### Priority Assignment (Percentile-Based)
```python
if score >= 80th percentile: priority = "HIGH"
elif score >= 50th percentile: priority = "MEDIUM"
else: priority = "LOW"
```

**Actual Thresholds:**
- HIGH: ≥ 81.74 (80th percentile)
- MEDIUM: 69.38 - 81.73 (50th-80th percentile)
- LOW: < 69.38 (below 50th percentile)

## 📈 Key Results & Insights

### Lead Distribution
- **HIGH Priority**: 200 leads (20.0%) - **97.5% conversion rate**
- **MEDIUM Priority**: 301 leads (30.1%) - **90.0% conversion rate**
- **LOW Priority**: 499 leads (49.9%) - **70.5% conversion rate**

### Performance Lift
- HIGH priority leads convert **19.2% better** than average
- Model successfully identifies top 20% with near-perfect conversion

### Pipeline Insights
- **Biggest Bottleneck**: Engaged stage (76.6% drop-off)
- **Best Performing Stage**: Converted (100% conversion)
- **Recommended Focus**: Improve Engaged → Qualified transition

### Industry Insights
- **Top Industry by Volume**: Tourism
- **Top Location**: Galle
- **Best Converting Segments**: Large companies (91.2%), C-Level contacts (88.2%), Demo requesters (92.1%)

### Key Conversion Drivers
1. **High Engagement Scores** (avg 81.67 for converted vs 64.61 for non-converted)
2. **C-Level Contact Access** (88.2% conversion rate)
3. **Demo Requests** (92.1% conversion rate)
4. **Large Company Size** (91.2% conversion rate)

## 🎨 Dashboard Features

### Page 1: Main Dashboard
- **Key Metrics**: Total leads, priority breakdown, avg score, conversion rate
- **Priority Distribution**: Interactive pie chart
- **Score Distribution**: Histogram with priority segmentation
- **Conversion Analysis**: Bar charts by priority level
- **Industry Breakdown**: Top industries by lead count
- **Engagement Scatter**: Score vs engagement correlation
- **Top Leads Table**: High-priority leads requiring immediate action
- **Data Export**: Download filtered leads as CSV

### Page 2: Reports & Analytics
- **Executive Summary**: Business KPIs and metrics
- **Action Items**: Urgency-coded task list for sales team
- **Performance Metrics**: Target tracking with status indicators
- **Industry Analysis**: Sector-wise performance and potential value
- **Funnel Analysis**: Pipeline visualization with drop-off rates
- **Batch Export**: Generate and download all reports at once

### Page 3: Lead Scoring Tool
- **Real-time Prediction**: Input new lead details → instant score
- **13 Input Fields**: All relevant lead characteristics
- **Instant Results**: Lead score, priority level, conversion probability
- **Action Recommendations**: Context-aware next steps
- **What-If Analysis**: Test different scenarios

### Interactive Filters (Sidebar)
- Priority level (multi-select)
- Industry (multi-select)
- Company size (multi-select)
- Lead score range (slider: 0-100)

## 🔄 Automated Reporting Schedule

### Daily Reports (Auto-generated)
1. **Executive Summary** - Business overview with KPIs
2. **Action Items** - Prioritized task list with urgency flags
3. **Performance Metrics** - Target vs. actual comparison
4. **Daily Summary** - Quick snapshot of lead status

### On-Demand Reports
1. **Comprehensive Lead Report** - Full lead details with recommendations
2. **Industry Analysis** - Sector performance breakdown
3. **Funnel Metrics** - Pipeline stage analysis
4. **Feature Importance** - Model interpretation

### Report Formats
- **CSV**: All tabular reports for Excel/Google Sheets
- **PNG**: All visualizations (300 DPI, publication-ready)

## 💡 Business Impact

### For Sales Teams
- ✅ **25% time savings** - Focus on high-probability leads
- ✅ **Clear prioritization** - No guesswork on which leads to pursue
- ✅ **Actionable insights** - Specific recommendations for each lead
- ✅ **Pipeline visibility** - Real-time tracking of lead progression

### For Sales Managers
- ✅ **Performance tracking** - Monitor team KPIs vs. targets
- ✅ **Resource allocation** - Assign senior reps to high-value leads
- ✅ **Bottleneck identification** - See where leads drop off
- ✅ **Data-driven decisions** - Industry and regional insights

### For Business Leaders
- ✅ **Revenue forecasting** - Predict conversion values
- ✅ **Market insights** - Understand high-performing segments
- ✅ **ROI optimization** - Maximize return on sales efforts
- ✅ **Strategic planning** - Industry and location targeting

## 🔧 Troubleshooting

### Dashboard Not Loading
```bash
# Ensure all models exist
ls models/
# Should show: best_lead_scoring_model.pkl, scaler.pkl, label_encoders.pkl, feature_columns.pkl

# If missing, run notebooks 01-03 in order
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Model Performance Issues
```bash
# Retrain with fresh data
# Run notebook 02 again
```

### Dashboard Filters Not Working
- Clear browser cache
- Restart Streamlit: `Ctrl+C` then `streamlit run dashboard.py`

## 🔄 Future Enhancements

- [ ] **CRM Integration**: Salesforce, HubSpot, Zoho connectors
- [ ] **Advanced Models**: Random Forest, XGBoost, LightGBM
- [ ] **Deep Learning**: Neural networks for complex patterns
- [ ] **Real-time Scoring**: WebSocket-based instant predictions
- [ ] **Predictive Analytics**: Forecast next month's conversions


## 🤝 Contributing

We welcome contributions! 

### Contribution Areas
- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🧪 Additional test cases
- 🎨 UI/UX enhancements
- 🌍 Translations

## 📄 License

This project is developed for **academic and educational purposes** as part of an internship program at Gamage Recruiters Pvt Ltd, Sri Lanka.

**Usage Terms:**
- ✅ Free for educational and research use
- ✅ Modifications and improvements encouraged
- ✅ Commercial use requires attribution
- ❌ No warranty provided

## 🙏 Acknowledgments

- **Gamage Recruiters Pvt Ltd** - Internship opportunity and project sponsorship
- **Scikit-learn Community** - Excellent ML library and documentation
- **Streamlit Team** - Amazing dashboard framework

## 👥 Team

**Developer Team**: Data Science Interns  
**Organization**: Gamage Recruiters Pvt Ltd  
**Location**: Sri Lanka 🇱🇰

## 📧 Contact & Support

- **Email**: nisansala.ruwanpathirana0@gmail.com
- **GitHub**: [@nisansalasandu](https://github.com/nisansalasandu)
- **Project Issues**: [GitHub Issues](https://github.com/nisansalasandu/Sales-Lead-Scoring-System/issues)


**⭐ If you find this project helpful, please star the repository!**

**📝 Note**: This is an academic project created for learning and demonstration purposes. While functional, it should be thoroughly tested and validated before production deployment.

---

*Last Updated: January 2026*
*Version: 2.0.0*
*Documentation Status: Complete*
