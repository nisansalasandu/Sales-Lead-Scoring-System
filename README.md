# Sales Lead Scoring System 🎯

## 📊 Overview
A data-driven lead prioritization system that uses machine learning to predict conversion likelihood and automatically scores sales leads as High, Medium, or Low priority.

## 🎓 Project Context
Developed as part of a Data Science internship group project at Gamage Recruiters Pvt Ltd, Sri Lanka. This system helps sales teams optimize their workflow by focusing on leads with the highest conversion potential.

## ✨ Features
- 🤖 **Predictive Models**: Logistic Regression & Decision Tree classifiers
- 📈 **Automated Scoring**: Real-time lead prioritization (High/Medium/Low)
- 📊 **Interactive Dashboard**: Visual analytics and pipeline tracking
- 📑 **Lead Quality Reports**: Automated insights and conversion metrics
- 🇱🇰 **Sri Lankan Context**: Industry and market-specific analysis

## 🛠️ Tech Stack
- **Language**: Python 3.8+
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit / Dash
- **Data**: CSV with 1500+ historical lead records

## 📁 Project Structure
```
sales-lead-scoring-system/
├── data/
│   ├── sales_leads_dataset.csv
│   └── README.md
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_building.ipynb
│   └── 03_evaluation_and_scoring.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── lead_scoring.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── reports/
│   └── lead_prioritization_matrix.pdf
├── requirements.txt
└── README.md
```

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/sales-lead-scoring-system.git
cd sales-lead-scoring-system

# Install dependencies
pip install -r requirements.txt

# Run analysis
python src/model_training.py

# Launch dashboard
streamlit run dashboard/app.py
```

## 📊 Dataset Features
- Company size, industry, location
- Engagement metrics (website visits, email opens)
- Contact level and referral source
- Budget indicators
- Conversion status (target variable)

## 🎯 Model Performance
- **Logistic Regression**: 82% accuracy
- **Decision Tree**: 79% accuracy
- **Best Model**: Logistic Regression with feature engineering

## 👥 Team Members
- Member 1 - Data Analysis & Preprocessing
- Member 2 - Model Development
- Member 3 - Dashboard Development
- Member 4 - Documentation & Testing
- Member 5 - Project Coordination

## 📈 Results
- **High Priority Leads**: 70+ score (Top 25% - 45% conversion rate)
- **Medium Priority Leads**: 40-70 score (Middle 50% - 28% conversion rate)
- **Low Priority Leads**: <40 score (Bottom 25% - 12% conversion rate)

## 📄 License
MIT License

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📧 Contact
For questions or collaboration: your.email@university.edu

---
**Note**: This is an academic project created for learning purposes.
