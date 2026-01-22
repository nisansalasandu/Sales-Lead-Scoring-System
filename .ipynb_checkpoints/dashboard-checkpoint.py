import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Sales Lead Scoring Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #ff7f0e;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 Sales Lead Scoring Dashboard")
st.markdown("### AI-Powered Lead Prioritization System")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load the scored leads data"""
    try:
        df = pd.read_csv('data/scored_leads.csv')
        
        # FIX: Ensure demo_requested is string for consistency
        if 'demo_requested' in df.columns:
            if df['demo_requested'].dtype in ['int64', 'float64']:
                df['demo_requested'] = df['demo_requested'].map({1: 'Yes', 0: 'No'})
        
        # FIX: Ensure demo_flag exists
        if 'demo_flag' not in df.columns and 'demo_requested' in df.columns:
            df['demo_flag'] = df['demo_requested'].map({'Yes': 1, 'No': 0})
        
        # FIX: Ensure competitor_flag exists
        if 'competitor_flag' not in df.columns and 'competitor_using' in df.columns:
            df['competitor_flag'] = df['competitor_using'].map({'Yes': 1, 'No': 0})
        
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Please run notebooks 01-03 first.")
        return None

# Load models
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        # FIX: Correct filename without space and "1"
        model = joblib.load('models/best_lead_scoring_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/label_encoders.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return model, scaler, encoders, feature_cols
    except FileNotFoundError as e:
        st.warning(f"⚠️ Models not found: {e}. Some features will be limited.")
        return None, None, None, None

def generate_executive_summary(df, report_date):
    """Generate executive summary report"""
    summary = {
        'Report Date': [report_date],
        'Total Leads': [len(df)],
        'High Priority Leads': [len(df[df['priority'] == 'HIGH'])],
        'Medium Priority Leads': [len(df[df['priority'] == 'MEDIUM'])],
        'Low Priority Leads': [len(df[df['priority'] == 'LOW'])],
        'Average Lead Score': [df['lead_score'].mean()],
        'Overall Conversion Rate (%)': [df['converted'].mean() * 100],
        'High Priority Conversion (%)': [df[df['priority'] == 'HIGH']['converted'].mean() * 100],
        'Total Converted Leads': [df['converted'].sum()],
        'Conversion Value': [df['converted'].sum() * 50000],
        'Top Industry': [df['industry'].mode()[0] if len(df) > 0 else 'N/A'],
        'Top Location': [df['location'].mode()[0] if len(df) > 0 else 'N/A'],
        'Avg Engagement Score': [df['engagement_score'].mean()]
    }
    return pd.DataFrame(summary)

def generate_action_items(df, report_date):
    """Generate action items for sales team"""
    high_priority = df[df['priority'] == 'HIGH'].nlargest(20, 'lead_score')
    
    action_items = []
    for idx, row in high_priority.iterrows():
        # FIX: Handle both string and numeric demo_requested
        demo_req = row['demo_requested']
        if isinstance(demo_req, (int, float)):
            demo_req = 'Yes' if demo_req == 1 else 'No'
        
        if demo_req == 'Yes':
            action = "Schedule demo ASAP - Lead requested demonstration"
            urgency = "URGENT"
        elif row['contact_level'] == 'C-Level':
            action = "Executive outreach - C-Level contact identified"
            urgency = "HIGH"
        elif row['engagement_score'] > 80:
            action = "Hot lead - High engagement, contact within 24h"
            urgency = "HIGH"
        else:
            action = "Standard follow-up within 24-48 hours"
            urgency = "MEDIUM"
        
        action_items.append({
            'Lead ID': row['lead_id'],
            'Company Size': row['company_size'],
            'Industry': row['industry'],
            'Location': row['location'],
            'Lead Score': row['lead_score'],
            'Priority': row['priority'],
            'Contact Level': row['contact_level'],
            'Urgency': urgency,
            'Recommended Action': action,
            'Assigned To': 'Senior Sales Rep' if urgency == 'URGENT' else 'Sales Team',
            'Pipeline Stage': row['pipeline_stage'],
            'Engagement Score': row['engagement_score']
        })
    
    return pd.DataFrame(action_items)

def generate_performance_metrics(df, report_date):
    """Generate detailed performance metrics"""
    metrics = {
        'Metric': [],
        'Value': [],
        'Target': [],
        'Status': []
    }
    
    # FIX: Handle both string and numeric demo_requested
    demo_count = df['demo_requested'].apply(lambda x: 1 if x in ['Yes', 1] else 0).sum()
    
    metric_data = [
        ('High Priority Lead %', (len(df[df['priority'] == 'HIGH']) / len(df)) * 100, 20, '>='),
        ('Avg Lead Score', df['lead_score'].mean(), 50, '>='),
        ('Overall Conversion Rate %', df['converted'].mean() * 100, 5, '>='),
        ('High Priority Conversion %', df[df['priority'] == 'HIGH']['converted'].mean() * 100, 15, '>='),
        ('Avg Engagement Score', df['engagement_score'].mean(), 50, '>='),
        ('C-Level Contact %', (df['contact_level'].eq('C-Level').sum() / len(df)) * 100, 10, '>='),
        ('Demo Request Rate %', (demo_count / len(df)) * 100, 25, '>=')
    ]
    
    for metric_name, value, target, comparison in metric_data:
        if comparison == '>=':
            status = '✅ On Track' if value >= target else '⚠️ Below Target'
        else:
            status = '✅ On Track' if value <= target else '⚠️ Above Target'
        
        metrics['Metric'].append(metric_name)
        metrics['Value'].append(round(value, 2))
        metrics['Target'].append(target)
        metrics['Status'].append(status)
    
    return pd.DataFrame(metrics)

def generate_industry_analysis(df, report_date):
    """Generate industry-wise analysis"""
    industry_metrics = []
    
    for industry in df['industry'].unique():
        industry_df = df[df['industry'] == industry]
        
        metrics = {
            'Industry': industry,
            'Total Leads': len(industry_df),
            'High Priority': len(industry_df[industry_df['priority'] == 'HIGH']),
            'Avg Lead Score': industry_df['lead_score'].mean(),
            'Conversion Rate (%)': industry_df['converted'].mean() * 100,
            'Avg Engagement': industry_df['engagement_score'].mean(),
            'Converted Leads': industry_df['converted'].sum(),
            'Potential Value (LKR)': industry_df['converted'].sum() * 50000
        }
        industry_metrics.append(metrics)
    
    return pd.DataFrame(industry_metrics).sort_values('Total Leads', ascending=False)

def analyze_sales_funnel(df):
    """Analyze and track leads through the sales funnel"""
    # FIX: Use the actual pipeline_stage values from the data
    if 'pipeline_stage' in df.columns:
        # Get unique stages and count leads in each
        stage_counts = df['pipeline_stage'].value_counts()
        
        # Define a logical order if stages match expected names
        stage_order = ['New', 'Engaged', 'Qualified', 'Demo/Trial', 'Converted']
        existing_stages = [s for s in stage_order if s in stage_counts.index]
        
        funnel_metrics = []
        for i, stage in enumerate(existing_stages):
            stage_leads = df[df['pipeline_stage'] == stage]
            
            metrics = {
                'stage': stage,
                'stage_number': i + 1,
                'total_leads': len(stage_leads),
                'converted_leads': stage_leads['converted'].sum(),
                'conversion_rate': (stage_leads['converted'].mean() * 100) if len(stage_leads) > 0 else 0,
                'avg_lead_score': stage_leads['lead_score'].mean() if len(stage_leads) > 0 else 0,
                'high_priority': len(stage_leads[stage_leads['priority'] == 'HIGH']),
                'medium_priority': len(stage_leads[stage_leads['priority'] == 'MEDIUM']),
                'low_priority': len(stage_leads[stage_leads['priority'] == 'LOW']),
                'avg_engagement': stage_leads['engagement_score'].mean() if len(stage_leads) > 0 else 0
            }
            funnel_metrics.append(metrics)
        
        funnel_df = pd.DataFrame(funnel_metrics)
        
        # Calculate drop-off rates
        funnel_df['drop_off_rate'] = 0.0
        for i in range(1, len(funnel_df)):
            prev_leads = funnel_df.loc[i-1, 'total_leads']
            curr_leads = funnel_df.loc[i, 'total_leads']
            if prev_leads > 0:
                funnel_df.loc[i, 'drop_off_rate'] = ((prev_leads - curr_leads) / prev_leads) * 100
        
        return funnel_df
    else:
        # Return empty dataframe with correct structure
        return pd.DataFrame(columns=['stage', 'stage_number', 'total_leads', 'converted_leads', 
                                    'conversion_rate', 'avg_lead_score', 'high_priority', 
                                    'medium_priority', 'low_priority', 'avg_engagement', 'drop_off_rate'])

# Load data and models
df = load_data()
model, scaler, encoders, feature_cols = load_models()

if df is not None:
    
    # Sidebar Navigation
    st.sidebar.header("📑 Navigation")
    page = st.sidebar.radio(
        "Select View:",
        ["📊 Dashboard", "📋 Reports & Analytics", "🤖 Lead Scoring Tool"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")
    
    # Filters
    priority_filter = st.sidebar.multiselect(
        "Select Priority Level:",
        options=['HIGH', 'MEDIUM', 'LOW'],
        default=['HIGH', 'MEDIUM', 'LOW']
    )
    
    industry_filter = st.sidebar.multiselect(
        "Select Industry:",
        options=df['industry'].unique().tolist(),
        default=df['industry'].unique().tolist()
    )
    
    size_filter = st.sidebar.multiselect(
        "Select Company Size:",
        options=df['company_size'].unique().tolist(),
        default=df['company_size'].unique().tolist()
    )
    
    score_range = st.sidebar.slider(
        "Lead Score Range:",
        min_value=0,
        max_value=100,
        value=(0, 100)
    )
    
    # Apply filters
    filtered_df = df[
        (df['priority'].isin(priority_filter)) &
        (df['industry'].isin(industry_filter)) &
        (df['company_size'].isin(size_filter)) &
        (df['lead_score'] >= score_range[0]) &
        (df['lead_score'] <= score_range[1])
    ]
    
    # PAGE 1: DASHBOARD
    if page == "📊 Dashboard":
        
        # KEY METRICS ROW
        st.header("📊 Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total Leads",
                value=f"{len(filtered_df):,}",
                delta=f"{len(filtered_df) - len(df)} from total"
            )
        
        with col2:
            high_count = len(filtered_df[filtered_df['priority'] == 'HIGH'])
            st.metric(
                label="HIGH Priority",
                value=high_count,
                delta=f"{high_count/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%"
            )
        
        with col3:
            medium_count = len(filtered_df[filtered_df['priority'] == 'MEDIUM'])
            st.metric(
                label="MEDIUM Priority",
                value=medium_count,
                delta=f"{medium_count/len(filtered_df)*100:.1f}%" if len(filtered_df) > 0 else "0%"
            )
        
        with col4:
            avg_score = filtered_df['lead_score'].mean()
            st.metric(
                label="Avg Lead Score",
                value=f"{avg_score:.1f}/100",
                delta=f"{avg_score - df['lead_score'].mean():.1f} from overall"
            )
        
        with col5:
            conv_rate = filtered_df['converted'].mean() * 100
            st.metric(
                label="Conversion Rate",
                value=f"{conv_rate:.1f}%",
                delta=f"{conv_rate - df['converted'].mean()*100:.1f}%"
            )
        
        st.markdown("---")
        
        # VISUALIZATIONS
        st.header("📈 Lead Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            priority_counts = filtered_df['priority'].value_counts()
            fig_pie = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Lead Priority Distribution",
                color=priority_counts.index,
                color_discrete_map={'HIGH':'#00CC96', 'MEDIUM':'#FFA15A', 'LOW':'#EF553B'},
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_hist = px.histogram(
                filtered_df,
                x='lead_score',
                nbins=30,
                title="Lead Score Distribution",
                color='priority',
                color_discrete_map={'HIGH':'#00CC96', 'MEDIUM':'#FFA15A', 'LOW':'#EF553B'},
                labels={'lead_score': 'Lead Score', 'count': 'Number of Leads'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # HIGH PRIORITY LEADS TABLE
        st.markdown("---")
        st.header("🎯 Top High Priority Leads - Action Required")
        
        high_priority_leads = filtered_df[filtered_df['priority'] == 'HIGH'].nlargest(10, 'lead_score')
        
        if len(high_priority_leads) > 0:
            display_cols = ['lead_id', 'industry', 'company_size', 'lead_score', 
                           'engagement_score', 'contact_level', 'pipeline_stage']
            
            st.dataframe(high_priority_leads[display_cols], use_container_width=True)
            
            csv = high_priority_leads.to_csv(index=False)
            st.download_button(
                label="📥 Download High Priority Leads (CSV)",
                data=csv,
                file_name=f"high_priority_leads_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No high priority leads match the current filters.")
    
    # PAGE 2: REPORTS & ANALYTICS
    elif page == "📋 Reports & Analytics":
        st.header("📋 Automated Reports & Analytics")
        
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Summary", 
            "🎯 Action Items", 
            "📈 Performance Metrics",
            "🏭 Industry Analysis",
            "🔄 Funnel Analysis"
        ])
        
        with tab1:
            st.subheader("📊 Executive Summary Report")
            exec_summary = generate_executive_summary(filtered_df, report_date)
            st.dataframe(exec_summary.T, use_container_width=True)
            
            csv_exec = exec_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Executive Summary",
                data=csv_exec,
                file_name=f"executive_summary_{report_date}.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.subheader("🎯 Action Items for Sales Team")
            action_items = generate_action_items(filtered_df, report_date)
            
            if len(action_items) > 0:
                st.dataframe(action_items, use_container_width=True)
                csv_actions = action_items.to_csv(index=False)
                st.download_button(
                    label="📥 Download Action Items",
                    data=csv_actions,
                    file_name=f"action_items_{report_date}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No high priority leads requiring action.")
        
        with tab3:
            st.subheader("📈 Performance Metrics")
            perf_metrics = generate_performance_metrics(filtered_df, report_date)
            st.dataframe(perf_metrics, use_container_width=True)
            
            csv_perf = perf_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Performance Metrics",
                data=csv_perf,
                file_name=f"performance_metrics_{report_date}.csv",
                mime="text/csv"
            )
        
        with tab4:
            st.subheader("🏭 Industry Analysis")
            industry_analysis = generate_industry_analysis(filtered_df, report_date)
            st.dataframe(industry_analysis, use_container_width=True)
            
            csv_industry = industry_analysis.to_csv(index=False)
            st.download_button(
                label="📥 Download Industry Analysis",
                data=csv_industry,
                file_name=f"industry_analysis_{report_date}.csv",
                mime="text/csv"
            )
        
        with tab5:
            st.subheader("🔄 Sales Funnel Analysis")
            funnel_metrics = analyze_sales_funnel(filtered_df)
            
            if len(funnel_metrics) > 0:
                st.dataframe(funnel_metrics, use_container_width=True)
                
                csv_funnel = funnel_metrics.to_csv(index=False)
                st.download_button(
                    label="📥 Download Funnel Metrics",
                    data=csv_funnel,
                    file_name=f"funnel_metrics_{report_date}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Pipeline stage data not available")
    
    # PAGE 3: LEAD SCORING TOOL
    elif page == "🤖 Lead Scoring Tool":
        st.header("🤖 Real-Time Lead Scoring Tool")
        
        if model is not None and scaler is not None and encoders is not None and feature_cols is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_company_size = st.selectbox("Company Size", ['Small', 'Medium', 'Large'])
                new_industry = st.selectbox("Industry", sorted(df['industry'].unique()))
                new_location = st.selectbox("Location", sorted(df['location'].unique()))
                new_referral = st.selectbox("Referral Source", sorted(df['referral_source'].unique()))
            
            with col2:
                new_engagement = st.slider("Engagement Score", 0, 100, 50)
                new_website_visits = st.number_input("Website Visits", 0, 100, 15)
                new_email_opens = st.number_input("Email Opens", 0, 50, 5)
                new_days_contact = st.number_input("Days Since First Contact", 1, 180, 30)
            
            with col3:
                new_demo = st.selectbox("Demo Requested", ['No', 'Yes'])
                new_contact_level = st.selectbox("Contact Level", ['Employee', 'Manager', 'C-Level'])
                new_competitor = st.selectbox("Using Competitor", ['No', 'Yes'])
                new_revenue = st.number_input("Annual Revenue (LKR)", 1000000, 1000000000, 50000000)
                new_budget = st.number_input("Budget Indicated (LKR)", 10000, 100000000, 1000000)
            
            if st.button("🎯 Calculate Lead Score", type="primary"):
                try:
                    # Create new lead dataframe
                    new_lead = pd.DataFrame({
                        'company_size': [new_company_size],
                        'industry': [new_industry],
                        'location': [new_location],
                        'annual_revenue_lkr': [new_revenue],
                        'engagement_score': [new_engagement],
                        'website_visits': [new_website_visits],
                        'email_opens': [new_email_opens],
                        'demo_requested': [new_demo],
                        'days_since_first_contact': [new_days_contact],
                        'contact_level': [new_contact_level],
                        'budget_indicated_lkr': [new_budget],
                        'competitor_using': [new_competitor],
                        'referral_source': [new_referral]
                    })
                    
                    # Apply feature engineering
                    new_lead['engagement_demo'] = new_lead['engagement_score'] * new_lead['demo_requested'].map({'Yes': 1, 'No': 0})
                    new_lead['budget_revenue_ratio'] = new_lead['budget_indicated_lkr'] / (new_lead['annual_revenue_lkr'] + 1)
                    new_lead['total_engagement'] = new_lead['website_visits'] + (new_lead['email_opens'] * 2)
                    
                    contact_scores = {'Employee': 30, 'Manager': 60, 'C-Level': 100}
                    new_lead['contact_quality'] = new_lead['contact_level'].map(contact_scores)
                    new_lead['demo_flag'] = new_lead['demo_requested'].map({'Yes': 1, 'No': 0})
                    new_lead['competitor_flag'] = new_lead['competitor_using'].map({'Yes': 1, 'No': 0})
                    
                    # Encode categorical variables
                    for col in ['company_size', 'industry', 'location', 'contact_level', 'referral_source']:
                        le = encoders[col]
                        new_lead[col + '_encoded'] = le.transform(new_lead[col])
                    
                    # Select and scale features
                    X_new = new_lead[feature_cols]
                    X_new_scaled = scaler.transform(X_new)
                    
                    # Predict
                    prediction_proba = model.predict_proba(X_new_scaled)[0][1]
                    
                    # Calculate composite score
                    composite_score = (
                        prediction_proba * 100 * 0.50 +
                        new_engagement * 0.25 +
                        min(new_lead['budget_revenue_ratio'].values[0] * 1000, 100) * 0.15 +
                        new_lead['contact_quality'].values[0] * 0.10
                    )
                    
                    # Determine priority
                    if composite_score >= 70:
                        priority = "HIGH"
                        color = "🟢"
                    elif composite_score >= 45:
                        priority = "MEDIUM"
                        color = "🟡"
                    else:
                        priority = "LOW"
                        color = "🔴"
                    
                    # Display results
                    st.success("✅ Lead Score Calculated!")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    with result_col1:
                        st.metric("Lead Score", f"{composite_score:.1f}/100")
                    with result_col2:
                        st.metric("Priority Level", f"{color} {priority}")
                    with result_col3:
                        st.metric("Conversion Probability", f"{prediction_proba*100:.1f}%")
                    
                    # Recommendation
                    if priority == "HIGH":
                        st.info("💡 **Recommendation:** Contact within 24 hours. Assign to senior sales rep.")
                    elif priority == "MEDIUM":
                        st.info("💡 **Recommendation:** Contact within 72 hours. Regular follow-up required.")
                    else:
                        st.info("💡 **Recommendation:** Add to nurture campaign. Revisit in 30 days.")
                    
                except Exception as e:
                    st.error(f"Error calculating score: {str(e)}")
                    st.exception(e)
        else:
            st.warning("⚠️ Models not loaded. Please run notebooks 01-03 first.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

else:
    st.error("Unable to load data. Please run notebooks 01-03 first.")