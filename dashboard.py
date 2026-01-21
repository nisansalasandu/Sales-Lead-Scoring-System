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

# Custom CSS for better styling
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

# Title and header
st.title("🎯 Sales Lead Scoring Dashboard")
st.markdown("### AI-Powered Lead Prioritization System")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load the scored leads data"""
    try:
        df = pd.read_csv('data/scored_leads.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Please run the lead scoring system first.")
        st.info("Run: `python scripts/lead_scoring_system.py`")
        return None

# Load models
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = joblib.load('models/best_lead_scoring_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoders = joblib.load('models/label_encoders.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return model, scaler, encoders, feature_cols
    except FileNotFoundError:
        st.warning("⚠️ Models not found. Some features will be limited.")
        return None, None, None, None

# Load data and models
df = load_data()
model, scaler, encoders, feature_cols = load_models()

if df is not None:
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Priority filter
    priority_filter = st.sidebar.multiselect(
        "Select Priority Level:",
        options=['HIGH', 'MEDIUM', 'LOW'],
        default=['HIGH', 'MEDIUM', 'LOW']
    )
    
    # Industry filter
    industry_filter = st.sidebar.multiselect(
        "Select Industry:",
        options=df['industry'].unique().tolist(),
        default=df['industry'].unique().tolist()
    )
    
    # Company size filter
    size_filter = st.sidebar.multiselect(
        "Select Company Size:",
        options=df['company_size'].unique().tolist(),
        default=df['company_size'].unique().tolist()
    )
    
    # Score range filter
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
    
    # Main dashboard content
    
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
    
    # VISUALIZATIONS ROW 1
    st.header("📈 Lead Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority Distribution Pie Chart
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
        # Score Distribution Histogram
        fig_hist = px.histogram(
            filtered_df,
            x='lead_score',
            nbins=30,
            title="Lead Score Distribution",
            color='priority',
            color_discrete_map={'HIGH':'#00CC96', 'MEDIUM':'#FFA15A', 'LOW':'#EF553B'},
            labels={'lead_score': 'Lead Score', 'count': 'Number of Leads'}
        )
        fig_hist.update_layout(showlegend=True)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # VISUALIZATIONS ROW 2
    col3, col4 = st.columns(2)
    
    with col3:
        # Conversion Rate by Priority
        conv_by_priority = filtered_df.groupby('priority')['converted'].agg(['mean', 'count']).reset_index()
        conv_by_priority['mean'] = conv_by_priority['mean'] * 100
        conv_by_priority = conv_by_priority.sort_values('mean', ascending=False)
        
        fig_conv = px.bar(
            conv_by_priority,
            x='priority',
            y='mean',
            title='Conversion Rate by Priority Level',
            text='mean',
            color='priority',
            color_discrete_map={'HIGH':'#00CC96', 'MEDIUM':'#FFA15A', 'LOW':'#EF553B'},
            labels={'mean': 'Conversion Rate (%)', 'priority': 'Priority Level'}
        )
        fig_conv.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_conv, use_container_width=True)
    
    with col4:
        # Industry Distribution
        industry_counts = filtered_df['industry'].value_counts().head(8)
        fig_industry = px.bar(
            x=industry_counts.values,
            y=industry_counts.index,
            orientation='h',
            title='Top Industries',
            labels={'x': 'Number of Leads', 'y': 'Industry'},
            color=industry_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_industry, use_container_width=True)
    
    st.markdown("---")
    
    # SALES FUNNEL
    st.header("🔄 Sales Pipeline Funnel")
    
    if 'pipeline_stage' in filtered_df.columns:
        funnel_data = filtered_df.groupby('pipeline_stage').size().reset_index(name='count')
        stage_order = ['New', 'Contacted', 'Qualified', 'Proposal', 'Negotiation']
        
        # Only include stages that exist in the data
        existing_stages = [s for s in stage_order if s in funnel_data['pipeline_stage'].values]
        funnel_data['pipeline_stage'] = pd.Categorical(
            funnel_data['pipeline_stage'], 
            categories=existing_stages, 
            ordered=True
        )
        funnel_data = funnel_data.sort_values('pipeline_stage')
        
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data['pipeline_stage'],
            x=funnel_data['count'],
            textinfo="value+percent initial",
            marker=dict(color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'])
        ))
        fig_funnel.update_layout(title='Sales Pipeline Funnel - Lead Progression')
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    st.markdown("---")
    
    # ENGAGEMENT ANALYSIS
    st.header("📊 Engagement Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Engagement Score vs Lead Score scatter
        fig_scatter = px.scatter(
            filtered_df,
            x='engagement_score',
            y='lead_score',
            color='priority',
            size='website_visits',
            hover_data=['industry', 'company_size'],
            title='Engagement Score vs Lead Score',
            color_discrete_map={'HIGH':'#00CC96', 'MEDIUM':'#FFA15A', 'LOW':'#EF553B'},
            labels={'engagement_score': 'Engagement Score', 'lead_score': 'Lead Score'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Company Size Distribution by Priority
        size_priority = filtered_df.groupby(['company_size', 'priority']).size().reset_index(name='count')
        fig_size = px.bar(
            size_priority,
            x='company_size',
            y='count',
            color='priority',
            title='Company Size Distribution by Priority',
            barmode='group',
            color_discrete_map={'HIGH':'#00CC96', 'MEDIUM':'#FFA15A', 'LOW':'#EF553B'},
            labels={'count': 'Number of Leads', 'company_size': 'Company Size'}
        )
        st.plotly_chart(fig_size, use_container_width=True)
    
    st.markdown("---")
    
    # HIGH PRIORITY LEADS TABLE
    st.header("🎯 Top High Priority Leads - Action Required")
    
    high_priority_leads = filtered_df[filtered_df['priority'] == 'HIGH'].nlargest(10, 'lead_score')
    
    if len(high_priority_leads) > 0:
        # Display as formatted table
        display_cols = ['lead_id', 'industry', 'company_size', 'lead_score', 
                       'engagement_score', 'contact_level', 'pipeline_stage']
        
        # Add styling
        def highlight_score(val):
            if isinstance(val, (int, float)):
                if val >= 80:
                    return 'background-color: #d4edda'
                elif val >= 60:
                    return 'background-color: #fff3cd'
            return ''
        
        styled_table = high_priority_leads[display_cols].style.applymap(
            highlight_score, 
            subset=['lead_score']
        )
        
        st.dataframe(styled_table, use_container_width=True)
        
        # Download button
        csv = high_priority_leads.to_csv(index=False)
        st.download_button(
            label="📥 Download High Priority Leads (CSV)",
            data=csv,
            file_name=f"high_priority_leads_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No high priority leads match the current filters.")
    
    st.markdown("---")
    
    # ALL LEADS TABLE (Expandable)
    with st.expander("📋 View All Filtered Leads"):
        st.dataframe(
            filtered_df[['lead_id', 'industry', 'company_size', 'lead_score', 
                        'priority', 'engagement_score', 'converted']].sort_values('lead_score', ascending=False),
            use_container_width=True
        )
        
        # Download all filtered leads
        csv_all = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download All Filtered Leads (CSV)",
            data=csv_all,
            file_name=f"filtered_leads_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # REAL-TIME LEAD SCORING TOOL
    st.header("🤖 Real-Time Lead Scoring Tool")
    st.markdown("Enter lead details to get instant score prediction")
    
    if model is not None and scaler is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_company_size = st.selectbox("Company Size", ['Small', 'Medium', 'Large'])
            new_industry = st.selectbox("Industry", df['industry'].unique())
            new_location = st.selectbox("Location", df['location'].unique())
        
        with col2:
            new_engagement = st.slider("Engagement Score", 0, 100, 50)
            new_website_visits = st.number_input("Website Visits", 0, 100, 5)
            new_email_opens = st.number_input("Email Opens", 0, 50, 3)
        
        with col3:
            new_demo = st.selectbox("Demo Requested", ['No', 'Yes'])
            new_contact_level = st.selectbox("Contact Level", ['Employee', 'Manager', 'C-Level'])
            new_budget = st.number_input("Budget Indicated (LKR)", 100000, 50000000, 1000000)
        
        if st.button("🎯 Calculate Lead Score", type="primary"):
            try:
                # Prepare input data
                new_lead = pd.DataFrame({
                    'company_size': [new_company_size],
                    'industry': [new_industry],
                    'location': [new_location],
                    'engagement_score': [new_engagement],
                    'website_visits': [new_website_visits],
                    'email_opens': [new_email_opens],
                    'demo_requested': [1 if new_demo == 'Yes' else 0],
                    'contact_level': [new_contact_level],
                    'budget_indicated_lkr': [new_budget],
                    'annual_revenue_lkr': [5000000],  # Default value
                    'days_since_first_contact': [1],
                    'competitor_using': [0],
                    'referral_source': ['Website'],
                    'pipeline_stage': ['New']
                })
                
                # Encode and engineer features
                for col in ['company_size', 'industry', 'location', 'contact_level', 'referral_source', 'pipeline_stage']:
                    new_lead[col + '_encoded'] = encoders[col].transform(new_lead[col])
                
                new_lead['engagement_demo'] = new_lead['engagement_score'] * new_lead['demo_requested']
                new_lead['budget_revenue_ratio'] = new_lead['budget_indicated_lkr'] / (new_lead['annual_revenue_lkr'] + 1)
                new_lead['total_engagement'] = new_lead['website_visits'] + (new_lead['email_opens'] * 2)
                
                contact_scores = {'Employee': 30, 'Manager': 60, 'C-Level': 100}
                new_lead['contact_quality'] = new_lead['contact_level'].map(contact_scores)
                
                # Predict
                features = new_lead[feature_cols].values
                features_scaled = scaler.transform(features)
                
                prediction_proba = model.predict_proba(features_scaled)[0][1]
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
    else:
        st.warning("⚠️ Models not loaded. Please train the models first.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Dashboard created with Streamlit** | Data Science Internship Project")

else:
    st.error("Unable to load data. Please check that the data file exists.")
    st.info("""
    **Quick Fix:**
    1. Make sure you're in the project root directory
    2. Run: `python scripts/lead_scoring_system.py`
    3. Refresh this dashboard
    """)