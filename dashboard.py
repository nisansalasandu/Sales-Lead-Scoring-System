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
    .report-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
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

# Report Generation Functions
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
        if row['demo_requested'] == 1:
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
    
    metric_data = [
        ('High Priority Lead %', (len(df[df['priority'] == 'HIGH']) / len(df)) * 100, 20, '>='),
        ('Avg Lead Score', df['lead_score'].mean(), 50, '>='),
        ('Overall Conversion Rate %', df['converted'].mean() * 100, 5, '>='),
        ('High Priority Conversion %', df[df['priority'] == 'HIGH']['converted'].mean() * 100, 15, '>='),
        ('Avg Engagement Score', df['engagement_score'].mean(), 50, '>='),
        ('C-Level Contact %', (df['contact_level'].eq('C-Level').sum() / len(df)) * 100, 10, '>='),
        ('Demo Request Rate %', (df['demo_requested'].sum() / len(df)) * 100, 25, '>=')
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
    stage_order = ['New', 'Contacted', 'Qualified', 'Proposal', 'Negotiation']
    funnel_metrics = []
    
    for i, stage in enumerate(stage_order):
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
        
        # VISUALIZATIONS ROW 1
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
            fig_hist.update_layout(showlegend=True)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # VISUALIZATIONS ROW 2
        col3, col4 = st.columns(2)
        
        with col3:
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
            display_cols = ['lead_id', 'industry', 'company_size', 'lead_score', 
                           'engagement_score', 'contact_level', 'pipeline_stage']
            
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
        
        with st.expander("📋 View All Filtered Leads"):
            st.dataframe(
                filtered_df[['lead_id', 'industry', 'company_size', 'lead_score', 
                            'priority', 'engagement_score', 'converted']].sort_values('lead_score', ascending=False),
                use_container_width=True
            )
            
            csv_all = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Filtered Leads (CSV)",
                data=csv_all,
                file_name=f"filtered_leads_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # PAGE 2: REPORTS & ANALYTICS
    elif page == "📋 Reports & Analytics":
        st.header("📋 Automated Reports & Analytics")
        
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        # Report tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Executive Summary", 
            "🎯 Action Items", 
            "📈 Performance Metrics",
            "🏭 Industry Analysis",
            "🔄 Funnel Analysis",
            "📥 Export All Reports"
        ])
        
        # TAB 1: Executive Summary
        with tab1:
            st.subheader("📊 Executive Summary Report")
            
            exec_summary = generate_executive_summary(filtered_df, report_date)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Leads", f"{exec_summary['Total Leads'].values[0]:,}")
                st.metric("High Priority", f"{exec_summary['High Priority Leads'].values[0]:,}")
                st.metric("Medium Priority", f"{exec_summary['Medium Priority Leads'].values[0]:,}")
            
            with col2:
                st.metric("Avg Lead Score", f"{exec_summary['Average Lead Score'].values[0]:.1f}/100")
                st.metric("Conversion Rate", f"{exec_summary['Overall Conversion Rate (%)'].values[0]:.1f}%")
                st.metric("High Priority Conv.", f"{exec_summary['High Priority Conversion (%)'].values[0]:.1f}%")
            
            with col3:
                st.metric("Total Converted", f"{exec_summary['Total Converted Leads'].values[0]:,}")
                st.metric("Conversion Value", f"LKR {exec_summary['Conversion Value'].values[0]:,.0f}")
                st.metric("Avg Engagement", f"{exec_summary['Avg Engagement Score'].values[0]:.1f}/100")
            
            st.markdown("---")
            
            st.markdown("### 📋 Detailed Summary")
            st.dataframe(exec_summary.T, use_container_width=True)
            
            csv_exec = exec_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Executive Summary (CSV)",
                data=csv_exec,
                file_name=f"executive_summary_{report_date}.csv",
                mime="text/csv"
            )
        
        # TAB 2: Action Items
        with tab2:
            st.subheader("🎯 Action Items for Sales Team")
            
            action_items = generate_action_items(filtered_df, report_date)
            
            if len(action_items) > 0:
                # Show urgent items first
                urgent_items = action_items[action_items['Urgency'] == 'URGENT']
                
                if len(urgent_items) > 0:
                    st.markdown("### 🚨 URGENT - Immediate Action Required")
                    st.dataframe(urgent_items, use_container_width=True)
                
                st.markdown("### 📋 All High Priority Action Items")
                
                # Color code by urgency
                def color_urgency(val):
                    if val == 'URGENT':
                        return 'background-color: #ffcccc'
                    elif val == 'HIGH':
                        return 'background-color: #ffffcc'
                    return ''
                
                styled_actions = action_items.style.applymap(
                    color_urgency,
                    subset=['Urgency']
                )
                
                st.dataframe(styled_actions, use_container_width=True)
                
                csv_actions = action_items.to_csv(index=False)
                st.download_button(
                    label="📥 Download Action Items (CSV)",
                    data=csv_actions,
                    file_name=f"action_items_{report_date}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No high priority leads requiring action at this time.")
        
        # TAB 3: Performance Metrics
        with tab3:
            st.subheader("📈 Performance Metrics Dashboard")
            
            perf_metrics = generate_performance_metrics(filtered_df, report_date)
            
            # Visualize metrics
            fig_perf = go.Figure()
            
            fig_perf.add_trace(go.Bar(
                name='Current Value',
                x=perf_metrics['Metric'],
                y=perf_metrics['Value'],
                marker_color='lightblue'
            ))
            
            fig_perf.add_trace(go.Scatter(
                name='Target',
                x=perf_metrics['Metric'],
                y=perf_metrics['Target'],
                mode='markers',
                marker=dict(size=12, color='red', symbol='line-ns-open')
            ))
            
            fig_perf.update_layout(
                title='Performance Metrics vs Targets',
                xaxis_title='Metric',
                yaxis_title='Value',
                height=500
            )
            
            st.plotly_chart(fig_perf, use_container_width=True)
            
            st.markdown("### 📊 Detailed Metrics")
            
            # Color code status
            def color_status(val):
                if '✅' in str(val):
                    return 'background-color: #d4edda'
                elif '⚠️' in str(val):
                    return 'background-color: #fff3cd'
                return ''
            
            styled_perf = perf_metrics.style.applymap(
                color_status,
                subset=['Status']
            )
            
            st.dataframe(styled_perf, use_container_width=True)
            
            csv_perf = perf_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Performance Metrics (CSV)",
                data=csv_perf,
                file_name=f"performance_metrics_{report_date}.csv",
                mime="text/csv"
            )
        
        # TAB 4: Industry Analysis
        with tab4:
            st.subheader("🏭 Industry-wise Performance Analysis")
            
            industry_analysis = generate_industry_analysis(filtered_df, report_date)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_ind_leads = px.bar(
                    industry_analysis.head(10),
                    x='Industry',
                    y='Total Leads',
                    title='Top 10 Industries by Lead Count',
                    color='Total Leads',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_ind_leads, use_container_width=True)
            
            with col2:
                fig_ind_conv = px.bar(
                    industry_analysis.head(10),
                    x='Industry',
                    y='Conversion Rate (%)',
                    title='Top 10 Industries by Conversion Rate',
                    color='Conversion Rate (%)',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_ind_conv, use_container_width=True)
            
            st.markdown("### 📊 Complete Industry Analysis")
            st.dataframe(industry_analysis, use_container_width=True)
            
            csv_industry = industry_analysis.to_csv(index=False)
            st.download_button(
                label="📥 Download Industry Analysis (CSV)",
                data=csv_industry,
                file_name=f"industry_analysis_{report_date}.csv",
                mime="text/csv"
            )
        
        # TAB 5: Funnel Analysis
        with tab5:
            st.subheader("🔄 Sales Funnel Deep Dive")
            
            funnel_metrics = analyze_sales_funnel(filtered_df)
            
            # Main funnel visualization
            fig_funnel_detail = go.Figure(go.Funnel(
                y=funnel_metrics['stage'],
                x=funnel_metrics['total_leads'],
                textinfo="value+percent initial",
                marker=dict(color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']),
                connector=dict(line=dict(color="royalblue", width=3))
            ))
            fig_funnel_detail.update_layout(
                title='Sales Pipeline Funnel - Detailed Analysis',
                height=500
            )
            st.plotly_chart(fig_funnel_detail, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Conversion rate by stage
                fig_conv_stage = px.bar(
                    funnel_metrics,
                    x='stage',
                    y='conversion_rate',
                    title='Conversion Rate by Stage',
                    text='conversion_rate',
                    color='conversion_rate',
                    color_continuous_scale='RdYlGn'
                )
                fig_conv_stage.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_conv_stage, use_container_width=True)
            
            with col2:
                # Drop-off analysis
                fig_dropoff = go.Figure()
                fig_dropoff.add_trace(go.Scatter(
                    x=funnel_metrics['stage'],
                    y=funnel_metrics['drop_off_rate'],
                    mode='lines+markers+text',
                    text=[f"{x:.1f}%" for x in funnel_metrics['drop_off_rate']],
                    textposition="top center",
                    marker=dict(size=12, color='red'),
                    line=dict(width=3, color='red')
                ))
                fig_dropoff.update_layout(
                    title='Lead Drop-off Rate Between Stages',
                    xaxis_title='Pipeline Stage',
                    yaxis_title='Drop-off Rate (%)',
                    height=400
                )
                st.plotly_chart(fig_dropoff, use_container_width=True)
            
            st.markdown("### 📊 Detailed Funnel Metrics")
            st.dataframe(funnel_metrics, use_container_width=True)
            
            # Key insights
            max_dropoff = funnel_metrics[funnel_metrics['drop_off_rate'] > 0]['drop_off_rate'].max()
            if not pd.isna(max_dropoff):
                bottleneck_stage = funnel_metrics[funnel_metrics['drop_off_rate'] == max_dropoff]['stage'].values[0]
                st.warning(f"⚠️ **Biggest Bottleneck:** {bottleneck_stage} stage with {max_dropoff:.1f}% drop-off")
            
            best_conv_stage = funnel_metrics.loc[funnel_metrics['conversion_rate'].idxmax(), 'stage']
            best_conv_rate = funnel_metrics['conversion_rate'].max()
            st.success(f"✅ **Best Conversion:** {best_conv_stage} stage with {best_conv_rate:.1f}% conversion rate")
            
            csv_funnel = funnel_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Funnel Metrics (CSV)",
                data=csv_funnel,
                file_name=f"funnel_metrics_{report_date}.csv",
                mime="text/csv"
            )
        
        # TAB 6: Export All Reports
        with tab6:
            st.subheader("📥 Export All Reports")
            
            st.markdown("""
            Generate and download all reports at once for easy distribution to your team.
            """)
            
            if st.button("🔄 Generate All Reports", type="primary"):
                with st.spinner("Generating all reports..."):
                    # Create reports folder if it doesn't exist
                    os.makedirs('reports', exist_ok=True)
                    
                    # Generate all reports
                    exec_sum = generate_executive_summary(filtered_df, report_date)
                    actions = generate_action_items(filtered_df, report_date)
                    perf = generate_performance_metrics(filtered_df, report_date)
                    industry = generate_industry_analysis(filtered_df, report_date)
                    funnel = analyze_sales_funnel(filtered_df)
                    
                    # Save to files
                    exec_sum.to_csv(f'reports/executive_summary_{report_date}.csv', index=False)
                    actions.to_csv(f'reports/action_items_{report_date}.csv', index=False)
                    perf.to_csv(f'reports/performance_metrics_{report_date}.csv', index=False)
                    industry.to_csv(f'reports/industry_analysis_{report_date}.csv', index=False)
                    funnel.to_csv(f'reports/funnel_metrics_{report_date}.csv', index=False)
                    
                    st.success("✅ All reports generated successfully!")
                    
                    st.markdown("### 📋 Generated Reports:")
                    st.markdown(f"- ✅ Executive Summary (`executive_summary_{report_date}.csv`)")
                    st.markdown(f"- ✅ Action Items (`action_items_{report_date}.csv`)")
                    st.markdown(f"- ✅ Performance Metrics (`performance_metrics_{report_date}.csv`)")
                    st.markdown(f"- ✅ Industry Analysis (`industry_analysis_{report_date}.csv`)")
                    st.markdown(f"- ✅ Funnel Metrics (`funnel_metrics_{report_date}.csv`)")
                    
                    st.info("📁 All reports saved to: `reports/` folder")
            
            st.markdown("---")
            
            st.markdown("### 📊 Individual Report Downloads")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Executive Summary"):
                    exec_sum = generate_executive_summary(filtered_df, report_date)
                    csv_data = exec_sum.to_csv(index=False)
                    st.download_button(
                        label="📥 Download",
                        data=csv_data,
                        file_name=f"executive_summary_{report_date}.csv",
                        mime="text/csv"
                    )
                
                if st.button("Download Performance Metrics"):
                    perf = generate_performance_metrics(filtered_df, report_date)
                    csv_data = perf.to_csv(index=False)
                    st.download_button(
                        label="📥 Download",
                        data=csv_data,
                        file_name=f"performance_metrics_{report_date}.csv",
                        mime="text/csv"
                    )
                
                if st.button("Download Funnel Metrics"):
                    funnel = analyze_sales_funnel(filtered_df)
                    csv_data = funnel.to_csv(index=False)
                    st.download_button(
                        label="📥 Download",
                        data=csv_data,
                        file_name=f"funnel_metrics_{report_date}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Download Action Items"):
                    actions = generate_action_items(filtered_df, report_date)
                    csv_data = actions.to_csv(index=False)
                    st.download_button(
                        label="📥 Download",
                        data=csv_data,
                        file_name=f"action_items_{report_date}.csv",
                        mime="text/csv"
                    )
                
                if st.button("Download Industry Analysis"):
                    industry = generate_industry_analysis(filtered_df, report_date)
                    csv_data = industry.to_csv(index=False)
                    st.download_button(
                        label="📥 Download",
                        data=csv_data,
                        file_name=f"industry_analysis_{report_date}.csv",
                        mime="text/csv"
                    )
    
    # PAGE 3: LEAD SCORING TOOL
    elif page == "🤖 Lead Scoring Tool":
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
                        'annual_revenue_lkr': [5000000],
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