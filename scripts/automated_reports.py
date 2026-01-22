import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

def generate_comprehensive_reports(data_path='data/scored_leads.csv'):
    """
    Generate comprehensive automated reports
    """
    print("📊 Generating Automated Reports...")
    
    # Load data
    df = pd.read_csv(data_path)
    report_date = datetime.now().strftime('%Y-%m-%d')
    
    # 1. Executive Summary Report
    generate_executive_summary(df, report_date)
    
    # 2. Daily Lead Quality Report
    generate_daily_quality_report(df, report_date)
    
    # 3. Action Items Report
    generate_action_items(df, report_date)
    
    # 4. Performance Metrics Report
    generate_performance_metrics(df, report_date)
    
    # 5. Industry Analysis Report
    generate_industry_analysis(df, report_date)
    
    # 6. Weekly Trend Report
    generate_weekly_trends(df, report_date)
    
    print("\n✅ All reports generated successfully!")
    print("   Check the reports/ folder")

def generate_executive_summary(df, report_date):
    """
    Generate executive summary report
    """
    print("\n📋 Generating Executive Summary...")
    
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
        'Conversion Value': [df['converted'].sum() * 50000],  # Assuming avg deal value
        'Top Industry': [df['industry'].mode()[0] if len(df) > 0 else 'N/A'],
        'Top Location': [df['location'].mode()[0] if len(df) > 0 else 'N/A'],
        'Avg Engagement Score': [df['engagement_score'].mean()]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'reports/executive_summary_{report_date}.csv', index=False)
    print(f"   ✅ Created: executive_summary_{report_date}.csv")
    
    # Print to console
    print("\n" + "="*60)
    print("📊 EXECUTIVE SUMMARY")
    print("="*60)
    for key, value in summary.items():
        if isinstance(value[0], float):
            print(f"{key:.<40} {value[0]:.2f}")
        else:
            print(f"{key:.<40} {value[0]}")
    print("="*60)

def generate_daily_quality_report(df, report_date):
    """
    Generate daily lead quality report
    """
    print("\n📊 Generating Daily Quality Report...")
    
    quality_metrics = []
    
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        priority_df = df[df['priority'] == priority]
        
        if len(priority_df) > 0:
            metrics = {
                'Priority Level': priority,
                'Number of Leads': len(priority_df),
                'Percentage of Total': (len(priority_df) / len(df)) * 100,
                'Avg Lead Score': priority_df['lead_score'].mean(),
                'Avg Engagement': priority_df['engagement_score'].mean(),
                'Conversion Rate (%)': priority_df['converted'].mean() * 100,
                'Converted Leads': priority_df['converted'].sum(),
                'Demo Requested (%)': (priority_df['demo_requested'].sum() / len(priority_df)) * 100,
                'C-Level Contacts (%)': (priority_df['contact_level'].eq('C-Level').sum() / len(priority_df)) * 100
            }
            quality_metrics.append(metrics)
    
    quality_df = pd.DataFrame(quality_metrics)
    quality_df.to_csv(f'reports/daily_quality_report_{report_date}.csv', index=False)
    print(f"   ✅ Created: daily_quality_report_{report_date}.csv")

def generate_action_items(df, report_date):
    """
    Generate action items for sales team
    """
    print("\n🎯 Generating Action Items...")
    
    # High priority leads requiring immediate action
    high_priority = df[df['priority'] == 'HIGH'].nlargest(20, 'lead_score')
    
    action_items = []
    for idx, row in high_priority.iterrows():
        # Determine action based on lead characteristics
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
    
    action_df = pd.DataFrame(action_items)
    action_df.to_csv(f'reports/action_items_{report_date}.csv', index=False)
    print(f"   ✅ Created: action_items_{report_date}.csv")
    
    # Print top 5 urgent actions
    print("\n🚨 TOP 5 URGENT ACTIONS:")
    for i, item in enumerate(action_items[:5], 1):
        print(f"\n{i}. Lead #{item['Lead ID']} - {item['Industry']}")
        print(f"   Score: {item['Lead Score']:.1f}/100")
        print(f"   Action: {item['Recommended Action']}")

def generate_performance_metrics(df, report_date):
    """
    Generate detailed performance metrics
    """
    print("\n📈 Generating Performance Metrics...")
    
    metrics = {
        'Metric': [],
        'Value': [],
        'Target': [],
        'Status': []
    }
    
    # Define metrics and targets
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
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'reports/performance_metrics_{report_date}.csv', index=False)
    print(f"   ✅ Created: performance_metrics_{report_date}.csv")

def generate_industry_analysis(df, report_date):
    """
    Generate industry-wise analysis
    """
    print("\n🏭 Generating Industry Analysis...")
    
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
    
    industry_df = pd.DataFrame(industry_metrics).sort_values('Total Leads', ascending=False)
    industry_df.to_csv(f'reports/industry_analysis_{report_date}.csv', index=False)
    print(f"   ✅ Created: industry_analysis_{report_date}.csv")

def generate_weekly_trends(df, report_date):
    """
    Generate weekly trend analysis
    """
    print("\n📅 Generating Weekly Trends...")
    
    # Simulate weekly data (in real scenario, you'd have date columns)
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
    
    trend_data = {
        'Week': weeks,
        'Total Leads': [len(df)//4] * 4,
        'High Priority': [len(df[df['priority'] == 'HIGH'])//4] * 4,
        'Conversions': [df['converted'].sum()//4] * 4,
        'Avg Lead Score': [df['lead_score'].mean()] * 4
    }
    
    trend_df = pd.DataFrame(trend_data)
    trend_df.to_csv(f'reports/weekly_trends_{report_date}.csv', index=False)
    print(f"   ✅ Created: weekly_trends_{report_date}.csv")

if __name__ == "__main__":
    generate_comprehensive_reports()
    print("\n" + "="*60)
    print("✅ ALL AUTOMATED REPORTS GENERATED!")
    print("="*60)
    print("\nGenerated Reports:")
    print("  1. Executive Summary")
    print("  2. Daily Quality Report")
    print("  3. Action Items")
    print("  4. Performance Metrics")
    print("  5. Industry Analysis")
    print("  6. Weekly Trends")
    print("\nAll reports saved in: reports/ folder")