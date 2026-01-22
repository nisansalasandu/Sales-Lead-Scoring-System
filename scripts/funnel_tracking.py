import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

def analyze_sales_funnel(data_path='data/scored_leads.csv'):
    """
    Analyze and track leads through the sales funnel
    """
    print("🔄 Analyzing Sales Funnel...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Define funnel stages in order
    stage_order = ['New', 'Contacted', 'Qualified', 'Proposal', 'Negotiation']
    
    # Calculate metrics for each stage
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
    
    # Save funnel metrics
    funnel_df.to_csv('reports/funnel_metrics.csv', index=False)
    print("   ✅ Funnel metrics saved to: reports/funnel_metrics.csv")
    
    # Create funnel visualization
    create_funnel_visualizations(funnel_df, df)
    
    # Print summary
    print_funnel_summary(funnel_df)
    
    return funnel_df

def create_funnel_visualizations(funnel_df, df):
    """
    Create comprehensive funnel visualizations
    """
    print("\n📊 Creating funnel visualizations...")
    
    # 1. Main Funnel Chart
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_df['stage'],
        x=funnel_df['total_leads'],
        textinfo="value+percent initial",
        marker=dict(
            color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
        ),
        connector=dict(line=dict(color="royalblue", width=3))
    ))
    
    fig_funnel.update_layout(
        title='Sales Pipeline Funnel - Lead Progression',
        height=500
    )
    fig_funnel.write_html('reports/sales_funnel_main.html')
    print("   ✅ Created: sales_funnel_main.html")
    
    # 2. Conversion Rate by Stage
    fig_conv = px.bar(
        funnel_df,
        x='stage',
        y='conversion_rate',
        title='Conversion Rate by Pipeline Stage',
        text='conversion_rate',
        color='conversion_rate',
        color_continuous_scale='RdYlGn'
    )
    fig_conv.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_conv.write_html('reports/conversion_by_stage.html')
    print("   ✅ Created: conversion_by_stage.html")
    
    # 3. Priority Distribution by Stage
    stage_priority = df.groupby(['pipeline_stage', 'priority']).size().reset_index(name='count')
    fig_priority = px.bar(
        stage_priority,
        x='pipeline_stage',
        y='count',
        color='priority',
        title='Priority Distribution Across Pipeline Stages',
        barmode='stack',
        color_discrete_map={'HIGH':'#00CC96', 'MEDIUM':'#FFA15A', 'LOW':'#EF553B'},
        category_orders={'pipeline_stage': funnel_df['stage'].tolist()}
    )
    fig_priority.write_html('reports/priority_by_stage.html')
    print("   ✅ Created: priority_by_stage.html")
    
    # 4. Drop-off Analysis
    fig_dropoff = go.Figure()
    fig_dropoff.add_trace(go.Scatter(
        x=funnel_df['stage'],
        y=funnel_df['drop_off_rate'],
        mode='lines+markers+text',
        text=[f"{x:.1f}%" for x in funnel_df['drop_off_rate']],
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
    fig_dropoff.write_html('reports/dropoff_analysis.html')
    print("   ✅ Created: dropoff_analysis.html")

def print_funnel_summary(funnel_df):
    """
    Print funnel summary to console
    """
    print("\n" + "="*70)
    print("🔄 SALES FUNNEL ANALYSIS SUMMARY")
    print("="*70)
    
    for _, row in funnel_df.iterrows():
        print(f"\n📍 Stage {row['stage_number']}: {row['stage']}")
        print(f"   Total Leads: {row['total_leads']}")
        print(f"   Converted: {row['converted_leads']} ({row['conversion_rate']:.1f}%)")
        print(f"   Avg Lead Score: {row['avg_lead_score']:.1f}/100")
        print(f"   Priority: HIGH={row['high_priority']}, MED={row['medium_priority']}, LOW={row['low_priority']}")
        if row['drop_off_rate'] > 0:
            print(f"   ⚠️  Drop-off from previous stage: {row['drop_off_rate']:.1f}%")
    
    print("\n" + "="*70)
    
    # Identify bottlenecks
    print("\n🔍 KEY INSIGHTS:")
    max_dropoff = funnel_df[funnel_df['drop_off_rate'] > 0]['drop_off_rate'].max()
    if not pd.isna(max_dropoff):
        bottleneck_stage = funnel_df[funnel_df['drop_off_rate'] == max_dropoff]['stage'].values[0]
        print(f"   ⚠️  Biggest bottleneck: {bottleneck_stage} stage ({max_dropoff:.1f}% drop-off)")
    
    best_conv_stage = funnel_df.loc[funnel_df['conversion_rate'].idxmax(), 'stage']
    best_conv_rate = funnel_df['conversion_rate'].max()
    print(f"   ✅ Best conversion: {best_conv_stage} stage ({best_conv_rate:.1f}%)")
    
    total_leads = funnel_df['total_leads'].sum()
    total_converted = funnel_df['converted_leads'].sum()
    overall_conv = (total_converted / total_leads * 100) if total_leads > 0 else 0
    print(f"   📊 Overall pipeline conversion: {overall_conv:.1f}%")
    
    print("="*70)

if __name__ == "__main__":
    analyze_sales_funnel()
    print("\n✅ Funnel tracking completed!")
    print("   View reports in the reports/ folder")