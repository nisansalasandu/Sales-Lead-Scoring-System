import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_realistic_sales_data(n_leads=1000):
    """
    Generate realistic sales lead data with 3-5% conversion rate
    """
    print("🔄 Generating realistic sales lead dataset...")
    
    # Base data generation
    data = {
        'lead_id': range(1, n_leads + 1),
        'company_size': np.random.choice(['Small', 'Medium', 'Large'], n_leads, p=[0.6, 0.3, 0.1]),
        'industry': np.random.choice([
            'IT/Software', 'Manufacturing', 'Retail', 'Finance', 
            'Healthcare', 'Education', 'Tourism', 'Agriculture'
        ], n_leads),
        'location': np.random.choice([
            'Colombo', 'Kandy', 'Galle', 'Jaffna', 'Negombo', 
            'Kurunegala', 'Matara', 'Trincomalee'
        ], n_leads),
        'engagement_score': np.random.randint(0, 101, n_leads),
        'website_visits': np.random.randint(0, 50, n_leads),
        'email_opens': np.random.randint(0, 20, n_leads),
        'demo_requested': np.random.choice([0, 1], n_leads, p=[0.75, 0.25]),
        'days_since_first_contact': np.random.randint(1, 180, n_leads),
        'contact_level': np.random.choice(['Employee', 'Manager', 'C-Level'], n_leads, p=[0.6, 0.3, 0.1]),
        'budget_indicated_lkr': np.random.randint(100000, 10000000, n_leads),
        'annual_revenue_lkr': np.random.randint(1000000, 500000000, n_leads),
        'competitor_using': np.random.choice([0, 1], n_leads, p=[0.7, 0.3]),
        'referral_source': np.random.choice([
            'Website', 'Social Media', 'Trade Show', 'Referral', 'Cold Call'
        ], n_leads)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate realistic conversion probability
    conversion_prob = (
        0.01 +  # Base 1% conversion
        (df['engagement_score'] / 100) * 0.04 +  # +4% for high engagement
        (df['demo_requested'] * 0.05) +  # +5% if demo requested
        (df['contact_level'].map({'Employee': 0, 'Manager': 0.02, 'C-Level': 0.06})) +
        (df['company_size'].map({'Small': 0, 'Medium': 0.01, 'Large': 0.03})) +
        (df['competitor_using'] * 0.01)  # +1% if using competitor
    )
    
    # Cap probabilities at realistic levels (max 20%)
    conversion_prob = np.clip(conversion_prob, 0, 0.20)
    
    # Generate conversions based on probability
    df['converted'] = np.random.binomial(1, conversion_prob)
    
    # Add pipeline stage (for funnel visualization)
    stages = ['New', 'Contacted', 'Qualified', 'Proposal', 'Negotiation']
    df['pipeline_stage'] = np.random.choice(stages, n_leads, p=[0.4, 0.25, 0.2, 0.1, 0.05])
    
    # Print statistics
    conversion_rate = df['converted'].mean() * 100
    print(f"✅ Dataset generated successfully!")
    print(f"   Total leads: {n_leads}")
    print(f"   Converted: {df['converted'].sum()} ({conversion_rate:.2f}%)")
    print(f"   Not converted: {len(df) - df['converted'].sum()} ({100-conversion_rate:.2f}%)")
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_realistic_sales_data(1000)
    
    # Save to CSV
    output_path = 'data/sales_leads_realistic.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")