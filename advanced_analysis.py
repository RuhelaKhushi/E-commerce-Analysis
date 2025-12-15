import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = 'analysis_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("="*70)
print("E-COMMERCE ADVANCED ANALYTICS")
print("="*70)

print("\n[1/5] Loading and preparing data...")
DATA_PATH = r'data.csv'

try:
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    print(f"✓ Loaded {len(df):,} transactions")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df_with_customers = df.dropna(subset=['CustomerID'])

print(f"✓ Clean dataset: {len(df_with_customers):,} transactions with customer data")

print("\n[2/5] Performing RFM Analysis...")

snapshot_date = df_with_customers['InvoiceDate'].max() + timedelta(days=1)

rfm = df_with_customers.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm['R_Score'] = rfm['R_Score'].astype(int)
rfm['F_Score'] = rfm['F_Score'].astype(int)
rfm['M_Score'] = rfm['M_Score'].astype(int)

rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

def segment_customer(row):
    if row['RFM_Score'] >= 13:
        return 'Champions'
    elif row['RFM_Score'] >= 11:
        return 'Loyal Customers'
    elif row['RFM_Score'] >= 9:
        return 'Potential Loyalists'
    elif row['RFM_Score'] >= 7:
        return 'At Risk'
    elif row['RFM_Score'] >= 5:
        return 'Need Attention'
    else:
        return 'Lost'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

print("\n" + "="*70)
print("RFM ANALYSIS RESULTS")
print("="*70)
print("\nCustomer Segments Distribution:")
print(rfm['Segment'].value_counts())
print("\nRFM Statistics:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe())

fig, ax = plt.subplots(figsize=(12, 6))
segment_counts = rfm['Segment'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
bars = ax.bar(segment_counts.index, segment_counts.values, color=colors, alpha=0.8, edgecolor='black')
ax.set_title('Customer Segmentation (RFM Analysis)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Customer Segment', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/17_rfm_customer_segments.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 17_rfm_customer_segments.png")

segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.Spectral(np.linspace(0, 1, len(segment_revenue)))
bars = ax.bar(segment_revenue.index, segment_revenue.values, color=colors, alpha=0.8, edgecolor='black')
ax.set_title('Revenue Contribution by Customer Segment', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Customer Segment', fontsize=12)
ax.set_ylabel('Total Revenue (£)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/18_rfm_segment_revenue.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 18_rfm_segment_revenue.png")

from mpl_toolkits.mplot3d import Axes3D

rfm_sample = rfm.sample(min(1000, len(rfm)))

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

segment_colors = {
    'Champions': 'red',
    'Potential Loyalists': 'blue',
    'Need Attention': 'green'
}

for segment in rfm_sample['Segment'].unique():
    segment_data = rfm_sample[rfm_sample['Segment'] == segment]
    ax.scatter(
        segment_data['Recency'],
        segment_data['Frequency'],
        label=segment,
        c=segment_colors.get(segment, 'gray')
    )


ax.set_xlabel('Recency (days)', fontsize=10)
ax.set_ylabel('Frequency (purchases)', fontsize=10)
ax.set_zlabel('Monetary (£)', fontsize=10)
ax.set_title('RFM Analysis: 3D Customer Segmentation', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/19_rfm_3d_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 19_rfm_3d_scatter.png")

print("\n[3/5] Performing Cohort Analysis...")

df_with_customers['OrderMonth'] = df_with_customers['InvoiceDate'].dt.to_period('M')
df_with_customers['CohortMonth'] = df_with_customers.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')

def get_cohort_index(row):
    cohort_date = row['CohortMonth']
    order_date = row['OrderMonth']
    return (order_date.year - cohort_date.year) * 12 + (order_date.month - cohort_date.month)

df_with_customers['CohortIndex'] = df_with_customers.apply(get_cohort_index, axis=1)

cohort_data = df_with_customers.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')

cohort_size = cohort_pivot.iloc[:, 0]
retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100

print("\n" + "="*70)
print("COHORT ANALYSIS RESULTS")
print("="*70)
print("\nRetention Matrix (first 6 months):")
print(retention_matrix.iloc[:, :7].round(1))

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(retention_matrix.iloc[:, :12], annot=True, fmt='.0f', cmap='RdYlGn', 
            ax=ax, cbar_kws={"shrink": 0.8}, vmin=0, vmax=100)
ax.set_title('Customer Retention Cohort Analysis (% Retained)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Cohort Index (Months Since First Purchase)', fontsize=12)
ax.set_ylabel('Cohort Month', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/20_cohort_retention_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 20_cohort_retention_heatmap.png")

avg_retention = retention_matrix.mean(axis=0).iloc[:12]
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    avg_retention.index,
    avg_retention.values,
    marker='o',
    linewidth=2.5,
    markersize=10,
    color='blue'
)

ax.set_title(
    'Average Customer Retention Rate Over Time',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax.set_xlabel('Months Since First Purchase', fontsize=12)
ax.set_ylabel('Retention Rate (%)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/21_average_retention_rate.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: 21_average_retention_rate.png")

print("\n[4/5] Performing Product Basket Analysis...")

basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

from itertools import combinations

top_products = df.groupby('Description')['Quantity'].sum().nlargest(30).index.tolist()
basket_top = basket[top_products]

co_occurrence = basket_top.T.dot(basket_top)
np.fill_diagonal(co_occurrence.values, 0)

print("\n" + "="*70)
print("BASKET ANALYSIS RESULTS")
print("="*70)
print("\nTop Product Pairs (Frequently Bought Together):")

pairs = []
for i in range(len(co_occurrence)):
    for j in range(i+1, len(co_occurrence)):
        pairs.append((co_occurrence.index[i], co_occurrence.columns[j], co_occurrence.iloc[i, j]))

pairs_df = pd.DataFrame(pairs, columns=['Product1', 'Product2', 'Co-occurrence'])
pairs_df = pairs_df.sort_values('Co-occurrence', ascending=False).head(10)
print(pairs_df.to_string(index=False))

top_20_products = df.groupby('Description')['Quantity'].sum().nlargest(20).index.tolist()
co_occurrence_top20 = co_occurrence.loc[top_20_products, top_20_products]

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(co_occurrence_top20, cmap='YlOrRd', ax=ax, cbar_kws={"shrink": 0.8},
            xticklabels=[p[:30] + '...' if len(p) > 30 else p for p in co_occurrence_top20.columns],
            yticklabels=[p[:30] + '...' if len(p) > 30 else p for p in co_occurrence_top20.index])
ax.set_title('Product Co-occurrence Matrix (Top 20 Products)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/22_product_cooccurrence_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 22_product_cooccurrence_heatmap.png")

print("\n[5/5] Calculating Customer Lifetime Value...")

clv_data = df_with_customers.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum',
    'InvoiceDate': ['min', 'max']
}).reset_index()

clv_data.columns = ['CustomerID', 'PurchaseCount', 'TotalSpent', 'FirstPurchase', 'LastPurchase']

clv_data['Lifespan'] = (clv_data['LastPurchase'] - clv_data['FirstPurchase']).dt.days

clv_data['AvgPurchaseValue'] = clv_data['TotalSpent'] / clv_data['PurchaseCount']
clv_data['PurchaseFrequency'] = clv_data['PurchaseCount'] / (clv_data['Lifespan'] + 1)

clv_data['EstimatedCLV'] = clv_data['AvgPurchaseValue'] * clv_data['PurchaseFrequency'] * 365

print("\n" + "="*70)
print("CUSTOMER LIFETIME VALUE ANALYSIS")
print("="*70)
print("\nCLV Statistics:")
print(clv_data[['TotalSpent', 'PurchaseCount', 'AvgPurchaseValue', 'EstimatedCLV']].describe())

fig, ax = plt.subplots(figsize=(12, 6))

clv_filtered = clv_data[
    clv_data['EstimatedCLV'] < clv_data['EstimatedCLV'].quantile(0.95)
]

ax.hist(
    clv_filtered['EstimatedCLV'],
    bins=50,
    color='green'
)

ax.set_title(
    'Customer Lifetime Value Distribution (95th Percentile)',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax.set_xlabel('Estimated CLV (£)', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/23_clv_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: 23_clv_distribution.png")

fig, ax = plt.subplots(figsize=(12, 8))
scatter_data = clv_data[(clv_data['AvgPurchaseValue'] < clv_data['AvgPurchaseValue'].quantile(0.95)) &
                        (clv_data['PurchaseCount'] < 100)]
scatter = ax.scatter(scatter_data['PurchaseCount'], scatter_data['AvgPurchaseValue'], 
                     c=scatter_data['TotalSpent'], cmap='viridis', alpha=0.6, s=50)
ax.set_title('Customer Segmentation: Purchase Frequency vs Average Purchase Value', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Purchase Count', fontsize=12)
ax.set_ylabel('Average Purchase Value (£)', fontsize=12)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Total Spent (£)', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/24_purchase_frequency_vs_value.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 24_purchase_frequency_vs_value.png")

print("\n" + "="*70)
print("ADVANCED ANALYTICS COMPLETE - KEY INSIGHTS")
print("="*70)

print(f"\n🎯 RFM SEGMENTATION:")
print(f"   • Champions: {len(rfm[rfm['Segment'] == 'Champions']):,} customers")
print(f"   • Loyal Customers: {len(rfm[rfm['Segment'] == 'Loyal Customers']):,} customers")
print(f"   • At Risk: {len(rfm[rfm['Segment'] == 'At Risk']):,} customers")
print(f"   • Lost: {len(rfm[rfm['Segment'] == 'Lost']):,} customers")

print(f"\n📊 COHORT INSIGHTS:")
first_month_retention = retention_matrix.iloc[0, 1] if len(retention_matrix) > 0 and retention_matrix.shape[1] > 1 else 0
print(f"   • Average 1-Month Retention: {first_month_retention:.1f}%")
print(f"   • Total Cohorts Analyzed: {len(cohort_pivot)}")

print(f"\n💰 CUSTOMER LIFETIME VALUE:")
print(f"   • Average CLV: £{clv_data['EstimatedCLV'].mean():,.2f}")
print(f"   • Median CLV: £{clv_data['EstimatedCLV'].median():,.2f}")
print(f"   • Top 10% CLV: £{clv_data['EstimatedCLV'].quantile(0.9):,.2f}")

print("\n" + "="*70)
print(f"✓ All advanced analytics visualizations saved to: {OUTPUT_DIR}/")
print(f"✓ Total graphs generated: 8 (Charts 17-24)")
print("="*70)

print("\n✅ Advanced analytics complete!")
