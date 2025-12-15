import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = 'analysis_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✓ Created output directory: {OUTPUT_DIR}")

print("="*70)
print("E-COMMERCE DATA ANALYSIS - COMPREHENSIVE REPORT")
print("="*70)

print("\n[1/10] Loading dataset...")
DATA_PATH = r'data.csv'

try:
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    print(f"✓ Successfully loaded {len(df):,} transactions")
    print(f"✓ Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

print("\n[2/10] Cleaning and preprocessing data...")

original_size = len(df)

print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

df_with_customers = df.dropna(subset=['CustomerID'])

print(f"✓ Removed {original_size - len(df):,} invalid transactions")
print(f"✓ Clean dataset: {len(df):,} transactions")
print(f"✓ Dataset with CustomerID: {len(df_with_customers):,} transactions")

df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
df['Date'] = df['InvoiceDate'].dt.date

print("✓ Created derived features: TotalPrice, Year, Month, Day, DayOfWeek, Hour")

print("\n[3/10] Calculating key metrics...")
print("\n" + "="*70)
print("KEY BUSINESS METRICS")
print("="*70)

total_revenue = df['TotalPrice'].sum()
total_transactions = df['InvoiceNo'].nunique()
total_customers = df_with_customers['CustomerID'].nunique()
total_products = df['StockCode'].nunique()
total_countries = df['Country'].nunique()
avg_order_value = total_revenue / total_transactions
avg_items_per_transaction = df.groupby('InvoiceNo')['Quantity'].sum().mean()

print(f"\n💰 Total Revenue: £{total_revenue:,.2f}")
print(f"📦 Total Transactions: {total_transactions:,}")
print(f"👥 Total Customers: {total_customers:,}")
print(f"🏷️  Total Products: {total_products:,}")
print(f"🌍 Countries Served: {total_countries}")
print(f"💵 Average Order Value: £{avg_order_value:,.2f}")
print(f"📊 Average Items per Transaction: {avg_items_per_transaction:.2f}")
print(f"📅 Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

print("\n[4/10] Performing time series analysis...")

daily_revenue = df.groupby('Date')['TotalPrice'].sum().reset_index()
daily_revenue.columns = ['Date', 'Revenue']

monthly_revenue = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
monthly_revenue.columns = ['YearMonth', 'Revenue']
monthly_revenue['YearMonth'] = monthly_revenue['YearMonth'].astype(str)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_revenue['Date'], daily_revenue['Revenue'], linewidth=1.5, color='#1f77b4')
ax.set_title('Daily Revenue Trend (Dec 2010 - Dec 2011)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Revenue (£)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_daily_revenue_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_daily_revenue_trend.png")

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(monthly_revenue)), monthly_revenue['Revenue'], color='#1f77b4')
ax.set_title('Monthly Revenue Trend', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Revenue (£)', fontsize=12)
ax.set_xticks(range(len(monthly_revenue)))
ax.set_xticklabels(monthly_revenue['YearMonth'], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_monthly_revenue_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_monthly_revenue_trend.png")

dow_revenue = df.groupby('DayOfWeek')['TotalPrice'].sum().reset_index()
dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_revenue['DayName'] = dow_revenue['DayOfWeek'].apply(lambda x: dow_names[x])

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, 7))
bars = ax.bar(dow_revenue['DayName'], dow_revenue['TotalPrice'], color=colors, alpha=0.8)
ax.set_title('Revenue by Day of Week', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Total Revenue (£)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_revenue_by_day_of_week.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_revenue_by_day_of_week.png")

hourly_revenue = df.groupby('Hour')['TotalPrice'].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hourly_revenue['Hour'], hourly_revenue['TotalPrice'], marker='o', linewidth=2, 
        markersize=8, color='#1f77b4')
ax.set_title('Revenue by Hour of Day', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Hour (24-hour format)', fontsize=12)
ax.set_ylabel('Total Revenue (£)', fontsize=12)
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_revenue_by_hour.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_revenue_by_hour.png")

print("\n[5/10] Analyzing product performance...")

product_revenue = df.groupby(['StockCode', 'Description'])['TotalPrice'].sum().reset_index()
product_revenue = product_revenue.sort_values('TotalPrice', ascending=False).head(20)

product_quantity = df.groupby(['StockCode', 'Description'])['Quantity'].sum().reset_index()
product_quantity = product_quantity.sort_values('Quantity', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 8))
top_15_revenue = product_revenue.head(15)
y_pos = np.arange(len(top_15_revenue))
bars = ax.barh(y_pos, top_15_revenue['TotalPrice'], color='#1f77b4')
ax.set_yticks(y_pos)
ax.set_yticklabels([desc[:40] + '...' if len(str(desc)) > 40 else desc 
                     for desc in top_15_revenue['Description']], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Total Revenue (£)', fontsize=12)
ax.set_title('Top 15 Products by Revenue', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_top_products_by_revenue.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_top_products_by_revenue.png")

fig, ax = plt.subplots(figsize=(12, 8))
top_15_quantity = product_quantity.head(15)
y_pos = np.arange(len(top_15_quantity))
bars = ax.barh(y_pos, top_15_quantity['Quantity'], color='#ff7f0e')
ax.set_yticks(y_pos)
ax.set_yticklabels([desc[:40] + '...' if len(str(desc)) > 40 else desc 
                     for desc in top_15_quantity['Description']], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Total Quantity Sold', fontsize=12)
ax.set_title('Top 15 Products by Quantity Sold', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_top_products_by_quantity.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_top_products_by_quantity.png")

print("\n[6/10] Analyzing customer behavior...")

customer_revenue = df_with_customers.groupby('CustomerID')['TotalPrice'].sum().reset_index()
customer_revenue = customer_revenue.sort_values('TotalPrice', ascending=False)

customer_frequency = df_with_customers.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
customer_frequency.columns = ['CustomerID', 'PurchaseCount']

fig, ax = plt.subplots(figsize=(12, 8))
top_20_customers = customer_revenue.head(20)
y_pos = np.arange(len(top_20_customers))
bars = ax.barh(y_pos, top_20_customers['TotalPrice'], color='#2ca02c')
ax.set_yticks(y_pos)
ax.set_yticklabels([f'Customer {int(cid)}' for cid in top_20_customers['CustomerID']], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Total Revenue (£)', fontsize=12)
ax.set_title('Top 20 Customers by Revenue', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_top_customers_by_revenue.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_top_customers_by_revenue.png")

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(customer_frequency['PurchaseCount'], bins=50, color='#9467bd')
ax.set_title('Customer Purchase Frequency Distribution', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Number of Purchases', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_customer_purchase_frequency.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 08_customer_purchase_frequency.png")

print("\n[7/10] Analyzing geographic distribution...")

country_revenue = df.groupby('Country')['TotalPrice'].sum().reset_index()
country_revenue = country_revenue.sort_values('TotalPrice', ascending=False)

country_transactions = df.groupby('Country')['InvoiceNo'].nunique().reset_index()
country_transactions.columns = ['Country', 'TransactionCount']
country_transactions = country_transactions.sort_values('TransactionCount', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
top_15_countries = country_revenue.head(15)
y_pos = np.arange(len(top_15_countries))
bars = ax.barh(y_pos, top_15_countries['TotalPrice'], color='#8c564b')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_15_countries['Country'], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Total Revenue (£)', fontsize=12)
ax.set_title('Top 15 Countries by Revenue', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_top_countries_by_revenue.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 09_top_countries_by_revenue.png")

uk_revenue = country_revenue[country_revenue['Country'] == 'United Kingdom']['TotalPrice'].sum()
international_revenue = country_revenue[country_revenue['Country'] != 'United Kingdom']['TotalPrice'].sum()

fig, ax = plt.subplots(figsize=(10, 8))
sizes = [uk_revenue, international_revenue]
labels = ['United Kingdom', 'International']
colors = ['#ff9999', '#66b3ff']
explode = (0.05, 0)

ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title('Revenue Distribution: UK vs International', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_uk_vs_international_revenue.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 10_uk_vs_international_revenue.png")

print("\n[8/10] Analyzing price and quantity distributions...")

price_filtered = df[df['UnitPrice'] < df['UnitPrice'].quantile(0.95)]

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(price_filtered['UnitPrice'], bins=100, color='#e377c2')
ax.set_title('Unit Price Distribution (95th Percentile)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Unit Price (£)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_unit_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 11_unit_price_distribution.png")

quantity_filtered = df[df['Quantity'] < df['Quantity'].quantile(0.95)]

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(quantity_filtered['Quantity'], bins=100, color='#7f7f7f')
ax.set_title('Quantity Distribution (95th Percentile)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Quantity', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_quantity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 12_quantity_distribution.png")

transaction_values = df.groupby('InvoiceNo')['TotalPrice'].sum()
transaction_filtered = transaction_values[transaction_values < transaction_values.quantile(0.95)]

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(transaction_filtered, bins=100, color='#bcbd22')
ax.set_title('Transaction Value Distribution (95th Percentile)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Transaction Value (£)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/13_transaction_value_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 13_transaction_value_distribution.png")

print("\n[9/10] Performing correlation analysis...")

numerical_features = df[['Quantity', 'UnitPrice', 'TotalPrice', 'Hour', 'DayOfWeek', 'Month']].copy()
correlation_matrix = numerical_features.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/14_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 14_correlation_heatmap.png")

print("\n[10/10] Analyzing seasonal trends...")

monthly_transactions = df.groupby('YearMonth')['InvoiceNo'].nunique().reset_index()
monthly_transactions.columns = ['YearMonth', 'TransactionCount']
monthly_transactions['YearMonth'] = monthly_transactions['YearMonth'].astype(str)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(monthly_transactions)), monthly_transactions['TransactionCount'], 
        marker='o', linewidth=2, markersize=8, color='#17becf')
ax.set_title('Monthly Transaction Count Trend', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Transactions', fontsize=12)
ax.set_xticks(range(len(monthly_transactions)))
ax.set_xticklabels(monthly_transactions['YearMonth'], rotation=45, ha='right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/15_monthly_transaction_count.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 15_monthly_transaction_count.png")

revenue_heatmap_data = df.groupby(['Month', 'DayOfWeek'])['TotalPrice'].sum().reset_index()
revenue_pivot = revenue_heatmap_data.pivot(index='DayOfWeek', columns='Month', values='TotalPrice')

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(revenue_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title('Revenue Heatmap: Month vs Day of Week', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Day of Week', fontsize=12)
ax.set_yticklabels(dow_names, rotation=0)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/16_revenue_heatmap_month_dow.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 16_revenue_heatmap_month_dow.png")

print("\n" + "="*70)
print("ANALYSIS COMPLETE - KEY INSIGHTS")
print("="*70)

print(f"\n📊 BUSINESS PERFORMANCE:")
print(f"   • Total Revenue: £{total_revenue:,.2f}")
print(f"   • Average Order Value: £{avg_order_value:,.2f}")
print(f"   • Total Customers: {total_customers:,}")
print(f"   • Total Products: {total_products:,}")

print(f"\n🌍 GEOGRAPHIC INSIGHTS:")
print(f"   • UK Revenue: £{uk_revenue:,.2f} ({uk_revenue/total_revenue*100:.1f}%)")
print(f"   • International Revenue: £{international_revenue:,.2f} ({international_revenue/total_revenue*100:.1f}%)")
print(f"   • Countries Served: {total_countries}")

print(f"\n📈 TOP PERFORMERS:")
top_product = product_revenue.iloc[0]
print(f"   • Best Product: {top_product['Description']}")
print(f"     Revenue: £{top_product['TotalPrice']:,.2f}")

top_customer = customer_revenue.iloc[0]
print(f"   • Best Customer: {int(top_customer['CustomerID'])}")
print(f"     Revenue: £{top_customer['TotalPrice']:,.2f}")

top_country = country_revenue.iloc[0]
print(f"   • Best Country: {top_country['Country']}")
print(f"     Revenue: £{top_country['TotalPrice']:,.2f}")

print(f"\n⏰ TEMPORAL INSIGHTS:")
busiest_month = monthly_revenue.loc[monthly_revenue['Revenue'].idxmax()]
print(f"   • Busiest Month: {busiest_month['YearMonth']}")
print(f"     Revenue: £{busiest_month['Revenue']:,.2f}")

busiest_day = dow_revenue.loc[dow_revenue['TotalPrice'].idxmax()]
print(f"   • Busiest Day: {busiest_day['DayName']}")
print(f"     Revenue: £{busiest_day['TotalPrice']:,.2f}")

busiest_hour = hourly_revenue.loc[hourly_revenue['TotalPrice'].idxmax()]
print(f"   • Peak Hour: {int(busiest_hour['Hour'])}:00")
print(f"     Revenue: £{busiest_hour['TotalPrice']:,.2f}")

print("\n" + "="*70)
print(f"✓ All visualizations saved to: {OUTPUT_DIR}/")
print(f"✓ Total graphs generated: 16")
print("="*70)

print("\n✅ Analysis complete! Check the 'analysis_output' folder for all visualizations.")
