import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

OUTPUT_DIR = 'analysis_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("="*70)
print("E-COMMERCE DATA QUALITY REPORT")
print("="*70)

print("\n[1/4] Loading dataset...")
DATA_PATH = r'data.csv'

try:
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    print(f"✓ Loaded {len(df):,} transactions")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit(1)

print("\n[2/4] Analyzing missing values...")

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print("\n" + "="*70)
print("MISSING VALUES ANALYSIS")
print("="*70)
if len(missing_data) > 0:
    print("\nColumns with missing values:")
    print(missing_data.to_string(index=False))
else:
    print("\n✓ No missing values found!")

if len(missing_data) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(missing_data['Column'], missing_data['Missing_Percentage'], color='
    ax.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax.set_title('Missing Values by Column', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/25_missing_values_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 25_missing_values_analysis.png")

print("\n[3/4] Detecting outliers...")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

numerical_cols = ['Quantity', 'UnitPrice']
outlier_summary = []

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df) * 100)
    
    outlier_summary.append({
        'Column': col,
        'Outlier_Count': outlier_count,
        'Outlier_Percentage': round(outlier_percentage, 2),
        'Min_Value': df[col].min(),
        'Max_Value': df[col].max(),
        'Q1': Q1,
        'Q3': Q3
    })

outlier_df = pd.DataFrame(outlier_summary)

print("\n" + "="*70)
print("OUTLIER DETECTION (IQR Method)")
print("="*70)
print("\nOutlier Summary:")
print(outlier_df.to_string(index=False))

print("\n" + "="*70)
print("NEGATIVE VALUES CHECK")
print("="*70)
negative_quantity = len(df[df['Quantity'] < 0])
negative_price = len(df[df['UnitPrice'] < 0])
print(f"\nNegative Quantities: {negative_quantity:,} ({negative_quantity/len(df)*100:.2f}%)")
print(f"Negative Prices: {negative_price:,} ({negative_price/len(df)*100:.2f}%)")

cancelled = df[df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"Cancelled Transactions (InvoiceNo starting with 'C'): {len(cancelled):,} ({len(cancelled)/len(df)*100:.2f}%)")

print("\n[4/4] Checking for duplicates...")

duplicate_rows = df.duplicated().sum()
print("\n" + "="*70)
print("DUPLICATE DETECTION")
print("="*70)
print(f"\nDuplicate Rows: {duplicate_rows:,} ({duplicate_rows/len(df)*100:.2f}%)")

duplicate_invoices = df[df.duplicated(subset=['InvoiceNo'], keep=False)]
unique_duplicate_invoices = duplicate_invoices['InvoiceNo'].nunique()
print(f"Invoices with Multiple Entries: {unique_duplicate_invoices:,}")

print("\n" + "="*70)
print("DATA CONSISTENCY CHECKS")
print("="*70)

min_date = df['InvoiceDate'].min()
max_date = df['InvoiceDate'].max()
print(f"\nDate Range: {min_date} to {max_date}")
print(f"Time Span: {(max_date - min_date).days} days")

future_dates = df[df['InvoiceDate'] > pd.Timestamp.now()]
print(f"Future Dates: {len(future_dates):,}")

zero_prices = len(df[df['UnitPrice'] == 0])
print(f"Zero Unit Prices: {zero_prices:,} ({zero_prices/len(df)*100:.2f}%)")

zero_quantities = len(df[df['Quantity'] == 0])
print(f"Zero Quantities: {zero_quantities:,} ({zero_quantities/len(df)*100:.2f}%)")

print("\n" + "="*70)
print("DATA QUALITY SCORE")
print("="*70)

total_cells = len(df) * len(df.columns)
missing_cells = df.isnull().sum().sum()
completeness_score = ((total_cells - missing_cells) / total_cells * 100)

valid_transactions = len(df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & 
                             (~df['InvoiceNo'].astype(str).str.startswith('C'))])
validity_score = (valid_transactions / len(df) * 100)

uniqueness_score = ((len(df) - duplicate_rows) / len(df) * 100)

overall_quality = (completeness_score + validity_score + uniqueness_score) / 3

print(f"\n📊 Completeness Score: {completeness_score:.2f}%")
print(f"✓ Validity Score: {validity_score:.2f}%")
print(f"🔍 Uniqueness Score: {uniqueness_score:.2f}%")
print(f"\n⭐ Overall Data Quality Score: {overall_quality:.2f}%")

fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Completeness', 'Validity', 'Uniqueness', 'Overall Quality']
scores = [completeness_score, validity_score, uniqueness_score, overall_quality]
colors = ['

bars = ax.bar(metrics, scores, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Data Quality Metrics', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80% Threshold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/26_data_quality_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: 26_data_quality_metrics.png")

print("\n" + "="*70)
print("DATA QUALITY RECOMMENDATIONS")
print("="*70)

recommendations = []

if missing_data['Missing_Percentage'].max() > 10:
    recommendations.append("⚠️  High percentage of missing values detected. Consider imputation or removal strategies.")

if negative_quantity > 0 or negative_price > 0:
    recommendations.append("⚠️  Negative values found in Quantity/Price. These may represent returns or adjustments.")

if len(cancelled) > 0:
    recommendations.append(f"⚠️  {len(cancelled):,} cancelled transactions detected. Filter these for revenue analysis.")

if zero_prices > 0:
    recommendations.append(f"⚠️  {zero_prices:,} transactions with zero price. Investigate these records.")

if duplicate_rows > 0:
    recommendations.append(f"⚠️  {duplicate_rows:,} duplicate rows found. Consider deduplication.")

if overall_quality >= 90:
    recommendations.append("✅ Excellent data quality! Dataset is ready for analysis.")
elif overall_quality >= 75:
    recommendations.append("✅ Good data quality. Minor cleaning recommended.")
else:
    recommendations.append("⚠️  Data quality needs improvement. Significant cleaning required.")

print("\n")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

print("\n" + "="*70)
print(f"✓ Data quality report saved to: {OUTPUT_DIR}/")
print("="*70)

print("\n✅ Data quality assessment complete!")
