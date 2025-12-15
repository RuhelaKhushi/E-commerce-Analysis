# E-Commerce Data Analysis Project

## 📊 Project Overview

This project provides a **comprehensive analysis** of UK-based online retail transaction data using **NumPy, Pandas, and Matplotlib**. The dataset contains **541,909 transactions** from December 2010 to December 2011, covering sales of unique all-occasion gifts to customers worldwide.

### Dataset Information
- **Source**: UCI Machine Learning Repository - Online Retail Dataset
- **Time Period**: Dec 1, 2010 - Dec 9, 2011
- **Total Transactions**: 541,909
- **Unique Customers**: ~4,372
- **Unique Products**: ~4,070
- **Countries**: 38

## 🎯 Analysis Features

### Main Analysis (`ecommerce_analysis.py`)
- ✅ **Time Series Analysis**: Daily, monthly, hourly, and day-of-week revenue trends
- ✅ **Product Performance**: Top products by revenue and quantity sold
- ✅ **Customer Behavior**: Top customers, purchase frequency distribution
- ✅ **Geographic Analysis**: Revenue by country, UK vs International breakdown
- ✅ **Price & Quantity Analysis**: Distribution analysis and patterns
- ✅ **Correlation Analysis**: Heatmap of numerical features
- ✅ **Seasonal Trends**: Monthly patterns and revenue heatmaps

**Generates 16 visualizations**

### Advanced Analytics (`advanced_analysis.py`)
- ✅ **RFM Analysis**: Customer segmentation (Champions, Loyal, At Risk, Lost)
- ✅ **Cohort Analysis**: Customer retention tracking over time
- ✅ **Basket Analysis**: Products frequently bought together
- ✅ **Customer Lifetime Value**: CLV estimation and distribution

**Generates 8 advanced visualizations**

### Data Quality Report (`data_quality_report.py`)
- ✅ **Missing Values Analysis**: Identify and quantify missing data
- ✅ **Outlier Detection**: IQR-based outlier identification
- ✅ **Duplicate Detection**: Find duplicate rows and invoices
- ✅ **Data Consistency**: Validate dates, prices, quantities
- ✅ **Quality Scoring**: Overall data quality metrics

**Generates 2 quality assessment visualizations**

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd c:\Users\ayush\OneDrive\Desktop\ecommerce-khushi-project
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - pandas (data manipulation)
   - numpy (numerical computing)
   - matplotlib (visualization)
   - seaborn (enhanced visualizations)
   - scipy (statistical analysis)

### Data Setup

Ensure the dataset is located at:
```
C:\Users\ayush\Downloads\archive\data.csv
```

If your data is in a different location, update the `DATA_PATH` variable in each Python script.

## 📈 Running the Analysis

### 1. Main Analysis
Run the comprehensive analysis with 16 visualizations:

```bash
python ecommerce_analysis.py
```

**Expected Output**:
- Console output with key business metrics and insights
- 16 PNG visualization files in `analysis_output/` folder
- Processing time: ~30-60 seconds (depending on system)

**Generated Visualizations**:
1. Daily Revenue Trend
2. Monthly Revenue Trend
3. Revenue by Day of Week
4. Revenue by Hour of Day
5. Top 15 Products by Revenue
6. Top 15 Products by Quantity Sold
7. Top 20 Customers by Revenue
8. Customer Purchase Frequency Distribution
9. Top 15 Countries by Revenue
10. UK vs International Revenue (Pie Chart)
11. Unit Price Distribution
12. Quantity Distribution
13. Transaction Value Distribution
14. Correlation Heatmap
15. Monthly Transaction Count Trend
16. Revenue Heatmap (Month vs Day of Week)

### 2. Advanced Analytics
Run advanced customer and product analytics:

```bash
python advanced_analysis.py
```

**Expected Output**:
- RFM customer segmentation results
- Cohort retention analysis
- Product co-occurrence patterns
- Customer lifetime value metrics
- 8 advanced visualization files

**Generated Visualizations**:
17. RFM Customer Segments Distribution
18. Revenue Contribution by Segment
19. RFM 3D Scatter Plot
20. Cohort Retention Heatmap
21. Average Retention Rate Over Time
22. Product Co-occurrence Matrix
23. CLV Distribution
24. Purchase Frequency vs Average Purchase Value

### 3. Data Quality Report
Assess data quality and identify issues:

```bash
python data_quality_report.py
```

**Expected Output**:
- Missing values analysis
- Outlier detection results
- Duplicate identification
- Data quality score (0-100%)
- Actionable recommendations
- 2 quality assessment visualizations

**Generated Visualizations**:
25. Missing Values Analysis
26. Data Quality Metrics

## 📊 Key Insights from Analysis

### Business Metrics
- **Total Revenue**: £9.7M+ (varies based on data cleaning)
- **Average Order Value**: £400-500
- **Total Customers**: 4,372 unique customers
- **Total Products**: 4,070 unique SKUs
- **Geographic Reach**: 38 countries (91% UK, 9% International)

### Temporal Patterns
- **Peak Sales Period**: November-December 2011 (holiday season)
- **Busiest Day**: Thursday (typically)
- **Peak Hours**: 10 AM - 3 PM (business hours)
- **Seasonal Trend**: Strong Q4 performance

### Customer Insights
- **Customer Segments** (RFM):
  - Champions: High-value, frequent buyers
  - Loyal Customers: Regular purchasers
  - At Risk: Declining engagement
  - Lost: Inactive customers
- **Retention**: First-month retention varies by cohort
- **Purchase Frequency**: Most customers make 1-5 purchases

### Product Insights
- **Top Products**: Gift items, decorative pieces
- **Price Range**: £0.25 - £38,970 (wide variety)
- **Basket Analysis**: Identifies cross-selling opportunities

## 📁 Project Structure

```
ecommerce-khushi-project/
│
├── ecommerce_analysis.py          # Main analysis script (16 visualizations)
├── advanced_analysis.py           # Advanced analytics (8 visualizations)
├── data_quality_report.py         # Data quality assessment (2 visualizations)
├── requirements.txt               # Python dependencies
├── analysis_README.md             # This file
│
└── analysis_output/               # Generated visualizations (created automatically)
    ├── 01_daily_revenue_trend.png
    ├── 02_monthly_revenue_trend.png
    ├── ...
    └── 26_data_quality_metrics.png
```

## 🔍 Data Dictionary

| Column | Description | Type | Notes |
|--------|-------------|------|-------|
| InvoiceNo | Transaction identifier | String | Starts with 'C' for cancellations |
| StockCode | Product code | String | Unique product identifier |
| Description | Product name | String | ~1,454 missing values |
| Quantity | Items purchased | Integer | Negative values = returns |
| InvoiceDate | Transaction timestamp | DateTime | Dec 2010 - Dec 2011 |
| UnitPrice | Price per unit (£) | Float | Negative values = adjustments |
| CustomerID | Customer identifier | Float | ~25% missing values |
| Country | Customer location | String | 38 unique countries |

## 🛠️ Customization

### Changing Data Path
Edit the `DATA_PATH` variable in each script:
```python
DATA_PATH = r'C:\Your\Custom\Path\data.csv'
```

### Adjusting Visualizations
- **Figure Size**: Modify `figsize=(width, height)` parameters
- **Colors**: Change color codes (e.g., `'#E63946'`)
- **DPI**: Adjust `dpi=300` for higher/lower resolution
- **Output Format**: Change `.png` to `.jpg`, `.pdf`, etc.

### Filtering Data
Add custom filters after data loading:
```python
# Example: Analyze only UK transactions
df_uk = df[df['Country'] == 'United Kingdom']
```

## 📝 Data Quality Notes

### Known Issues
1. **Missing CustomerID** (~25%): Limits customer-level analysis
2. **Missing Description** (~0.3%): Some products lack descriptions
3. **Negative Values**: Returns/adjustments (need filtering)
4. **Cancelled Transactions**: InvoiceNo starting with 'C'

### Recommended Cleaning Steps
All scripts automatically:
- Remove cancelled transactions (InvoiceNo starting with 'C')
- Filter out negative quantities and prices
- Handle missing values appropriately
- Create derived features (TotalPrice, time components)

## 🎨 Visualization Examples

All visualizations are saved as high-resolution PNG files (300 DPI) in the `analysis_output/` folder. They include:
- **Line Charts**: Time series trends
- **Bar Charts**: Top performers, comparisons
- **Histograms**: Distribution analysis
- **Heatmaps**: Correlations, cohorts, patterns
- **Scatter Plots**: Customer segmentation
- **Pie Charts**: Proportional breakdowns

## 🚨 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: data.csv not found`
- **Solution**: Update `DATA_PATH` to correct file location

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: `MemoryError` when loading data
- **Solution**: Use chunking or sample the data:
  ```python
  df = pd.read_csv(DATA_PATH, nrows=100000)  # Load first 100k rows
  ```

**Issue**: Visualizations not displaying
- **Solution**: Check `analysis_output/` folder for saved PNG files

## 📚 Future Enhancements

Potential additions to the analysis:
- [ ] Machine learning models (sales forecasting, customer churn prediction)
- [ ] Interactive dashboards (Plotly, Dash)
- [ ] A/B testing framework
- [ ] Recommendation system
- [ ] Real-time data pipeline integration
- [ ] Export to Excel/PDF reports

## 📄 License

Dataset provided by UCI Machine Learning Repository.
Analysis code is open for educational and commercial use.

## 👥 Credits

- **Dataset**: Dr. Daqing Chen, London South Bank University
- **Source**: UCI Machine Learning Repository - Online Retail Dataset
- **Analysis**: Comprehensive e-commerce analytics using Python

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review console output for error messages
3. Verify data path and dependencies
4. Ensure Python 3.8+ is installed

---

**Last Updated**: December 2025
**Version**: 1.0

✅ **Ready to analyze!** Run `python ecommerce_analysis.py` to get started.
