# Financial Statement Fraud Detection System

## How to Run

```
# Step 1 — install dependencies (once)
pip install -r requirements.txt

# Step 2 — train the model (once)
python train.py

# Step 3 — launch the dashboard
streamlit run app.py
```

Opens at http://localhost:8501

## Test Files (in test_data/)
| File | Type | Rows |
|---|---|---|
| bulk_100_companies.csv | Raw financials (USD) | 139 |
| bulk_500_companies.csv | Raw financials (USD) | 272 |
| pakistan_companies.csv | Raw financials (PKR) | 73 |
| beneish_ratios_500.csv | Pre-computed ratios | 588 |

## Upload Formats
**Raw financials** — app computes ratios automatically:
`company, year, revenue, net_income, total_assets, total_liabilities, operating_cash_flow, accounts_receivable, gross_profit, ebit, depreciation, capex`

**Pre-computed Beneish ratios** — direct analysis:
`Company, DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA`
