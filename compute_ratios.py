# compute_ratios.py
# Takes raw financial data and computes the 8 Beneish ratios.
# Called from app.py when the user uploads raw financials instead of pre-computed ratios.

import pandas as pd


def compute_ratios(dataframe):

    all_results = []

    # Loop through each company separately
    for company_name, company_rows in dataframe.groupby('company'):

        # Sort by year so we always go oldest to newest
        company_rows = company_rows.sort_values('year').reset_index(drop=True)

        # We need at least 2 years because every ratio compares this year to last year
        if len(company_rows) < 2:
            continue

        # Compare each year to the one before it
        for i in range(1, len(company_rows)):

            this_year = company_rows.iloc[i]
            last_year = company_rows.iloc[i - 1]

            # Pull out the values we need — use 1 instead of 0 to avoid divide by zero
            this_revenue = this_year['revenue']      if this_year['revenue']      else 1
            last_revenue = last_year['revenue']      if last_year['revenue']      else 1
            this_assets  = this_year['total_assets'] if this_year['total_assets'] else 1
            last_assets  = last_year['total_assets'] if last_year['total_assets'] else 1
            this_depr    = this_year['depreciation'] if this_year['depreciation'] else 1
            last_depr    = last_year['depreciation'] if last_year['depreciation'] else 1

            # SGA expenses = gross profit minus EBIT
            if pd.notna(this_year.get('ebit')):
                this_sga = (this_year['gross_profit'] - this_year['ebit']) / this_revenue
            else:
                this_sga = 0

            if pd.notna(last_year.get('ebit')):
                last_sga = (last_year['gross_profit'] - last_year['ebit']) / last_revenue
            else:
                last_sga = 0

            # Non-current assets = total assets minus current assets
            this_noncurrent = this_assets - this_year['accounts_receivable'] - (this_year.get('inventory') or 0) - (this_year.get('capex') or 0)
            last_noncurrent = last_assets - last_year['accounts_receivable'] - (last_year.get('inventory') or 0) - (last_year.get('capex') or 0)

            # --- Now calculate each ratio ---

            # DSRI: are receivables growing faster than revenue? (fake sales signal)
            DSRI = (this_year['accounts_receivable'] / this_revenue) / ((last_year['accounts_receivable'] / last_revenue) or 1)

            # GMI: are margins worse than last year? (note: last year is on top)
            GMI = (last_year['gross_profit'] / last_revenue) / ((this_year['gross_profit'] / this_revenue) or 1)

            # AQI: are assets becoming more intangible?
            AQI = (this_noncurrent / this_assets) / ((last_noncurrent / last_assets) or 1)

            # SGI: how much did revenue grow?
            SGI = this_revenue / last_revenue

            # DEPI: is depreciation slowing down? (last year on top means higher = slower)
            DEPI = (last_depr / last_assets) / ((this_depr / this_assets) or 1)

            # SGAI: are SGA expenses being suppressed?
            SGAI = this_sga / (last_sga or 1e-9)

            # LVGI: is the company taking on more debt?
            LVGI = (this_year['total_liabilities'] / this_assets) / ((last_year['total_liabilities'] / last_assets) or 1)

            # TATA: is profit far above actual cash received? (the most important one)
            net_income   = this_year['net_income']           if this_year['net_income']           else 0
            cash_flow    = this_year['operating_cash_flow']  if this_year['operating_cash_flow']  else 0
            TATA = (net_income - cash_flow) / this_assets

            # Save this row
            result = {
                'company': company_name,
                'year':    int(this_year['year']),
                'DSRI':    round(DSRI, 3),
                'GMI':     round(GMI,  3),
                'AQI':     round(AQI,  3),
                'SGI':     round(SGI,  3),
                'DEPI':    round(DEPI, 3),
                'SGAI':    round(SGAI, 3),
                'LVGI':    round(LVGI, 3),
                'TATA':    round(TATA, 4),
            }
            all_results.append(result)

    return pd.DataFrame(all_results)
