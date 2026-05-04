import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from compute_ratios import compute_ratios
from styles import apply_styles
apply_styles()

model_info = pd.read_csv('model_info.csv')
accuracy = model_info['accuracy'][0]
auc_roc  = model_info['auc_roc'][0]

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")

# --- Constants ---
RATIOS = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']

THRESHOLD = {
    'DSRI': 1.465, 'GMI': 1.193, 'AQI': 1.254, 'SGI': 1.607,
    'DEPI': 1.077, 'SGAI': 1.041, 'LVGI': 1.111, 'TATA': 0.031,
}

MEANING = {
    'DSRI': 'Receivables inflated vs revenue',
    'GMI':  'Margins deteriorating',
    'AQI':  'Asset quality declining',
    'SGI':  'Aggressive revenue growth',
    'DEPI': 'Depreciation slowed',
    'TATA': 'Earnings far exceed cash flow ★',
}

RAW_COLS = ['revenue', 'net_income', 'total_assets', 'total_liabilities',
            'operating_cash_flow', 'accounts_receivable',
            'gross_profit', 'ebit', 'depreciation', 'capex']

PL = dict(paper_bgcolor='#E9E4D7', plot_bgcolor='#E9E4D7', height=230,
          margin=dict(t=10, b=0, l=0, r=0), font=dict(color='#94a3b8'), showlegend=False)

# --- M-Score formula ---
def get_mscore(row):
    m = (-4.84 + 0.920*row['DSRI'] + 0.528*row['GMI'] + 0.404*row['AQI']
              + 0.892*row['SGI']  + 0.115*row['DEPI'] - 0.172*row['SGAI']
              + 4.679*row['TATA'] - 0.327*row['LVGI'])
    prob = float(np.clip(1 / (1 + np.exp(-m - 2.22)), 0, 1)) * 100
    return round(m, 3), round(prob, 1)

# --- Load and score ---
def load(uploaded_file):
    df = pd.read_csv(io.StringIO(uploaded_file.read().decode()))
    df.columns = [col.strip() for col in df.columns]

    if any(col.lower() in RAW_COLS for col in df.columns):
        df.columns = [col.lower() for col in df.columns]
        if 'company' not in df.columns or 'year' not in df.columns:
            st.error('Raw financials need "company" and "year" columns.')
            return None
        df = compute_ratios(df)
        if df.empty:
            st.error('Each company needs at least 2 years of data.')
            return None
    else:
        for col in list(df.columns):
            if col.upper() in RATIOS:
                df = df.rename(columns={col: col.upper()})
            if col.lower() == 'company':
                df = df.rename(columns={col: 'company'})
        missing = [r for r in RATIOS if r not in df.columns]
        if missing:
            st.error('Missing columns: ' + ', '.join(missing))
            return None

    rows = []
    for index, row in df.iterrows():
        m, prob = get_mscore(row)
        verdict  = '⚠ Fraud' if m > -1.78 else '✓ Legit'
        company  = str(row['company']) if 'company' in row else 'Row ' + str(index + 1)
        year     = int(row['year']) if 'year' in row and row['year'] else '—'
        rows.append({'Company': company, 'Year': year,
                     'DSRI': row['DSRI'], 'GMI': row['GMI'], 'AQI': row['AQI'], 'SGI': row['SGI'],
                     'DEPI': row['DEPI'], 'SGAI': row['SGAI'], 'LVGI': row['LVGI'], 'TATA': row['TATA'],
                     'M-Score': m, 'Fraud Risk %': prob, 'Verdict': verdict})
    return pd.DataFrame(rows)

# --- UI ---
st.title("🔍  Financial Statement Fraud Detection")
st.caption("Beneish M-Score  ·  Random Forest Classifier  ·  Academic Project 2026")

st.markdown(
    "**Model Performance** — Accuracy: **" + str(round(accuracy * 100, 1)) + "%**"
    + "   ·   AUC-ROC: **" + str(round(auc_roc * 100, 1)) + "%**"
    + "   ·   Trained on **1,036 companies** (454 fraud + 582 legit)"
)
st.divider()

uploaded = st.file_uploader("Upload a CSV — raw financials or pre-computed Beneish ratios", type='csv')
if not uploaded:
    a, b = st.columns(2)
    a.info("**Raw financials:** company · year · revenue · net_income · total_assets · total_liabilities · operating_cash_flow · accounts_receivable · gross_profit · ebit · depreciation · capex")
    b.info("**Pre-computed ratios:** Company · DSRI · GMI · AQI · SGI · DEPI · SGAI · LVGI · TATA")
    st.stop()

df = load(uploaded)
if df is None:
    st.stop()

fraud = df[df['Verdict'] == '⚠ Fraud']
legit = df[df['Verdict'] == '✓ Legit']
avg_m = df['M-Score'].mean()
status = "🔴 Likely Manipulated" if avg_m > -1.78 else ("🟡 Grey Zone" if avg_m > -2.22 else "🟢 Likely Clean")

# KPI cards
st.subheader("Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("🏢 Companies Analysed", len(df), border=True)
k2.metric("🚨 Flagged as Fraud",   len(fraud), str(round(len(fraud)/len(df)*100)) + "% of total", delta_color="inverse", border=True)
k3.metric("✅ Appear Legitimate",  len(legit), border=True)
k4.metric("📊 Avg M-Score",        str(round(avg_m, 2)), status, delta_color="off", border=True)
st.divider()

# Charts
st.subheader("Analysis Results")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.caption("Fraud Split")
    fig1 = go.Figure(go.Pie(values=[len(legit), len(fraud)], labels=['Legitimate', 'Fraudulent'],
                            hole=0.55, marker_colors=["#17C660", "#EB1D1D"], textfont_size=11))
    fig1.update_layout(**{**PL, 'showlegend': True},
                       legend=dict(orientation='h', y=-0.08, font=dict(color='#94a3b8')))
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

with c2:
    st.caption("Fraud Probability — Top 25")
    top25 = df.sort_values('Fraud Risk %', ascending=False).head(25)
    colors = ['#EB1D1D' if v == '⚠ Fraud' else '#17C660' for v in top25['Verdict']]
    fig2 = go.Figure(go.Bar(x=top25['Company'].str[:12], y=top25['Fraud Risk %'],
                            marker_color=colors, marker_line_width=0))
    fig2.update_layout(**PL, xaxis=dict(tickangle=45, tickfont=dict(size=9)),
                       yaxis=dict(range=[0, 115], title='%', gridcolor='#1e293b'))
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

with c3:
    st.caption("Ratio Trigger Frequency")
    if len(fraud) > 0:
        counts = []
        colors3 = []
        for ratio in RATIOS:
            count = int((fraud[ratio] > THRESHOLD[ratio]).sum())
            counts.append(count)
            freq = count / len(fraud)
            colors3.append('#EB1D1D' if freq > 0.6 else "#DCC023" if freq > 0.3 else "#1F4CE1")
        fig3 = go.Figure(go.Bar(x=RATIOS, y=counts, marker_color=colors3, marker_line_width=0))
        fig3.update_layout(**PL, yaxis=dict(title='# flagged', gridcolor='#1e293b'))
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("No fraud detected.")

with c4:
    st.caption("Avg Ratio — Fraud vs Legit")
    fraud_avg = [round(float(fraud[r].mean()), 3) if len(fraud) > 0 else 0 for r in RATIOS]
    legit_avg = [round(float(legit[r].mean()), 3) if len(legit) > 0 else 0 for r in RATIOS]
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name='Fraud', x=RATIOS, y=fraud_avg, marker_color='#EB1D1D'))
    fig4.add_trace(go.Bar(name='Legit', x=RATIOS, y=legit_avg, marker_color='#17C660'))
    fig4.update_layout(**{**PL, 'showlegend': True, 'barmode': 'group'},
                       legend=dict(orientation='h', y=-0.08, font=dict(color='#94a3b8')),
                       yaxis=dict(gridcolor='#1e293b'))
    st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})

# Root cause cards
if len(fraud) > 0:
    st.subheader("Root Cause of Fraud Signals")
    st.caption("Worst year per company shown")
    worst = fraud.loc[fraud.groupby('Company')['M-Score'].idxmax()].sort_values('M-Score').head(12)
    for start in range(0, len(worst), 3):
        chunk = worst.iloc[start: start + 3]
        cols = st.columns(3)
        col_i = 0
        for _, row in chunk.iterrows():
            triggered = [MEANING[r] for r in ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'TATA'] if row[r] > THRESHOLD[r]]
            if len(triggered) == 0:
                triggered.append(MEANING['TATA'])
            with cols[col_i]:
                with st.container(border=True):
                    st.markdown("**" + row['Company'] + "**")
                    st.caption("M-Score: " + str(row['M-Score']) + "  ·  Risk: " + str(row['Fraud Risk %']) + "%")
                    for signal in triggered:
                        st.markdown("- " + signal)
            col_i = col_i + 1
    st.divider()

# Results table
st.subheader("Full Results Table")
table = df[['Company', 'Year', 'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 'M-Score', 'Fraud Risk %', 'Verdict']].copy()

def colour_verdict(v):
    return 'color:#ef4444;font-weight:700' if v == '⚠ Fraud' else 'color:#10b981;font-weight:700'

def colour_mscore(v):
    if v > -1.78:   return 'color:#ef4444;font-weight:700'
    elif v > -2.22: return 'color:#f59e0b;font-weight:700'
    return 'color:#10b981'

fmt = {'Fraud Risk %': '{:.1f}%', 'M-Score': '{:.3f}',
       'DSRI': '{:.3f}', 'GMI': '{:.3f}', 'AQI': '{:.3f}', 'SGI': '{:.3f}',
       'DEPI': '{:.3f}', 'SGAI': '{:.3f}', 'LVGI': '{:.3f}', 'TATA': '{:.3f}'}

st.dataframe(table.style.map(colour_verdict, subset=['Verdict']).map(colour_mscore, subset=['M-Score']).format(fmt),
             use_container_width=True, height=420)
st.download_button("⬇ Download Results CSV", table.to_csv(index=False).encode(), "fraud_results.csv", "text/csv")
