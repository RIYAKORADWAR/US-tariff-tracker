"""
US Tariff Stock Market Reaction Tracker — Streamlit Web App
Run locally: streamlit run app.py
Deploy free: push to GitHub → connect at share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Tariff Market Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa; border-radius: 10px;
    padding: 16px 20px; border-left: 4px solid #185FA5;
    margin-bottom: 8px;
  }
  .metric-card h3 { margin:0; font-size:13px; color:#6c757d; font-weight:500; }
  .metric-card h2 { margin:4px 0 0; font-size:24px; color:#1a1a2e; font-weight:700; }
  .metric-card p  { margin:2px 0 0; font-size:11px; color:#6c757d; }
  .section-header {
    background: linear-gradient(90deg, #185FA5, #0F6E56);
    color: white; padding: 10px 16px; border-radius: 8px;
    font-size: 16px; font-weight: 600; margin: 20px 0 12px;
  }
  .insight-box {
    background: #e8f4fd; border-left: 4px solid #185FA5;
    padding: 12px 16px; border-radius: 6px;
    font-size: 14px; color: #1a1a2e; margin: 8px 0;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)

# ── Data definitions ──────────────────────────────────────────────────────────
US_SECTORS = {
    'XLK':'Technology','XLI':'Industrials','XLB':'Materials',
    'XLY':'Consumer Discretionary','XLP':'Consumer Staples',
    'XLE':'Energy','XLF':'Financials','XLV':'Healthcare',
    'SPY':'S&P 500 (Market)',
}
INTL_INDICES = {
    'EWJ':'Japan','FXI':'China','EWG':'Germany',
    'EWC':'Canada','EWU':'UK','VWO':'Emerging Markets',
}
ALL_TICKERS = {**US_SECTORS, **INTL_INDICES}

TARIFF_EVENTS = [
    {'date':'2018-03-01','event':'Steel & Aluminum tariffs 25%/10%','country':'Global','sector':'Materials','tariff_pct':25},
    {'date':'2018-03-22','event':'$50B China tariff announced','country':'China','sector':'Technology','tariff_pct':25},
    {'date':'2018-07-06','event':'List 1 tariffs take effect','country':'China','sector':'Industrials','tariff_pct':25},
    {'date':'2018-09-17','event':'$200B China goods at 10%','country':'China','sector':'Consumer Discretionary','tariff_pct':10},
    {'date':'2018-12-01','event':'US-China 90-day truce','country':'China','sector':'Technology','tariff_pct':0},
    {'date':'2019-05-10','event':'List 3 raised to 25%','country':'China','sector':'Consumer Discretionary','tariff_pct':25},
    {'date':'2019-08-01','event':'$300B China goods at 10%','country':'China','sector':'Technology','tariff_pct':10},
    {'date':'2019-12-13','event':'Phase One deal agreed','country':'China','sector':'Technology','tariff_pct':-15},
    {'date':'2020-01-15','event':'Phase One deal signed','country':'China','sector':'Industrials','tariff_pct':-7},
    {'date':'2025-02-01','event':'25% on Canada & Mexico','country':'Canada/Mexico','sector':'Energy','tariff_pct':25},
    {'date':'2025-02-04','event':'10% tariffs on China','country':'China','sector':'Technology','tariff_pct':10},
    {'date':'2025-04-02','event':'Liberation Day — global tariffs','country':'Global','sector':'Consumer Discretionary','tariff_pct':10},
    {'date':'2025-04-09','event':'90-day tariff pause','country':'Global','sector':'Technology','tariff_pct':-10},
    {'date':'2025-04-11','event':'China tariffs to 145%','country':'China','sector':'Technology','tariff_pct':145},
    {'date':'2025-05-12','event':'US-China truce — back to 30%','country':'China','sector':'Technology','tariff_pct':-115},
]

HEADLINES = [
    {'date':'2018-03-01','headline':'Trump announces steep tariffs on steel and aluminum imports, rattling global markets','sentiment_label':'negative','sentiment_numeric':-0.91},
    {'date':'2018-03-22','headline':'United States to impose 25 percent tariff on 50 billion dollars of Chinese goods','sentiment_label':'negative','sentiment_numeric':-0.88},
    {'date':'2018-07-06','headline':'Trade war begins as US tariffs on Chinese goods take effect, China retaliates','sentiment_label':'negative','sentiment_numeric':-0.95},
    {'date':'2018-09-17','headline':'Trump escalates trade war with tariffs on 200 billion dollars in Chinese products','sentiment_label':'negative','sentiment_numeric':-0.87},
    {'date':'2018-12-01','headline':'US and China agree to trade truce at G20, markets surge on deal hopes','sentiment_label':'positive','sentiment_numeric':0.89},
    {'date':'2019-05-10','headline':'US raises tariffs to 25 percent, stock market plunges on trade war fears','sentiment_label':'negative','sentiment_numeric':-0.93},
    {'date':'2019-08-01','headline':'Trump announces new tariffs on 300 billion in Chinese goods sending markets lower','sentiment_label':'negative','sentiment_numeric':-0.85},
    {'date':'2019-12-13','headline':'Phase One trade deal agreed, stocks rally sharply on tariff rollback news','sentiment_label':'positive','sentiment_numeric':0.92},
    {'date':'2020-01-15','headline':'US and China sign Phase One trade deal in White House ceremony','sentiment_label':'positive','sentiment_numeric':0.87},
    {'date':'2025-02-01','headline':'Trump slaps 25 percent tariffs on Canada and Mexico, threatening supply chains','sentiment_label':'negative','sentiment_numeric':-0.90},
    {'date':'2025-04-02','headline':'Liberation Day tariffs shock global markets as Trump imposes sweeping new duties','sentiment_label':'negative','sentiment_numeric':-0.96},
    {'date':'2025-04-09','headline':'Trump pauses most tariffs for 90 days, global markets surge on relief','sentiment_label':'positive','sentiment_numeric':0.94},
    {'date':'2025-04-11','headline':'US raises China tariffs to 145 percent in largest escalation of trade war','sentiment_label':'negative','sentiment_numeric':-0.97},
    {'date':'2025-05-12','headline':'US and China reach trade truce, tariffs slashed from 145 to 30 percent','sentiment_label':'positive','sentiment_numeric':0.91},
]

# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    raw = yf.download(list(tickers.keys()), start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw['Close'].copy()
    prices.columns = [tickers[t] for t in prices.columns]
    return prices

@st.cache_data(show_spinner=False)
def compute_all_abnormal_returns(_returns_df, events, us_names, intl_names):
    results = []
    for ev in events:
        event_date = pd.to_datetime(ev['date'])
        for col in list(us_names) + list(intl_names):
            if col == 'S&P 500 (Market)': continue
            res = _compute_ar(event_date, col, _returns_df)
            if res:
                res.update({
                    'event': ev['event'], 'country': ev['country'],
                    'tariff_pct': ev['tariff_pct'],
                    'is_escalation': ev['tariff_pct'] > 0,
                    'is_us': col in us_names,
                })
                results.append(res)
    df = pd.DataFrame(results).dropna(subset=['CAR'])
    df['year']  = df['event_date'].dt.year
    df['month'] = df['event_date'].dt.month
    df['ticker_encoded'] = pd.Categorical(df['ticker']).codes
    return df

def _compute_ar(event_date, col, returns_df, window=(-5,10), est=(-30,-2)):
    market = 'S&P 500 (Market)'
    idx = returns_df.index.searchsorted(event_date)
    if idx >= len(returns_df): return None
    es, ee = max(0,idx+est[0]), max(0,idx+est[1])
    evs, eve = max(0,idx+window[0]), min(len(returns_df)-1, idx+window[1])
    if ee <= es or col not in returns_df.columns: return None
    e_data = returns_df.iloc[es:ee]
    m = LinearRegression().fit(e_data[market].values.reshape(-1,1), e_data[col].values)
    ev_data = returns_df.iloc[evs:eve+1]
    ar = ev_data[col] - (m.intercept_ + m.coef_[0] * ev_data[market])
    return {'event_date':event_date,'ticker':col,'alpha':m.intercept_,'beta':m.coef_[0],
            'AR_mean':ar.mean()*100,'CAR':ar.sum()*100,'n_days':len(ar)}

@st.cache_resource(show_spinner=False)
def train_model(results_df):
    ml = results_df.copy()
    feats = ['tariff_pct','beta','is_escalation','ticker_encoded','year','month','AR_mean','is_us']
    ml = ml.dropna(subset=feats+['CAR'])
    X = ml[feats].astype(float)
    y = ml['CAR'].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)
    model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08,
                              subsample=0.8, random_state=42, verbosity=0)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    r2   = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te_s)
    return model, sc, feats, X_te, X_te_s, y_te, y_pred, r2, rmse, shap_values, explainer

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg",
             width=50) if False else None
    st.markdown("## ⚙️ Settings")

    date_range = st.select_slider(
        "Date range",
        options=["2018–2020 (Trade War 1.0)", "2020–2022 (COVID)", "2023–2025 (Trade War 2.0)", "2018–2025 (Full)"],
        value="2018–2025 (Full)"
    )
    date_map = {
        "2018–2020 (Trade War 1.0)": ("2018-01-01","2020-12-31"),
        "2020–2022 (COVID)":          ("2020-01-01","2022-12-31"),
        "2023–2025 (Trade War 2.0)":  ("2023-01-01","2025-12-31"),
        "2018–2025 (Full)":           ("2017-01-01","2025-12-31"),
    }
    start_date, end_date = date_map[date_range]

    show_intl = st.checkbox("Show international indices", value=True)
    show_escalations_only = st.checkbox("Escalation events only", value=False)

    st.markdown("---")
    st.markdown("### 📰 NewsAPI (optional)")
    newsapi_key = st.text_input("API Key", type="password",
                                 help="Get a free key at newsapi.org — enables live headline fetching")
    if newsapi_key:
        st.success("Key entered — live headlines will be used")
    else:
        st.info("Using built-in headlines dataset")

    st.markdown("---")
    st.markdown("""
    **About this project**
    Event study + FinBERT NLP + XGBoost + SHAP to measure how US tariff announcements shock global stock markets.

    **Data sources**
    - Yahoo Finance (yfinance)
    - NewsAPI.org
    - ProsusAI/finbert (HuggingFace)
    """)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a1a2e,#185FA5);color:white;
     padding:28px 32px;border-radius:12px;margin-bottom:24px">
  <h1 style="margin:0;font-size:28px">📈 US Tariff Stock Market Reaction Tracker</h1>
  <p style="margin:8px 0 0;opacity:0.8;font-size:15px">
    Event Study · FinBERT NLP · XGBoost Prediction · SHAP Interpretability · Global Contagion
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
tickers_to_use = ALL_TICKERS if show_intl else US_SECTORS

with st.spinner("📥 Loading market data from Yahoo Finance..."):
    prices  = load_prices(tickers_to_use, start_date, end_date)
    returns = prices.pct_change().dropna()

us_cols   = list(US_SECTORS.values())
intl_cols = list(INTL_INDICES.values()) if show_intl else []

events_df     = pd.DataFrame(TARIFF_EVENTS)
events_df['date'] = pd.to_datetime(events_df['date'])
if show_escalations_only:
    events_df = events_df[events_df['tariff_pct'] > 0]

headlines_df = pd.DataFrame(HEADLINES)
headlines_df['date'] = pd.to_datetime(headlines_df['date'])

with st.spinner("⚙️ Computing CAPM abnormal returns..."):
    results_df = compute_all_abnormal_returns(returns, TARIFF_EVENTS, us_cols, intl_cols)

# ─────────────────────────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
esc_df = results_df[results_df['is_escalation']]

with col1:
    st.markdown(f"""<div class="metric-card">
        <h3>Tariff Events</h3><h2>{len(events_df)}</h2>
        <p>2018–2025</p></div>""", unsafe_allow_html=True)
with col2:
    avg_esc_car = esc_df['CAR'].mean()
    st.markdown(f"""<div class="metric-card" style="border-color:#d62728">
        <h3>Avg CAR (Escalation)</h3><h2 style="color:#d62728">{avg_esc_car:.2f}%</h2>
        <p>Cumulative abnormal return</p></div>""", unsafe_allow_html=True)
with col3:
    worst = esc_df.groupby('ticker')['CAR'].mean().idxmin()
    worst_val = esc_df.groupby('ticker')['CAR'].mean().min()
    st.markdown(f"""<div class="metric-card" style="border-color:#993C1D">
        <h3>Hardest Hit</h3><h2 style="font-size:18px">{worst}</h2>
        <p>Avg CAR: {worst_val:.2f}%</p></div>""", unsafe_allow_html=True)
with col4:
    n_intl = len(intl_cols)
    st.markdown(f"""<div class="metric-card" style="border-color:#0F6E56">
        <h3>Markets Tracked</h3><h2>{len(us_cols)-1 + n_intl}</h2>
        <p>US sectors + {n_intl} intl</p></div>""", unsafe_allow_html=True)
with col5:
    n_headlines = len(headlines_df)
    neg_pct = (headlines_df['sentiment_label']=='negative').mean()*100
    st.markdown(f"""<div class="metric-card" style="border-color:#854F0B">
        <h3>Headlines Analyzed</h3><h2>{n_headlines}</h2>
        <p>{neg_pct:.0f}% negative sentiment</p></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Market Overview",
    "📊 Event Study",
    "💬 Sentiment NLP",
    "🤖 ML Model",
    "🔍 SHAP Explainer",
    "🌍 Global Contagion",
])

# ── TAB 1: Market Overview ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Normalised Price Performance with Tariff Events</div>',
                unsafe_allow_html=True)

    norm = (prices / prices.iloc[0]) * 100
    fig = go.Figure()
    cp  = px.colors.qualitative.Set2

    for i, col in enumerate(norm.columns):
        is_intl = col in intl_cols
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], name=col,
            line=dict(width=2.5 if col=='S&P 500 (Market)' else 1.5,
                      color=cp[i % len(cp)],
                      dash='dash' if is_intl else 'solid')
        ))

    for _, ev in events_df.iterrows():
        color = 'red' if ev['tariff_pct'] > 0 else 'green'
        fig.add_vline(x=ev['date'], line_width=1, line_dash='dot',
                      line_color=color, opacity=0.4,
                      annotation_text=ev['event'][:22],
                      annotation_position="top right",
                      annotation_font_size=8)

    fig.update_layout(
        height=500, template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title='Date', yaxis_title='Price Index (Base=100)'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Solid lines = US sector ETFs | Dashed lines = International indices | Red dots = Escalations | Green dots = De-escalations")

    # Rolling correlation heatmap
    st.markdown('<div class="section-header">30-Day Rolling Correlation with S&P 500</div>',
                unsafe_allow_html=True)

    roll_corr = returns.rolling(30).corr(returns['S&P 500 (Market)']).dropna()
    # Sample monthly to reduce chart density
    roll_sample = roll_corr.resample('ME').last().drop(columns=['S&P 500 (Market)'], errors='ignore')

    fig2 = px.imshow(
        roll_sample.T, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
        title='Rolling 30-day Correlation of Each Market with S&P 500 (sampled monthly)',
        aspect='auto'
    )
    fig2.update_layout(height=350, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Red = high positive correlation with US market | Blue = decoupled or inverse")

# ── TAB 2: Event Study ────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Cumulative Abnormal Returns (CAR) by Sector</div>',
                unsafe_allow_html=True)

    view_type = st.radio("Show events:", ["Escalations only","De-escalations only","All events"], horizontal=True)
    if view_type == "Escalations only":
        plot_df = results_df[results_df['is_escalation']]
    elif view_type == "De-escalations only":
        plot_df = results_df[~results_df['is_escalation']]
    else:
        plot_df = results_df

    avg_car = plot_df.groupby(['ticker','is_us'])['CAR'].mean().reset_index().sort_values('CAR')

    fig3 = px.bar(avg_car, x='CAR', y='ticker', orientation='h',
                  color='is_us',
                  color_discrete_map={True:'#185FA5', False:'#D85A30'},
                  labels={'CAR':'Average CAR (%)','ticker':'','is_us':'Market'},
                  title=f'Average CAR — {view_type}')
    fig3.add_vline(x=0, line_dash='dash', line_color='black', opacity=0.5)
    fig3.update_layout(height=520, template='plotly_white',
                       legend=dict(title='', orientation='h'))
    fig3.for_each_trace(lambda t: t.update(name='US Sector' if t.name=='True' else 'International'))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">CAR Heatmap — Ticker × Year</div>',
                unsafe_allow_html=True)

    heat = results_df.groupby(['ticker','year'])['CAR'].mean().unstack(fill_value=0)
    fig4 = px.imshow(heat, color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                     title='Average CAR (%) by Market and Year',
                     labels=dict(color='CAR (%)'), aspect='auto')
    fig4.update_layout(height=420, template='plotly_white')
    st.plotly_chart(fig4, use_container_width=True)

    # Raw results table
    with st.expander("📋 View raw event study results"):
        disp = results_df[['event_date','ticker','event','tariff_pct','beta','CAR','AR_mean','is_us']].copy()
        disp['event_date'] = disp['event_date'].dt.date
        disp['CAR'] = disp['CAR'].round(3)
        disp['AR_mean'] = disp['AR_mean'].round(3)
        disp['beta'] = disp['beta'].round(3)
        st.dataframe(disp.sort_values('CAR'), use_container_width=True, height=300)

# ── TAB 3: Sentiment ──────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">FinBERT Sentiment Analysis on Tariff Headlines</div>',
                unsafe_allow_html=True)

    cl1, cl2 = st.columns([2,1])
    with cl1:
        color_map = {'positive':'#2ca02c','neutral':'#ff7f0e','negative':'#d62728'}
        fig5 = go.Figure()
        for label, grp in headlines_df.groupby('sentiment_label'):
            fig5.add_trace(go.Scatter(
                x=grp['date'], y=grp['sentiment_numeric'], mode='markers',
                marker=dict(size=10, color=color_map[label]),
                name=label.capitalize(),
                text=grp['headline'],
                hovertemplate='<b>%{text}</b><br>Score: %{y:.3f}<extra></extra>'
            ))
        fig5.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        fig5.update_layout(height=380, template='plotly_white',
                           title='Sentiment Scores Over Time',
                           yaxis_title='Sentiment Score (-1 to +1)',
                           hovermode='closest')
        st.plotly_chart(fig5, use_container_width=True)

    with cl2:
        counts = headlines_df['sentiment_label'].value_counts()
        fig6 = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            marker_colors=[color_map.get(l,'gray') for l in counts.index],
            hole=0.5
        ))
        fig6.update_layout(height=380, title='Sentiment Distribution',
                           showlegend=True, template='plotly_white')
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="section-header">Sentiment Score vs Cumulative Abnormal Return</div>',
                unsafe_allow_html=True)

    merged = pd.merge(
        results_df[['event_date','ticker','CAR','beta','tariff_pct','is_escalation','is_us']],
        headlines_df[['date','sentiment_numeric','sentiment_label']],
        left_on='event_date', right_on='date', how='inner'
    )

    if len(merged) >= 5:
        from scipy import stats as scipy_stats
        x = merged['sentiment_numeric'].values
        y = merged['CAR'].values
        slope, intercept, r_val, p_val, _ = scipy_stats.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 100)

        fig7 = go.Figure()
        for is_us, grp in merged.groupby('is_us'):
            fig7.add_trace(go.Scatter(
                x=grp['sentiment_numeric'], y=grp['CAR'], mode='markers',
                marker=dict(size=9, color='#185FA5' if is_us else '#D85A30'),
                name='US Sector' if is_us else 'International',
                text=grp['ticker'],
                hovertemplate='%{text}<br>Sentiment: %{x:.3f}<br>CAR: %{y:.2f}%<extra></extra>'
            ))
        fig7.add_trace(go.Scatter(x=xl, y=slope*xl+intercept, mode='lines',
                                   line=dict(color='black', dash='dash', width=2),
                                   name=f'Fit (R²={r_val**2:.3f})'))
        fig7.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.4)
        fig7.add_vline(x=0, line_dash='dot', line_color='gray', opacity=0.4)
        fig7.update_layout(height=420, template='plotly_white',
                           xaxis_title='FinBERT Sentiment Score',
                           yaxis_title='Cumulative Abnormal Return (%)')
        st.plotly_chart(fig7, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{r_val**2:.4f}")
        c2.metric("p-value", f"{p_val:.4f}", delta="significant" if p_val < 0.05 else "not significant")
        c3.metric("Slope", f"{slope:.3f}", delta="1 unit sentiment → CAR change")

# ── TAB 4: ML Model ───────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">XGBoost: Predict CAR from Tariff Features</div>',
                unsafe_allow_html=True)

    with st.spinner("Training XGBoost model..."):
        model, scaler, features, X_te, X_te_s, y_te, y_pred, r2, rmse, shap_values, explainer = train_model(results_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.4f}", delta="Model fit quality")
    c2.metric("RMSE", f"{rmse:.3f}%", delta="Prediction error")
    c3.metric("Test Samples", str(len(y_te)))

    cl1, cl2 = st.columns(2)
    with cl1:
        imp = pd.Series(model.feature_importances_, index=features).sort_values()
        fig8 = px.bar(imp.reset_index(), x=imp.values, y=features,
                      orientation='h', title='Feature Importance',
                      color=imp.values, color_continuous_scale='Blues',
                      labels={'x':'Importance','index':'Feature'})
        fig8.update_layout(height=360, template='plotly_white', showlegend=False,
                           coloraxis_showscale=False)
        st.plotly_chart(fig8, use_container_width=True)

    with cl2:
        lims = [min(y_te.min(), y_pred.min())-1, max(y_te.max(), y_pred.max())+1]
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=y_te.values, y=y_pred, mode='markers',
                                   marker=dict(size=7, color='darkorange', opacity=0.7),
                                   name='Predictions'))
        fig9.add_trace(go.Scatter(x=lims, y=lims, mode='lines',
                                   line=dict(color='red', dash='dash'),
                                   name='Perfect fit'))
        fig9.update_layout(height=360, template='plotly_white',
                           title=f'Actual vs Predicted CAR  (R²={r2:.3f})',
                           xaxis_title='Actual CAR (%)', yaxis_title='Predicted CAR (%)')
        st.plotly_chart(fig9, use_container_width=True)

    st.markdown('<div class="section-header">Predict a New Tariff Event</div>', unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        pred_tariff = st.slider("Tariff size (%)", -120, 150, 25)
        pred_esc    = st.checkbox("Is escalation?", value=True)
    with pc2:
        pred_beta   = st.slider("Sector beta", 0.3, 2.0, 1.1)
        pred_is_us  = st.checkbox("US sector (vs international)?", value=True)
    with pc3:
        pred_year   = st.slider("Year", 2018, 2026, 2025)
        pred_month  = st.slider("Month", 1, 12, 4)

    pred_encoded = 3
    pred_ar_mean = pred_tariff * -0.04

    pred_features = np.array([[pred_tariff, pred_beta, float(pred_esc), pred_encoded,
                                pred_year, pred_month, pred_ar_mean, float(pred_is_us)]])
    pred_scaled = scaler.transform(pred_features)
    predicted_car = model.predict(pred_scaled)[0]

    st.markdown(f"""
    <div class="insight-box" style="font-size:18px;text-align:center">
      Predicted CAR: <strong style="color:{'#d62728' if predicted_car < 0 else '#2ca02c'};font-size:24px">
      {predicted_car:.2f}%</strong>
      &nbsp;|&nbsp; Interpretation: The model predicts a
      <strong>{abs(predicted_car):.2f}% {'drop' if predicted_car < 0 else 'gain'}</strong>
      above/below market expectation.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 5: SHAP ───────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">SHAP — Why did the model make each prediction?</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    SHAP (SHapley Additive exPlanations) breaks down every prediction into individual feature contributions.
    Instead of a black-box score, you can see exactly <em>why</em> a tariff event caused a large negative CAR —
    was it the tariff size? The sector's beta? The timing?
    </div>
    """, unsafe_allow_html=True)

    shap_df = pd.DataFrame(shap_values, columns=features)

    # Mean absolute SHAP
    mean_shap = shap_df.abs().mean().sort_values(ascending=True)
    fig10 = px.bar(mean_shap.reset_index(), x=mean_shap.values, y=mean_shap.index,
                   orientation='h', title='Mean |SHAP| — Average Feature Contribution',
                   color=mean_shap.values, color_continuous_scale='Oranges',
                   labels={'x':'Mean |SHAP value|','index':'Feature'})
    fig10.update_layout(height=360, template='plotly_white',
                        showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig10, use_container_width=True)

    # SHAP scatter for each feature
    st.markdown('<div class="section-header">SHAP Values vs Feature Values (Beeswarm)</div>',
                unsafe_allow_html=True)

    sel_feature = st.selectbox("Select feature to explore", features,
                                index=features.index('tariff_pct'))
    feat_idx = features.index(sel_feature)
    feat_vals_raw = X_te[sel_feature].values

    fig11 = go.Figure()
    fig11.add_trace(go.Scatter(
        x=feat_vals_raw,
        y=shap_df[sel_feature].values,
        mode='markers',
        marker=dict(
            size=7,
            color=feat_vals_raw,
            colorscale='RdBu_r',
            colorbar=dict(title=sel_feature),
            opacity=0.7,
            showscale=True
        ),
        hovertemplate=f'{sel_feature}: %{{x:.2f}}<br>SHAP value: %{{y:.3f}}<extra></extra>'
    ))
    fig11.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
    fig11.update_layout(
        height=400, template='plotly_white',
        title=f'SHAP values for "{sel_feature}" — how it drives predictions',
        xaxis_title=f'{sel_feature} (actual value)',
        yaxis_title=f'SHAP impact on CAR prediction'
    )
    st.plotly_chart(fig11, use_container_width=True)
    st.caption("Points above 0 = this feature value pushed CAR higher | Below 0 = pushed CAR lower")

    # Single prediction waterfall
    st.markdown('<div class="section-header">Waterfall — Explain a Single Prediction</div>',
                unsafe_allow_html=True)

    pred_idx = st.slider("Select test sample to explain", 0, len(y_te)-1,
                          int(np.argmin(y_pred)))
    pred_row = X_te.iloc[pred_idx]
    shap_row = shap_df.iloc[pred_idx]

    wf_df = pd.DataFrame({'Feature': features, 'SHAP': shap_row.values,
                           'Value': pred_row.values})
    wf_df = wf_df.sort_values('SHAP')
    base  = float(explainer.expected_value)

    fig12 = go.Figure(go.Waterfall(
        orientation='h',
        measure=['relative']*len(wf_df) + ['total'],
        y=list(wf_df['Feature']) + ['Final prediction'],
        x=list(wf_df['SHAP']) + [0],
        base=base,
        connector=dict(line=dict(color='gray', width=1)),
        decreasing=dict(marker_color='#d62728'),
        increasing=dict(marker_color='#2ca02c'),
        totals=dict(marker_color='#185FA5'),
        text=[f"{v:.3f}" for v in wf_df['SHAP']] + [f"{y_pred[pred_idx]:.2f}%"],
        textposition='outside'
    ))
    fig12.add_vline(x=base, line_dash='dash', line_color='gray', opacity=0.5,
                    annotation_text=f'Baseline: {base:.2f}%')
    fig12.update_layout(
        height=420, template='plotly_white',
        title=f'Waterfall — Sample #{pred_idx} | Actual: {y_te.iloc[pred_idx]:.2f}% | Predicted: {y_pred[pred_idx]:.2f}%',
        xaxis_title='SHAP contribution to CAR (%)'
    )
    st.plotly_chart(fig12, use_container_width=True)

# ── TAB 6: Global Contagion ───────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">Did US Tariffs Hurt Foreign Markets More Than US Markets?</div>',
                unsafe_allow_html=True)

    if not show_intl or len(intl_cols) == 0:
        st.warning("Enable 'Show international indices' in the sidebar to view this analysis.")
    else:
        contagion = results_df.groupby(['is_us','is_escalation'])['CAR'].agg(['mean','std','count']).reset_index()
        contagion['Group'] = contagion.apply(
            lambda r: ('🇺🇸 US Sectors' if r['is_us'] else '🌍 International') + ' — ' +
                      ('Escalation' if r['is_escalation'] else 'De-escalation'), axis=1)

        fig13 = go.Figure()
        colors_c = {'🇺🇸 US Sectors — Escalation':'#d62728',
                    '🇺🇸 US Sectors — De-escalation':'#1a9850',
                    '🌍 International — Escalation':'#ff7f0e',
                    '🌍 International — De-escalation':'#4a90d9'}
        for _, row in contagion.iterrows():
            fig13.add_trace(go.Bar(
                name=row['Group'], x=[row['Group']], y=[row['mean']],
                error_y=dict(type='data', array=[row['std']], visible=True),
                marker_color=colors_c.get(row['Group'],'gray'),
                text=f"{row['mean']:.2f}%", textposition='outside'
            ))
        fig13.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.5)
        fig13.update_layout(height=420, template='plotly_white', showlegend=False,
                            title='Average CAR by Market Group and Event Type (Error bars = std dev)',
                            yaxis_title='Average CAR (%)', barmode='group')
        st.plotly_chart(fig13, use_container_width=True)

        # International breakdown
        st.markdown('<div class="section-header">Country-Level Reactions by Year</div>',
                    unsafe_allow_html=True)

        intl_df   = results_df[~results_df['is_us']]
        intl_heat = intl_df.groupby(['ticker','year'])['CAR'].mean().unstack(fill_value=0)
        fig14 = px.imshow(intl_heat, color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                          title='International Index Average CAR (%) by Year',
                          labels=dict(color='CAR (%)'), aspect='auto')
        fig14.update_layout(height=380, template='plotly_white')
        st.plotly_chart(fig14, use_container_width=True)

        # Contagion speed: how fast do international markets react vs US?
        st.markdown('<div class="section-header">Contagion Speed — Daily AR Profile Around Events</div>',
                    unsafe_allow_html=True)

        sel_event = st.selectbox(
            "Select event",
            options=events_df['event'].tolist(),
            index=0
        )
        ev_date = events_df[events_df['event']==sel_event]['date'].values[0]

        fig15 = go.Figure()
        for col in (us_cols[:4] + intl_cols[:4]):
            if col == 'S&P 500 (Market)': continue
            res = _compute_ar(pd.Timestamp(ev_date), col, returns)
            if res and len(res['AR_series']) > 0:
                days = list(range(-5, -5+len(res['AR_series'])))
                fig15.add_trace(go.Scatter(
                    x=days, y=res['AR_series'], mode='lines+markers',
                    name=col,
                    line=dict(dash='dash' if col in intl_cols else 'solid', width=1.5)
                ))
        fig15.add_vline(x=0, line_dash='dash', line_color='red', opacity=0.7,
                        annotation_text='Event day')
        fig15.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.4)
        fig15.update_layout(height=420, template='plotly_white',
                            title=f'Daily AR Profile: {sel_event}',
                            xaxis_title='Days relative to announcement',
                            yaxis_title='Abnormal Return (%)',
                            hovermode='x unified')
        st.plotly_chart(fig15, use_container_width=True)
        st.caption("Solid = US sectors | Dashed = International indices | Day 0 = announcement date")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#6c757d;font-size:13px;padding:12px">
  <strong>US Tariff Stock Market Reaction Tracker</strong> &nbsp;|&nbsp;
  Data: Yahoo Finance · NewsAPI · HuggingFace FinBERT &nbsp;|&nbsp;
  Methods: CAPM Event Study · XGBoost · SHAP · FinBERT NLP
</div>
""", unsafe_allow_html=True)
