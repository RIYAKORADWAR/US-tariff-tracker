# 📈 US Tariff Stock Market Reaction Tracker

A research-grade data science project measuring how US tariff announcements (2018–2025) shocked global stock markets.

## Methods
- **CAPM Event Study** — Abnormal returns around each tariff announcement
- **FinBERT NLP** — Sentiment analysis on financial news headlines
- **XGBoost ML** — Predict cumulative abnormal returns from tariff features
- **SHAP** — Explain every model prediction at the feature level
- **International Contagion** — Compare US vs global market reactions

## Data Sources
| Source | What it provides |
|--------|-----------------|
| Yahoo Finance (`yfinance`) | Daily prices for 8 US sector ETFs + 6 international indices |
| NewsAPI.org | Live tariff-related news headlines (free tier: 100 req/day) |
| ProsusAI/finbert (HuggingFace) | Financial NLP sentiment model |

## Tickers Tracked
**US Sectors:** XLK (Tech), XLI (Industrials), XLB (Materials), XLY (Consumer Disc.), XLP (Consumer Staples), XLE (Energy), XLF (Financials), XLV (Healthcare)

**International:** EWJ (Japan), FXI (China), EWG (Germany), EWC (Canada), EWU (UK), VWO (Emerging Markets)

## Project Structure
```
├── app.py                          # Streamlit web app
├── US_Tariff_Tracker_v2.ipynb      # Full Google Colab notebook
├── requirements.txt                # Python dependencies
└── README.md
```

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit app
streamlit run app.py
```

## Deploying to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file as `app.py`
4. Deploy — your app gets a public URL instantly

## Google Colab

Open `US_Tariff_Tracker_v2.ipynb` in [Google Colab](https://colab.research.google.com). Run cells top to bottom. FinBERT downloads automatically on first run (~400 MB).

**Optional:** Add your NewsAPI key in Section 6 for live headlines.

## Key Findings
- Technology and Consumer Discretionary sectors had the most negative average CAR during tariff escalations
- International markets (especially China ETF - FXI) showed higher sensitivity to US tariff announcements than US sectors themselves
- Negative FinBERT sentiment scores correlate with more negative CARs
- The most predictive feature (via SHAP) is `tariff_pct` magnitude — larger tariffs → larger market shocks

## Skills Demonstrated
`Python` `Pandas` `Scikit-learn` `XGBoost` `SHAP` `NLP` `Transformers` `Plotly` `Streamlit` `Financial Econometrics` `Event Study Methodology`
