# 📦 Dark-Store Operations Intelligence Platform

An analytics and AI platform for quick-commerce dark-store operations —
built from first-hand experience working in one of the UK's first
30-minute grocery delivery dark stores.

**Live demo:** _(add your Streamlit Cloud URL here)_

## What it does

| Layer | Feature | Tech |
|---|---|---|
| Analytics | KPIs & charts: hourly demand peaks, click-to-door vs the 30-min promise, item-not-found (INF) rates by category | pandas, Plotly |
| ML | 14-day daily demand forecast with holdout validation | scikit-learn |
| AI | "Ask the data anything" chat assistant grounded in computed metrics | Groq LLM (Llama 3.3 70B) |
| App | Interactive dashboard with date filtering | Streamlit |

## Why this project

Quick-commerce dark stores live or die by minutes: an order must go from
"customer taps buy" to "rider leaves the store" in under ~3 minutes to
protect a 30-minute delivery promise. Working inside one, I saw the
operational questions managers ask daily — *when are our peaks? where do
item-not-found defects cluster? how many pickers do we need on Saturday?*
— and built the tool I wished we had.

Real store data is confidential, so the dataset is **synthetic, generated
by `data_generator.py`** with parameters modelled on realistic
quick-commerce benchmarks (demand peaks at lunch/dinner, weekend uplift,
lognormal pick times, ~1.5% INF rate).

## How the AI assistant works

The assistant does **not** let the LLM guess at numbers. Instead:

1. pandas computes a verified metrics summary from the data
2. The user's question + that summary are sent to the LLM
3. The LLM is instructed to answer *only* from the supplied figures

This "grounding" pattern keeps answers factual — the pandas code is the
source of truth; the LLM is just the natural-language interface.

## Run it locally

```bash
git clone <this repo>
cd darkstore-intel
pip install -r requirements.txt
python data_generator.py     # generates orders.csv
streamlit run app.py
```

For the AI assistant, get a free API key at
[console.groq.com](https://console.groq.com) and paste it in the app
(or set `GROQ_API_KEY` in `.streamlit/secrets.toml`).

## Project structure

```
data_generator.py   # synthetic order data (120 days, ~30k orders)
forecast.py         # demand forecasting model + validation
ai_assistant.py     # LLM assistant with metric grounding
app.py              # Streamlit dashboard (3 tabs)
```

## Roadmap

- Pick-path simulation: estimate walking distance per order from a store layout grid
- Staffing recommendation: convert the demand forecast into pickers-per-shift
- Swap linear regression for gradient boosting and compare validation MAE
