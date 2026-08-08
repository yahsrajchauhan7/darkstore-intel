"""
Dark-Store Operations Intelligence Platform — Streamlit app.

Three tabs:
  1. Overview  — KPIs and operational charts (pandas + plotly)
  2. Forecast  — 14-day demand forecast (scikit-learn)
  3. AI Assistant — ask questions about the data in plain English (Groq LLM)

Run locally:
    python data_generator.py      (once, to create orders.csv)
    streamlit run app.py
"""



import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

import ai_assistant
import forecast as fc
from data_generator import generate

st.set_page_config(page_title="Dark-Store Intelligence",
                   page_icon="📦", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached so it doesn't reload on every interaction)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load orders from SQLite, generating the database if it doesn't exist."""
    if not os.path.exists("orders.db"):
        with sqlite3.connect("orders.db") as conn:
            generate().to_sql("orders", conn, if_exists="replace", index=False)
    with sqlite3.connect("orders.db") as conn:
        df = pd.read_sql(
            "SELECT order_id, timestamp, items, category, utr_minutes, "
            "delivery_minutes, click_to_door_minutes, inf_flag, on_time "
            "FROM orders",
            conn, parse_dates=["timestamp"],
        )
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    return df

df = load_data()

# ---------------------------------------------------------------------------
# Sidebar: date filter + info
# ---------------------------------------------------------------------------

st.sidebar.title("📦 Dark-Store Intel")
st.sidebar.caption(
    "Operations analytics for a UK quick-commerce dark store "
    "(30-minute delivery promise). Synthetic data modelled on "
    "real quick-commerce benchmarks."
)

min_d, max_d = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    mask = (df["date"] >= pd.Timestamp(date_range[0])) & (
        df["date"] <= pd.Timestamp(date_range[1])
    )
    view = df[mask]
else:
    view = df

tab_overview, tab_forecast, tab_ai = st.tabs(
    ["📊 Overview", "📈 Demand Forecast", "🤖 AI Assistant"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.subheader("Store performance")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total orders", f"{len(view):,}")
    c2.metric("Avg click-to-door", f"{view['click_to_door_minutes'].mean():.1f} min")
    c3.metric("On-time (≤30 min)", f"{view['on_time'].mean():.1%}")
    c4.metric("Avg UTR", f"{view['utr_minutes'].mean():.2f} min")
    c5.metric("INF rate", f"{view['inf_flag'].mean():.2%}")

    left, right = st.columns(2)

    with left:
        hourly = view.groupby("hour").size().reset_index(name="orders")
        st.plotly_chart(
            px.bar(hourly, x="hour", y="orders",
                   title="Orders by hour of day (find your peaks)"),
            use_container_width=True,
        )

        daily_ctd = (view.groupby("date")["click_to_door_minutes"]
                     .mean().reset_index())
        fig = px.line(daily_ctd, x="date", y="click_to_door_minutes",
                      title="Daily avg click-to-door time vs 30-min promise")
        fig.add_hline(y=30, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        by_dow = (view.groupby("weekday").size()
                  .reindex(dow_order).reset_index(name="orders"))
        st.plotly_chart(
            px.bar(by_dow, x="weekday", y="orders",
                   title="Orders by day of week"),
            use_container_width=True,
        )

        inf_cat = (view.groupby("category")["inf_flag"].mean()
                   .sort_values(ascending=False).reset_index())
        inf_cat["inf_rate_%"] = inf_cat["inf_flag"] * 100
        st.plotly_chart(
            px.bar(inf_cat, x="category", y="inf_rate_%",
                   title="Item-not-found rate by category "
                         "(where shelf accuracy slips)"),
            use_container_width=True,
        )

    # Operational insight box — shows you interpret data, not just plot it.
    peak_hour = view.groupby("hour").size().idxmax()
    slow_hour = view.groupby("hour")["click_to_door_minutes"].mean().idxmax()
    worst_cat = view.groupby("category")["inf_flag"].mean().idxmax()
    st.info(
        f"**Auto-insight:** Peak demand at **{peak_hour}:00**; deliveries are "
        f"slowest around **{slow_hour}:00** — staffing pickers ahead of that "
        f"window protects the 30-min promise. Highest INF rate: "
        f"**{worst_cat}** — prioritise cycle counts there."
    )

# ---------------------------------------------------------------------------
# Tab 2: Forecast
# ---------------------------------------------------------------------------

with tab_forecast:
    st.subheader("14-day order volume forecast")
    st.caption(
        "Linear regression on a trend feature + day-of-week dummies. "
        "Validated on the last 14 days of history (MAE shown below)."
    )

    daily, fcast, mae = fc.forecast_next_days(view, horizon=14)

    hist_plot = daily.rename(columns={"orders": "value"})[["date", "value"]]
    hist_plot["series"] = "actual"
    fut_plot = fcast.rename(columns={"predicted_orders": "value"})
    fut_plot["series"] = "forecast"
    combined = pd.concat([hist_plot.tail(45), fut_plot])

    st.plotly_chart(
        px.line(combined, x="date", y="value", color="series",
                title="Daily orders: last 45 days + 14-day forecast"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Validation MAE", f"±{mae:.0f} orders/day")
    c2.metric("Forecast peak day",
              f"{fcast.loc[fcast['predicted_orders'].idxmax(), 'date']:%a %d %b} "
              f"({fcast['predicted_orders'].max()} orders)")

    st.dataframe(fcast, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 3: AI Assistant
# ---------------------------------------------------------------------------

with tab_ai:
    st.subheader("Ask the data anything")
    st.caption(
        'e.g. "Which hour has the slowest deliveries?" · '
        '"Where should we focus cycle counts?" · '
        '"How busy are Saturdays vs Tuesdays?"'
    )

    # API key: from Streamlit secrets (deployed) or user input (local).
    # API key: from Streamlit secrets (deployed) or user input (local).
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = st.text_input("Groq API key (free at console.groq.com)",
                                type="password")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about store performance...")
    if question:
        if not api_key:
            st.warning("Add a Groq API key above to enable the assistant.")
        else:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    summary = ai_assistant.build_metrics_summary(view)
                    try:
                        answer = ai_assistant.ask(
                            api_key, question, summary, st.session_state.chat
                        )
                    except Exception as e:  # bad key, rate limit, etc.
                        answer = f"Assistant error: {e}"
                    st.markdown(answer)
            st.session_state.chat.append({"role": "user", "content": question})
            st.session_state.chat.append({"role": "assistant", "content": answer})
