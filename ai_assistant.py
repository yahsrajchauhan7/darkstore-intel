"""
AI operations assistant — "talk to your data" in plain English.

How it works (simple but real LLM integration):
  1. We pre-compute a compact metrics summary of the dataset with pandas
     (daily volumes, peak hours, INF rates by category, on-time rate...).
  2. The user's question + that summary go to an LLM (Groq free tier).
  3. The LLM answers using ONLY the supplied numbers — it is instructed
     to never invent figures.

This pattern is called "grounding": instead of letting the model guess,
you feed it verified numbers computed by real code, and it handles the
natural-language part. The pandas code is the source of truth; the LLM
is just the interface.
"""

import pandas as pd

LLM_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an operations analyst assistant for a UK
quick-commerce dark store with a 30-minute delivery promise.
Answer questions using ONLY the metrics summary provided.
Rules:
- Never invent numbers. If the summary doesn't contain the answer,
  say what data would be needed.
- Be concise. Use the actual figures from the summary.
- When relevant, add one short operational insight (e.g. staffing,
  restocking) — the user works in store operations."""


def build_metrics_summary(df: pd.DataFrame) -> str:
    """Compute a compact, factual summary of the dataset with pandas."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["date"] = ts.dt.normalize()
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.day_name()

    daily = df.groupby("date").size()
    by_hour = df.groupby("hour").size()
    by_weekday = df.groupby("weekday").size()
    ctd_by_hour = df.groupby("hour")["click_to_door_minutes"].mean()
    inf_by_cat = (
        df.groupby("category")["inf_flag"].mean().sort_values(ascending=False)
    )

    lines = [
        f"Period: {df['date'].min().date()} to {df['date'].max().date()}",
        f"Total orders: {len(df):,}",
        f"Average orders/day: {daily.mean():.0f} (min {daily.min()}, max {daily.max()})",
        f"Average items/order: {df['items'].mean():.1f}",
        f"Average UTR (pick-to-dispatch): {df['utr_minutes'].mean():.2f} min",
        f"Average click-to-door: {df['click_to_door_minutes'].mean():.1f} min",
        f"On-time rate (<=30 min): {df['on_time'].mean():.1%}",
        f"Overall INF rate: {df['inf_flag'].mean():.2%}",
        "",
        "Orders by weekday: "
        + ", ".join(f"{d}: {c}" for d, c in by_weekday.items()),
        "Orders by hour: "
        + ", ".join(f"{h}:00={c}" for h, c in by_hour.items()),
        "Avg click-to-door by hour: "
        + ", ".join(f"{h}:00={m:.1f}min" for h, m in ctd_by_hour.items()),
        "INF rate by category: "
        + ", ".join(f"{c}: {r:.2%}" for c, r in inf_by_cat.items()),
        "Busiest hour: "
        f"{by_hour.idxmax()}:00 ({by_hour.max()} orders); "
        f"slowest-delivery hour: {ctd_by_hour.idxmax()}:00 "
        f"({ctd_by_hour.max():.1f} min avg)",
    ]
    return "\n".join(lines)


def ask(api_key: str, question: str, summary: str,
        history: list[dict]) -> str:
    """Send question + grounded metrics to the LLM, return its answer."""
    from groq import Groq  # imported here so the rest of the module
    client = Groq(api_key=api_key)  # works even without groq installed
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])  # keep recent turns for follow-ups
    messages.append({
        "role": "user",
        "content": f"METRICS SUMMARY:\n{summary}\n\nQUESTION: {question}",
    })
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,   # low temperature -> factual, less creative
        max_tokens=800,
    )
    return response.choices[0].message.content
