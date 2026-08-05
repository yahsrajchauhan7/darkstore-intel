"""
Demand forecasting: predict daily order volume for the next 14 days.

Model: scikit-learn LinearRegression with two kinds of features:
  1. A trend feature (day number) — captures growth/decline over time.
  2. Day-of-week dummy variables — captures the weekly pattern
     (weekends busier than Tuesdays, etc.).

Why linear regression and not something fancier (ARIMA, Prophet, LSTM)?
- The dominant signal in store demand is the weekly cycle + trend,
  which linear regression captures cleanly.
- It's interpretable: you can read the coefficients and explain
  exactly why the model predicts what it predicts.
- Simple models are the right baseline — you only add complexity
  when a baseline demonstrably underperforms.
(That reasoning is a good interview answer. Know it.)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the raw orders into one row per day with an order count."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    daily = df.groupby("date").size().reset_index(name="orders")
    daily["day_num"] = np.arange(len(daily))          # trend feature
    daily["dow"] = daily["date"].dt.dayofweek          # 0=Mon ... 6=Sun
    return daily


def _features(day_nums: np.ndarray, dows: np.ndarray) -> np.ndarray:
    """Build the feature matrix: [day_num, is_Tue, is_Wed, ... is_Sun]."""
    dow_dummies = np.zeros((len(dows), 6))
    for i, dow in enumerate(dows):
        if dow > 0:                     # Monday is the baseline category
            dow_dummies[i, dow - 1] = 1
    return np.column_stack([day_nums, dow_dummies])


def forecast_next_days(df: pd.DataFrame, horizon: int = 14):
    """
    Fit on history, predict the next `horizon` days.
    Returns (daily_history, forecast_dataframe, mae) where mae is the
    mean absolute error on the last 14 days held out for validation.
    """
    daily = build_daily(df)

    # --- Validation: train on all but the last 14 days, test on those 14.
    train, test = daily.iloc[:-14], daily.iloc[-14:]
    model = LinearRegression()
    model.fit(_features(train["day_num"].values, train["dow"].values),
              train["orders"].values)
    test_pred = model.predict(_features(test["day_num"].values,
                                        test["dow"].values))
    mae = float(np.mean(np.abs(test_pred - test["orders"].values)))

    # --- Final model: refit on ALL history, then predict the future.
    model.fit(_features(daily["day_num"].values, daily["dow"].values),
              daily["orders"].values)

    future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1),
                                 periods=horizon)
    future_day_nums = np.arange(len(daily), len(daily) + horizon)
    preds = model.predict(_features(future_day_nums,
                                    future_dates.dayofweek.values))

    forecast = pd.DataFrame({
        "date": future_dates,
        "predicted_orders": np.round(preds).astype(int),
    })
    return daily, forecast, mae
