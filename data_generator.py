"""
Synthetic order data generator for a quick-commerce dark store.

Why synthetic data?
Real store data is confidential, so this script simulates realistic order
patterns based on publicly known quick-commerce benchmarks (30-minute
delivery promise, lunch/dinner demand peaks) plus operational experience
of where item-not-found (INF) defects actually cluster.

Run:  python data_generator.py   -> creates orders.csv (~120 days of orders)
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # fixed seed = same data every run (reproducible)

# ---------------------------------------------------------------------------
# Store parameters (tune these to your intuition)
# ---------------------------------------------------------------------------
DAYS = 120                      # ~4 months of history
BASE_ORDERS_PER_DAY = 250       # average daily order volume
OPEN_HOUR, CLOSE_HOUR = 7, 23   # store operating hours

# Demand multiplier per weekday (Mon=0 ... Sun=6). Weekends are busier.
WEEKDAY_FACTOR = [0.90, 0.85, 0.90, 0.95, 1.10, 1.25, 1.05]

# Relative demand per hour of day — two peaks: lunch and dinner.
HOURLY_WEIGHTS = {
    7: 2, 8: 4, 9: 5, 10: 5, 11: 7, 12: 10, 13: 10, 14: 7,
    15: 5, 16: 6, 17: 8, 18: 10, 19: 11, 20: 9, 21: 6, 22: 3,
}

CATEGORIES = [
    "Fresh Produce", "Dairy & Eggs", "Snacks", "Drinks", "Household",
    "Frozen", "Bakery", "Meat & Fish", "Health & Beauty", "Alcohol",
]
# Some categories are ordered more than others.
CATEGORY_WEIGHTS = [0.16, 0.14, 0.13, 0.13, 0.09, 0.08, 0.08, 0.08, 0.06, 0.05]

# INF probability per category — from operational experience: small-format
# items (medicine, cosmetics) and chilled lines are misplaced and
# miscounted most; large/fixed items like alcohol rarely go missing.
CATEGORY_INF = {
    "Health & Beauty": 0.030,
    "Dairy & Eggs": 0.025,
    "Frozen": 0.020,
    "Fresh Produce": 0.018,
    "Meat & Fish": 0.015,
    "Bakery": 0.010,
    "Snacks": 0.008,
    "Drinks": 0.006,
    "Household": 0.006,
    "Alcohol": 0.005,
}


def generate() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=DAYS)
    order_id = 10000

    hours = list(HOURLY_WEIGHTS.keys())
    hour_p = np.array(list(HOURLY_WEIGHTS.values()), dtype=float)
    hour_p /= hour_p.sum()

    for d in range(DAYS):
        date = start + pd.Timedelta(days=d)
        # Daily volume = base * weekday effect * random noise
        n_orders = int(
            BASE_ORDERS_PER_DAY
            * WEEKDAY_FACTOR[date.dayofweek]
            * np.random.normal(1.0, 0.08)
        )

        for _ in range(n_orders):
            order_id += 1
            hour = np.random.choice(hours, p=hour_p)
            minute = np.random.randint(0, 60)
            ts = date + pd.Timedelta(hours=int(hour), minutes=int(minute))

            items = max(1, np.random.poisson(4))  # items per order

            # UTR = under-the-roof time (order drop -> dispatch), target < 3 min.
            # Lognormal: most orders fast, occasional slow outliers (realistic).
            utr = float(np.random.lognormal(mean=0.85, sigma=0.35))
            # Larger baskets take slightly longer to pick.
            utr += items * 0.08

            # Rider wait + ride time. Peaks are slower (rider availability).
            peak = hour in (12, 13, 18, 19, 20)
            delivery = float(np.random.normal(21 if peak else 18, 4))
            delivery = max(8.0, delivery)

            click_to_door = utr + delivery

            # Category first, because INF risk depends on it.
            category = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)
            inf = np.random.random() < CATEGORY_INF[category]

            rows.append({
                "order_id": order_id,
                "timestamp": ts,
                "items": items,
                "category": category,
                "utr_minutes": round(utr, 2),
                "delivery_minutes": round(delivery, 2),
                "click_to_door_minutes": round(click_to_door, 2),
                "inf_flag": inf,
                "on_time": click_to_door <= 30,  # the 30-minute promise
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("orders.csv", index=False)
    print(f"Generated {len(df):,} orders across {DAYS} days -> orders.csv")
    print(f"Avg click-to-door: {df['click_to_door_minutes'].mean():.1f} min")
    print(f"On-time rate: {df['on_time'].mean():.1%}")
    print(f"INF rate: {df['inf_flag'].mean():.2%}")