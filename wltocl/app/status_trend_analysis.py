"""
status_trend_analysis.py

Run from project root:

    python app/status_trend_analysis.py

What it does:
- Loads cleaned_wl_data.csv
- Computes daily WL change between:
    status1Month -> status1Week
    status1Week  -> status2Days
    status2Days  -> status1Day
- Prints summary statistics
- Provides a function to estimate WL at any day between 1 and 30
  using linear interpolation between the four known points.
"""

import os
import numpy as np
import pandas as pd

# --------- Paths (relative to project root) ----------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../wltocl/app
BASE_DIR = os.path.dirname(THIS_DIR)                    # .../wltocl
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_wl_data.csv")

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

required_cols = [
    "bookingStatus",
    "status1Month",
    "status1Week",
    "status2Days",
    "status1Day",
    # "travelClass",  # optional
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Just show first few rows of the relevant columns
print("\nFirst 5 rows of important columns:")
print(df[required_cols].head())

# --------- 1. Compute per-day changes between status columns ----------

# We interpret:
#   status1Month -> WL about 30 days before
#   status1Week  -> WL about  7 days before
#   status2Days  -> WL about  2 days before
#   status1Day   -> WL about  1 day before

gap_30_7 = 30 - 7   # days between 1 month and 1 week
gap_7_2  = 7 - 2    # days between 1 week and 2 days
gap_2_1  = 2 - 1    # days between 2 days and 1 day

# Change per day (could be negative if WL improves)
df["drop_30_7_per_day"] = (df["status1Month"] - df["status1Week"]) / gap_30_7
df["drop_7_2_per_day"]  = (df["status1Week"]  - df["status2Days"]) / gap_7_2
df["drop_2_1_per_day"]  = (df["status2Days"]  - df["status1Day"]) / gap_2_1  # gap_2_1 = 1

print("\nSummary of daily changes (positive = WL number going down on average):")
print(df[["drop_30_7_per_day", "drop_7_2_per_day", "drop_2_1_per_day"]].describe())

# Means give you an "average" drop rate between those windows
mean_30_7 = df["drop_30_7_per_day"].mean()
mean_7_2  = df["drop_7_2_per_day"].mean()
mean_2_1  = df["drop_2_1_per_day"].mean()

print("\nAverage daily drop:")
print(f"  30->7 days : {mean_30_7:.3f} WL positions per day")
print(f"   7->2 days : {mean_7_2:.3f} WL positions per day")
print(f"   2->1 day  : {mean_2_1:.3f} WL positions per day")

# --------- 2. Function to estimate WL at any day via interpolation ----------

def estimate_status_for_day(row, day_before: float) -> float:
    """
    Estimate WL at an arbitrary 'day_before' (1..30) using
    linear interpolation between the four known points:

        (30, status1Month)
        ( 7, status1Week)
        ( 2, status2Days)
        ( 1, status1Day)

    If day_before >= 30 -> return status1Month
    If day_before <= 1  -> return status1Day
    """
    # Known points as (days_before, status)
    points = [
        (30.0, float(row["status1Month"])),
        (7.0,  float(row["status1Week"])),
        (2.0,  float(row["status2Days"])),
        (1.0,  float(row["status1Day"])),
    ]

    x = day_before

    # Clamp extremes
    if x >= points[0][0]:
        return points[0][1]
    if x <= points[-1][0]:
        return points[-1][1]

    # Find which segment x lies in, then do straight-line interpolation
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        # Note: x1 > x2 (30 > 7 > 2 > 1)
        if x2 <= x <= x1:
            # linear interpolation
            # t = 0 at x1, 1 at x2
            t = (x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)

    # Fallback (should not happen if 1 <= x <= 30)
    return float(row["status1Day"])

# --------- 3. Demonstrate estimation on a few rows ----------

print("\nExample interpolation for a few random rows:")
sample = df.sample(5, random_state=42)  # 5 random passengers

for idx, row in sample.iterrows():
    s30 = row["status1Month"]
    s7  = row["status1Week"]
    s2  = row["status2Days"]
    s1  = row["status1Day"]

    est_day20 = estimate_status_for_day(row, 20.0)
    est_day10 = estimate_status_for_day(row, 10.0)
    est_day3  = estimate_status_for_day(row, 3.0)

    print("\nRow index:", idx)
    print(f"  status1Month (30d): {s30}")
    print(f"  status1Week  (7d) : {s7}")
    print(f"  status2Days  (2d) : {s2}")
    print(f"  status1Day   (1d) : {s1}")
    print(f"  est @ 20 days: {est_day20:.2f}")
    print(f"  est @ 10 days: {est_day10:.2f}")
    print(f"  est @  3 days: {est_day3:.2f}")

print("\nDone. Use 'estimate_status_for_day(row, day)' in other code to get WL at any day between 1 and 30.")
