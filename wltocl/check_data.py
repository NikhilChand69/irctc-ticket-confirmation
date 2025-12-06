import pandas as pd

# Path to the ML-ready dataset we built
df = pd.read_csv(r"data\ml_ready_data.csv")

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nUnique days_until values (first 10):")
print(df["days_until"].unique()[:10])

print("\nBookingStatus range (min, max):")
print(df["bookingStatus"].min(), "to", df["bookingStatus"].max())

print("\nSample rows (first 5):")
print(df.head())
