import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_DATA = os.path.join(BASE_DIR, "data", "cleaned_wl_data.csv")
NEW_DATA = os.path.join(BASE_DIR, "data", "ml_ready_data.csv")

df = pd.read_csv(OLD_DATA)

rows = []

for _, row in df.iterrows():
    travel_class = row["travelClass"]
    label = row["labels"]

    # Create multiple realistic time snapshots
    rows.append([row["status1Month"], 30, travel_class, label])
    rows.append([row["status1Week"],   7,  travel_class, label])
    rows.append([row["status2Days"],   2,  travel_class, label])
    rows.append([row["status1Day"],    1,  travel_class, label])

ml_df = pd.DataFrame(rows, columns=[
    "bookingStatus", "days_until", "travelClass", "label"
])

ml_df.to_csv(NEW_DATA, index=False)

print("✅ Rebuilt ML dataset with time variation:")
print(NEW_DATA)
print("Total rows:", len(ml_df))
