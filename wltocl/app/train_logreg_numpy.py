"""
train_logreg_numpy.py

Train a simple logistic regression using NumPy on cleaned_wl_data.csv
and save it to model/logreg_model.json.

Run from project root:
> python app/train_logreg_numpy.py
"""

import os
import json
import numpy as np
import pandas as pd

# Base paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../wltocl/app
BASE_DIR = os.path.dirname(THIS_DIR)                    # .../wltocl
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_wl_data.csv")
OUT_MODEL = os.path.join(BASE_DIR, "model", "logreg_model.json")

# Features expected by the app
FEATURES = [
    "bookingStatus",
    "status1Day",
    "status1Month",
    "status1Week",
    "status2Days",
    "travelClass",
]

# ---- 1. Load data ----
df = pd.read_csv(DATA_PATH)
X = df[FEATURES].values.astype(float)
y = df["labels"].values.astype(float)

# ---- 2. Standardize features ----
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1.0
X_std = (X - mean) / std

# ---- 3. Initialize parameters ----
n_features = X_std.shape[1]
w = np.zeros(n_features)
b = 0.0

# Hyperparameters
lr = 0.01
epochs = 2000
l2 = 1e-4

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ---- 4. Gradient descent ----
for epoch in range(epochs):
    z = X_std.dot(w) + b
    p = sigmoid(z)

    error = p - y
    grad_w = (X_std.T.dot(error) / len(X_std)) + l2 * w
    grad_b = np.mean(error)

    w -= lr * grad_w
    b -= lr * grad_b

    if (epoch + 1) % 200 == 0:
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        print(f"Epoch {epoch+1}/{epochs} loss={loss:.4f}")

print("Training complete. Saving model...")

model = {
    "feature_names": FEATURES,
    "weights": w.tolist(),
    "bias": float(b),
    "mean": mean.tolist(),
    "std": std.tolist(),
}

os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
with open(OUT_MODEL, "w") as f:
    json.dump(model, f, indent=2)

print("Saved:", OUT_MODEL)
