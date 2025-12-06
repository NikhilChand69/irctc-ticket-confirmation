import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ml_ready_data.csv")
MODEL_OUT = os.path.join(BASE_DIR, "model", "accurate_model.json")

df = pd.read_csv(DATA_PATH)

FEATURES = ["bookingStatus", "days_until", "travelClass"]
X = df[FEATURES].values.astype(float)
y = df["label"].values.astype(float)

# Normalize
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1
X = (X - mean) / std

# Train logistic regression using gradient descent
weights = np.zeros(X.shape[1])
bias = 0
lr = 0.01
epochs = 2000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for _ in range(epochs):
    z = X.dot(weights) + bias
    p = sigmoid(z)
    error = p - y
    weights -= lr * (X.T.dot(error) / len(X))
    bias -= lr * np.mean(error)

model = {
    "feature_names": FEATURES,
    "weights": weights.tolist(),
    "bias": float(bias),
    "mean": mean.tolist(),
    "std": std.tolist()
}

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
with open(MODEL_OUT, "w") as f:
    json.dump(model, f, indent=2)

print("✅ Accurate ML model saved at:", MODEL_OUT)
