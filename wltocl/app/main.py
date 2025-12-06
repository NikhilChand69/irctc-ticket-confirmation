from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime, date
import os
import json
import numpy as np
import traceback

# ---------------- PATHS ----------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))     # ...\wltocl\app
BASE_DIR = os.path.dirname(THIS_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "model", "accurate_model.json")

app = Flask(__name__)
app.secret_key = "change_this_for_production"

# ---------------- LOAD MODEL ----------------
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        app.logger.warning("accurate_model.json not found. Run train_accurate_model.py")
        model = None
        return

    with open(MODEL_PATH, "r") as f:
        saved = json.load(f)

    mean = np.array(saved["mean"], dtype=float)
    std = np.array(saved["std"], dtype=float)
    std[std == 0] = 1.0

    model = {
        "feature_names": saved["feature_names"],
        "weights": np.array(saved["weights"], dtype=float),
        "bias": float(saved["bias"]),
        "mean": mean,
        "std": std,
    }
    app.logger.info("Loaded accurate model with features: %s", model["feature_names"])

load_model()

# ---------------- CLASS ENCODING ----------------
CLASS_NAME_TO_CODE = {
    "SL": 0,
    "3A": 1,
    "2A": 2,
    "1A": 3,
    "CC": 4,
    "2S": 5,
}

# ---------------- HELPERS ----------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_probability(input_row: dict) -> float:
    """Return raw model probability in percent (0..100)."""
    if model is None:
        raise RuntimeError("Model not loaded")

    fnames = model["feature_names"]
    x = np.array([float(input_row[name]) for name in fnames])
    x_std = (x - model["mean"]) / model["std"]

    z = x_std.dot(model["weights"]) + model["bias"]
    p = sigmoid(z)
    return float(p * 100.0)

def risk_label(wl: float, days_until: int, travel_class_code: float) -> str:
    """
    Rule-based risk band (tune if you want).
    - Better chance for lower WL and more days remaining.
    - AC classes (1A/2A/3A) treated slightly better.
    """
    is_ac = travel_class_code in (1, 2, 3)  # 3A, 2A, 1A

    # High chance
    if wl <= 20 and days_until >= (5 if is_ac else 7):
        return "High"

    # Medium chance
    if wl <= 60 and days_until >= 2:
        return "Medium"

    # Otherwise low
    return "Low"

def calibrate_probability(raw_prob: float, label: str) -> float:
    """
    Adjust raw model probability so it matches the High/Medium/Low band
    more realistically for users.
    """
    p = max(0.0, min(raw_prob, 99.0))

    if label == "High":
        # push into 70–95 range
        p = max(p, 70.0)
        p = min(p, 95.0)
    elif label == "Medium":
        # keep in 35–70 range
        p = max(p, 35.0)
        p = min(p, 70.0)
    else:  # Low
        # keep under 35
        p = min(p, 35.0)

    return round(p, 2)

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    explanation = None

    if request.method == "POST":
        try:
            app.logger.info("FORM DATA: %s", dict(request.form))

            # 1) WL
            booking_raw = request.form.get("bookingStatus", "").strip()
            if not booking_raw:
                flash("Please enter booking WL.", "error")
                return redirect(url_for("index"))
            bookingStatus = float(booking_raw)

            # 2) Current date
            current_raw = request.form.get("currentDate", "").strip()
            if current_raw:
                try:
                    current_date = datetime.strptime(current_raw, "%Y-%m-%d").date()
                except ValueError:
                    flash("Invalid current date.", "error")
                    return redirect(url_for("index"))
            else:
                current_date = date.today()

            # 3) Journey date
            train_raw = request.form.get("trainDate", "").strip()
            if not train_raw:
                flash("Please enter journey date.", "error")
                return redirect(url_for("index"))
            try:
                train_date = datetime.strptime(train_raw, "%Y-%m-%d").date()
            except ValueError:
                flash("Invalid journey date.", "error")
                return redirect(url_for("index"))

            days_until = (train_date - current_date).days
            if days_until < 0:
                days_until = 0

            # 4) Travel class
            travel_class_name = request.form.get("travelClass", "").strip().upper()
            if not travel_class_name:
                flash("Please select a travel class.", "error")
                return redirect(url_for("index"))

            if travel_class_name not in CLASS_NAME_TO_CODE:
                flash(
                    f"Unknown travel class '{travel_class_name}'. "
                    f"Allowed: {', '.join(CLASS_NAME_TO_CODE.keys())}",
                    "error"
                )
                return redirect(url_for("index"))

            travelClass = float(CLASS_NAME_TO_CODE[travel_class_name])

            # 5) Check model
            if model is None:
                flash("Model not loaded. Run: python app\\train_accurate_model.py", "error")
                return redirect(url_for("index"))

            # 6) Build feature row
            input_row = {
                "bookingStatus": bookingStatus,
                "days_until": days_until,
                "travelClass": travelClass,
            }

            # 7) Raw ML probability
            raw_prob = predict_probability(input_row)
            raw_prob = round(max(0.0, min(raw_prob, 99.0)), 2)

            # 8) Rule-based risk band
            label = risk_label(bookingStatus, days_until, travelClass)

            # 9) Calibrated user-facing probability
            calibrated_prob = calibrate_probability(raw_prob, label)

            result = {
                "probability": calibrated_prob,     # what user sees
                "label": label,
            }

            explanation = [
                {"feature": "Booking WL", "importance": int(bookingStatus)},
                {"feature": "Days until journey", "importance": int(days_until)},
                {"feature": "Travel class", "importance": travel_class_name},
                {"feature": "Raw ML probability", "importance": f"{raw_prob}%"},
                {"feature": "Displayed (calibrated) probability", "importance": f"{calibrated_prob}%"},
            ]

        except Exception as e:
            app.logger.error("Error during prediction: %s", traceback.format_exc())
            flash(f"Error: {e}", "error")
            return redirect(url_for("index"))

    return render_template("index.html", result=result, explanation=explanation)

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
