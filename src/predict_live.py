import joblib
import pandas as pd

from src.live_data import get_live_environment_data


FEATURE_COLS = [
    "PM2.5", "PM10", "NO2", "CO", "SO2",
    "temperature_2m", "Humidity"
]


def get_possible_cause(row):
    causes = []

    if row["PM2.5"] > 120:
        causes.append("Sudden PM2.5 spike")
    if row["PM10"] > 180:
        causes.append("High particulate matter")
    if row["NO2"] > 80:
        causes.append("Elevated NO2 level")
    if row["SO2"] > 50:
        causes.append("Elevated SO2 level")
    if row["CO"] > 1000:
        causes.append("Elevated CO level")
    if row["temperature_2m"] > 38:
        causes.append("Unusual temperature rise")
    if row["Humidity"] < 25:
        causes.append("Low humidity condition")

    if not causes:
        return "Mixed environmental variation detected"

    return ", ".join(causes)


def build_alert(city, anomaly_label, severity):
    if anomaly_label == "Anomaly":
        if severity == "High":
            return f"Severe pollution anomaly detected in {city}."
        if severity == "Medium":
            return f"Moderate environmental anomaly detected in {city}."
        return f"Mild anomaly detected in {city}."
    return f"No major anomaly detected in {city}."


def predict_live(city, input_date,
                 model_path="models/isolation_forest_model.pkl",
                 scaler_path="models/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    live = get_live_environment_data(city, input_date)

    df = pd.DataFrame([live])
    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    score = model.decision_function(X_scaled)[0]

    anomaly_label = "Anomaly" if pred == -1 else "Normal"
    severity = "High" if score < -0.05 else ("Medium" if score < 0 else "Low")
    if anomaly_label == "Normal":
        severity = "Normal"

    result = {
        "city": live["city"],
        "date": live["date"],
        "anomaly_detected": "Yes" if anomaly_label == "Anomaly" else "No",
        "severity": severity,
        "alert": build_alert(live["city"], anomaly_label, severity),
        "possible_cause": get_possible_cause(live),
        "anomaly_score": float(score),
        **live
    }
    return result


if __name__ == "__main__":
    city = input("Enter city: ")
    input_date = input("Enter date (YYYY-MM-DD): ")
    try:
        result = predict_live(city, input_date)
        for k, v in result.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("Error:", e)