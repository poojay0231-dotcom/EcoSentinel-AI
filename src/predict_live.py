import joblib
import pandas as pd

from src.live_data import get_live_environment_data


FEATURE_COLS = [
    "PM2.5", "PM10", "NO2", "CO", "SO2",
    "temperature_2m", "Humidity"
]


def get_possible_cause(row):
    causes = []
    mild_factors = []

    # PM2.5
    if row["PM2.5"] > 120:
        causes.append("High PM2.5")
    elif row["PM2.5"] > 50:
        mild_factors.append("slightly elevated PM2.5")

    # PM10
    if row["PM10"] > 180:
        causes.append("High PM10")
    elif row["PM10"] > 30:   # 👈 lowered threshold
        mild_factors.append("moderately elevated PM10")

    # NO2
    if row["NO2"] > 80:
        causes.append("High NO2")
    elif row["NO2"] > 20:
        mild_factors.append("slight NO2 variation")

    # CO
    if row["CO"] > 1000:
        causes.append("High CO")
    elif row["CO"] > 300:
        mild_factors.append("moderately elevated CO")

    # SO2
    if row["SO2"] > 50:
        causes.append("High SO2")
    elif row["SO2"] > 10:
        mild_factors.append("slight SO2 variation")

    # Temperature
    if row["temperature_2m"] > 35:
        mild_factors.append("warm temperature conditions")

    # Humidity
    if row["Humidity"] < 40:
        mild_factors.append("lower humidity")

    # 🔥 FINAL LOGIC

    # If strong causes
    if len(causes) >= 2:
        return "Combination of major factors: " + ", ".join(causes)

    if len(causes) == 1 and mild_factors:
        return causes[0] + " along with " + ", ".join(mild_factors[:2])

    if mild_factors:
        return "Multiple mild variations: " + ", ".join(mild_factors[:3])

    return "Subtle environmental fluctuations across parameters"


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
    recommendations = get_recommendations(live, severity)
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
        "recommendations": recommendations,
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

def get_recommendations(row, severity):
    recommendations = []

    # 🌫️ Pollution-based
    if row.get("PM2.5", 0) > 120 or row.get("PM10", 0) > 180:
        recommendations.extend([
            "Avoid outdoor activities, especially in traffic-heavy areas.",
            "Wear an N95 mask when going outside.",
            "Keep windows closed and use air purifiers indoors."
        ])

    # 🚗 Traffic pollution
    elif row.get("NO2", 0) > 80:
        recommendations.extend([
            "Avoid busy roads during peak hours.",
            "Use public transport or carpooling.",
            "Sensitive individuals should stay indoors."
        ])

    # 🌡️ Heat
    elif row.get("temperature_2m", 0) > 38:
        recommendations.extend([
            "Stay hydrated throughout the day.",
            "Avoid going out in peak afternoon hours.",
            "Wear light and breathable clothing."
        ])

    # 💧 Dry air
    elif row.get("Humidity", 0) < 25:
        recommendations.extend([
            "Use humidifiers indoors.",
            "Drink plenty of fluids.",
            "Avoid long exposure to AC environments."
        ])

    # ⚠️ If no specific issue → general recommendations
    if not recommendations:
        recommendations = [
            "Maintain a healthy indoor environment with proper ventilation.",
            "Stay updated with daily air quality reports.",
            "Adopt eco-friendly habits like reducing vehicle usage."
        ]

    # 🔥 Severity priority
    if severity == "High":
        recommendations = [
            "Avoid all outdoor activities and follow government advisories.",
            "Use masks and air purifiers indoors.",
            "Seek medical attention if experiencing breathing issues."
        ]
    elif severity == "Medium":
        recommendations.insert(0, "Limit prolonged outdoor exposure and stay informed on air quality.")

    return " | ".join(recommendations[:3])

    
