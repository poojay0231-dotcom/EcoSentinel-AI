import os
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def train_anomaly_model(
    input_path="data/All_cities_cleaned.csv",
    output_path="data/anomaly_results.csv",
    model_path="models/isolation_forest_model.pkl",
    scaler_path="models/scaler.pkl"
):
    df = pd.read_csv(input_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Live-friendly features
    feature_cols = [
        "PM2.5", "PM10", "NO2", "CO", "SO2",
        "temperature_2m", "Humidity"
    ]

    # Build temperature_2m if your cleaned file only has max/min
    if "temperature_2m" not in df.columns:
        if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
            df["temperature_2m"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")

    model_df = df.dropna(subset=feature_cols).copy()
    X = model_df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)

    model_df["anomaly"] = model.predict(X_scaled)
    model_df["anomaly_label"] = model_df["anomaly"].map({1: "Normal", -1: "Anomaly"})
    model_df["anomaly_score"] = model.decision_function(X_scaled)

    model_df["severity"] = model_df["anomaly_score"].apply(
        lambda x: "High" if x < -0.05 else ("Medium" if x < 0 else "Low")
    )
    model_df.loc[model_df["anomaly_label"] == "Normal", "severity"] = "Normal"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model_df.to_csv(output_path, index=False)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print("Model training completed.")
    print("Features used:", feature_cols)
    print("Final data shape used for training:", model_df.shape)
    print("\nAnomaly counts:")
    print(model_df["anomaly_label"].value_counts())
    print("\nSeverity counts:")
    print(model_df["severity"].value_counts())

    return model_df, model, scaler


if __name__ == "__main__":
    train_anomaly_model()