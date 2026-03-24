import requests
import pandas as pd
from datetime import datetime, date


def get_coordinates(city: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "results" not in data or not data["results"]:
        return None

    r = data["results"][0]
    return {
        "city": r["name"],
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "country": r.get("country", "")
    }


def _normalize_date_for_openmeteo(input_date: str) -> str:
    d = pd.to_datetime(input_date, errors="coerce")
    if pd.isna(d):
        raise ValueError("Invalid date format. Use YYYY-MM-DD.")

    selected = d.date()
    today = date.today()

    # Forecast/current APIs work best for today and near-future.
    # For older dates, this keeps the app from requesting unsupported ranges.
    if selected < today:
        raise ValueError("Please choose today or a future date for live forecast data.")

    return selected.isoformat()


def get_live_environment_data(city: str, input_date: str):
    selected_date = _normalize_date_for_openmeteo(input_date)

    loc = get_coordinates(city)
    if not loc:
        raise ValueError("City not found.")

    lat, lon = loc["latitude"], loc["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m",
        "start_date": selected_date,
        "end_date": selected_date,
        "timezone": "auto"
    }

    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    air_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide",
        "start_date": selected_date,
        "end_date": selected_date,
        "timezone": "auto"
    }

    weather_resp = requests.get(weather_url, params=weather_params, timeout=30)
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()

    air_resp = requests.get(air_url, params=air_params, timeout=30)
    air_resp.raise_for_status()
    air_data = air_resp.json()

    if "hourly" not in weather_data or "hourly" not in air_data:
        raise ValueError("Live data could not be fetched from Open-Meteo.")

    wdf = pd.DataFrame(weather_data["hourly"])
    adf = pd.DataFrame(air_data["hourly"])

    if wdf.empty or adf.empty:
        raise ValueError("No hourly data returned for the selected city/date.")

    result = {
        "city": loc["city"],
        "date": selected_date,
        "PM2.5": float(adf["pm2_5"].mean()),
        "PM10": float(adf["pm10"].mean()),
        "NO2": float(adf["nitrogen_dioxide"].mean()),
        "CO": float(adf["carbon_monoxide"].mean()),
        "SO2": float(adf["sulphur_dioxide"].mean()),
        "temperature_2m": float(wdf["temperature_2m"].mean()),
        "Humidity": float(wdf["relative_humidity_2m"].mean())
    }

    return result


if __name__ == "__main__":
    city = input("Enter city: ")
    input_date = input("Enter date (YYYY-MM-DD): ")
    try:
        data = get_live_environment_data(city, input_date)
        print(data)
    except Exception as e:
        print("Error:", e)