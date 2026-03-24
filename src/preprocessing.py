import pandas as pd

def clean_environment_data(input_path, output_path):
    df = pd.read_csv(input_path, low_memory=False)

    # Drop rows missing key identifiers
    df = df.dropna(subset=["city", "date"]).copy()

    # Drop rows where all pollution fields are missing
    pollution_cols = ["PM2.5", "PM10", "NO2", "CO", "SO2", "AQI"]
    df = df.dropna(subset=pollution_cols, how="all")

    # Remove duplicates
    df = df.drop_duplicates()

    # Fix date format
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Standardize city names
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # Sort rows
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Fill numeric missing values city-wise
    numeric_cols = pollution_cols + ["temperature_2m_max", "temperature_2m_min", "Humidity"]
    for col in numeric_cols:
        df[col] = df.groupby("city")[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

    # Backward/forward fill after interpolation
    for col in numeric_cols:
        df[col] = df.groupby("city")[col].transform(lambda x: x.bfill().ffill())

    df.to_csv(output_path, index=False)

    print("Cleaning completed.")
    print("Final shape:", df.shape)
    print(df.isna().sum())

    return df


if __name__ == "__main__":
    cleaned_df = clean_environment_data(
        "data/All_cities_merged.csv",
        "data/All_cities_cleaned.csv"
    )