import pandas as pd

def load_and_clean(path):
    data = pd.read_csv(path, parse_dates=["timestamp"])
    data = data.fillna(method="ffill")
    return data
