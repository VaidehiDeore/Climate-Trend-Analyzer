import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_climate_data(
    start_date: str = "2010-01-01",
    end_date: str = "2020-12-31",
    location: str = "Pune",
    output_path: str = "data/raw/climate_data.csv",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic climate dataset with:
    - daily temperature
    - daily precipitation
    - seasonality
    - a mild long-term warming trend
    - occasional anomalies
    """
    np.random.seed(random_seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    df = pd.DataFrame({"date": dates})
    df["location"] = location
    df["station"] = "ST001"

    day_of_year = df["date"].dt.dayofyear.values
    year_offset = (df["date"].dt.year - df["date"].dt.year.min()).values

    seasonal_temp = 8 * np.sin(2 * np.pi * day_of_year / 365.25)
    warming_trend = 0.05 * year_offset
    temp_noise = np.random.normal(loc=0, scale=2.2, size=n)

    avg_temp = 24 + seasonal_temp + warming_trend + temp_noise
    min_temp = avg_temp - np.random.uniform(4, 8, size=n)
    max_temp = avg_temp + np.random.uniform(4, 8, size=n)

    monthly = df["date"].dt.month
    humidity = []
    precipitation = []

    for month in monthly:
        if month in [6, 7, 8, 9]:
            rain = max(0, np.random.gamma(shape=2.5, scale=4.0))
            hum = np.random.randint(70, 95)
        elif month in [3, 4, 5]:
            rain = max(0, np.random.gamma(shape=1.0, scale=1.0))
            hum = np.random.randint(30, 55)
        else:
            rain = max(0, np.random.gamma(shape=1.2, scale=1.2))
            hum = np.random.randint(40, 65)

        precipitation.append(rain)
        humidity.append(hum)

    df["avg_temp"] = avg_temp.round(2)
    df["min_temp"] = min_temp.round(2)
    df["max_temp"] = max_temp.round(2)
    df["precipitation"] = np.array(precipitation).round(2)
    df["humidity"] = humidity

    anomaly_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[anomaly_indices[:10], "avg_temp"] += np.random.uniform(6, 10, size=10)
    df.loc[anomaly_indices[10:], "precipitation"] += np.random.uniform(30, 70, size=10)

    missing_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[missing_indices[:5], "avg_temp"] = np.nan
    df.loc[missing_indices[5:10], "precipitation"] = np.nan
    df.loc[missing_indices[10:], "min_temp"] = np.nan

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    generated_df = generate_synthetic_climate_data()
    print("Sample climate dataset generated successfully.")
    print(generated_df.head())
    print("\nDataset shape:", generated_df.shape)