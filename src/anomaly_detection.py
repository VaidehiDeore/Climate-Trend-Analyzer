import pandas as pd


def detect_temperature_anomalies(df: pd.DataFrame, threshold: float = 2.5):
    """
    Detect anomalies using Z-score method.
    """

    df = df.copy()

    mean = df["avg_temp"].mean()
    std = df["avg_temp"].std()

    df["z_score"] = (df["avg_temp"] - mean) / std

    df["is_anomaly"] = df["z_score"].abs() > threshold

    anomalies = df[df["is_anomaly"]]

    print("\n🚨 Temperature Anomaly Detection")
    print("---------------------------------")
    print(f"Total anomalies detected: {len(anomalies)}")

    return df, anomalies


def detect_rainfall_anomalies(df: pd.DataFrame):
    """
    Detect extreme rainfall using top 1% threshold.
    """

    df = df.copy()

    threshold = df["precipitation"].quantile(0.99)

    df["extreme_rain"] = df["precipitation"] >= threshold

    extreme_days = df[df["extreme_rain"]]

    print("\n🌧️ Extreme Rainfall Detection")
    print("---------------------------------")
    print(f"Extreme rainfall days: {len(extreme_days)}")

    return df, extreme_days