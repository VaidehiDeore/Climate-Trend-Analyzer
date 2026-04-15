import pandas as pd
from scipy.stats import linregress


def temperature_trend(df: pd.DataFrame):
    """
    Analyze yearly temperature trend.
    """

    yearly_temp = df.groupby("year")["avg_temp"].mean().reset_index()

    slope, intercept, r_value, p_value, std_err = linregress(
        yearly_temp["year"], yearly_temp["avg_temp"]
    )

    print("\n📊 Temperature Trend Analysis")
    print("---------------------------------")
    print(f"Trend slope: {slope:.4f}")
    print(f"R-squared: {r_value**2:.4f}")

    if slope > 0:
        print("Temperature is increasing over time 🔥")
    else:
        print("Temperature is decreasing over time ❄️")

    return yearly_temp


def rainfall_trend(df: pd.DataFrame):
    """
    Analyze yearly rainfall trend.
    """

    yearly_rain = df.groupby("year")["precipitation"].sum().reset_index()

    slope, intercept, r_value, p_value, std_err = linregress(
        yearly_rain["year"], yearly_rain["precipitation"]
    )

    print("\n🌧️ Rainfall Trend Analysis")
    print("---------------------------------")
    print(f"Trend slope: {slope:.4f}")
    print(f"R-squared: {r_value**2:.4f}")

    if slope > 0:
        print("Rainfall is increasing over time 🌧️")
    else:
        print("Rainfall is decreasing over time ☀️")

    return yearly_rain