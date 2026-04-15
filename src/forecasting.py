import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def forecast_temperature(df: pd.DataFrame, periods: int = 12):
    """
    Forecast monthly temperature using Exponential Smoothing.
    """

    # Convert daily data to monthly
    monthly_temp = df.resample("ME", on="date")["avg_temp"].mean()

    # Train model
    model = ExponentialSmoothing(
        monthly_temp,
        trend="add",
        seasonal="add",
        seasonal_periods=12
    ).fit()

    # Forecast future months
    forecast = model.forecast(periods)

    forecast_df = forecast.reset_index()
    forecast_df.columns = ["date", "forecast_temp"]

    print("\n🔮 Temperature Forecasting")
    print("---------------------------------")
    print(f"Forecasted {periods} months ahead")

    return monthly_temp, forecast_df