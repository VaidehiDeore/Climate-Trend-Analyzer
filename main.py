import pandas as pd

from src.data_loader import load_climate_data
from src.preprocess import preprocess_climate_data
from src.trend_analysis import temperature_trend, rainfall_trend
from src.anomaly_detection import detect_temperature_anomalies, detect_rainfall_anomalies
from src.forecasting import forecast_temperature
from src.visualize import (
    save_yearly_temperature_trend,
    save_yearly_precipitation_trend,
    save_temperature_anomaly_plot,
    save_forecast_plot,
)
from src.insights import generate_business_insights


def main():
    file_path = "data/raw/climate_data.csv"

    # Load + preprocess
    df = load_climate_data(file_path)
    clean_df = preprocess_climate_data(df)

    # Save cleaned dataset
    clean_df.to_csv("data/processed/cleaned_climate_data.csv", index=False)

    # Trend analysis
    yearly_temp = temperature_trend(clean_df)
    yearly_rain = rainfall_trend(clean_df)

    # Anomaly detection
    clean_df, temp_anomalies = detect_temperature_anomalies(clean_df)
    clean_df, rain_anomalies = detect_rainfall_anomalies(clean_df)

    # Forecasting
    historical, forecast = forecast_temperature(clean_df)

    # Save tables
    yearly_temp.to_csv("outputs/tables/yearly_temperature_trend.csv", index=False)
    yearly_rain.to_csv("outputs/tables/yearly_precipitation_trend.csv", index=False)
    temp_anomalies.to_csv("outputs/tables/temperature_anomalies.csv", index=False)
    rain_anomalies.to_csv("outputs/tables/extreme_rainfall_events.csv", index=False)
    forecast.to_csv("outputs/predictions/temperature_forecast.csv", index=False)

    # Save charts
    save_yearly_temperature_trend(
        yearly_temp,
        "outputs/charts/yearly_temperature_trend.png"
    )
    save_yearly_precipitation_trend(
        yearly_rain,
        "outputs/charts/yearly_precipitation_trend.png"
    )
    save_temperature_anomaly_plot(
        clean_df,
        "outputs/charts/temperature_anomalies.png"
    )
    save_forecast_plot(
        historical,
        forecast,
        "outputs/charts/temperature_forecast.png"
    )

    # Generate insights
    insights = generate_business_insights(yearly_temp, yearly_rain, temp_anomalies)
    insights_df = pd.DataFrame({"insight": insights})
    insights_df.to_csv("outputs/tables/business_insights.csv", index=False)

    print("\nPROJECT SUMMARY")
    print("-" * 50)
    for i, insight in enumerate(insights, start=1):
        print(f"{i}. {insight}")

    print("\nAll outputs saved successfully.")


if __name__ == "__main__":
    main()