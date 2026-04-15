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

file_path = "data/raw/climate_data.csv"

df = load_climate_data(file_path)
clean_df = preprocess_climate_data(df)

yearly_temp = temperature_trend(clean_df)
yearly_rain = rainfall_trend(clean_df)

clean_df, temp_anomalies = detect_temperature_anomalies(clean_df)
clean_df, rain_anomalies = detect_rainfall_anomalies(clean_df)

historical, forecast = forecast_temperature(clean_df)

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

print("\nCharts saved successfully in outputs/charts/")
print("\nSample forecast:")
print(forecast.head())