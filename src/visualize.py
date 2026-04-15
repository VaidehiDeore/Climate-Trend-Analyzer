from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output(output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def save_yearly_temperature_trend(yearly_temp: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(yearly_temp["year"], yearly_temp["avg_temp"], marker="o", linewidth=1.5)
    plt.title("Yearly Average Temperature Trend")
    plt.xlabel("Year")
    plt.ylabel("Average Temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _prepare_output(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_yearly_precipitation_trend(yearly_precip: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(yearly_precip["year"], yearly_precip["precipitation"], marker="o", linewidth=1.5)
    plt.title("Yearly Total Precipitation Trend")
    plt.xlabel("Year")
    plt.ylabel("Total Precipitation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _prepare_output(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_temperature_anomaly_plot(df: pd.DataFrame, output_path: str) -> None:
    # Downsample to reduce memory load
    plot_df = df.iloc[::7].copy()  # weekly sampling for background line
    anomalies = df[df["is_anomaly"]].copy()

    plt.figure(figsize=(9, 4), dpi=90)
    plt.plot(plot_df["date"], plot_df["avg_temp"], linewidth=1.0, label="Average Temperature")

    if not anomalies.empty:
        plt.scatter(
            anomalies["date"],
            anomalies["avg_temp"],
            s=20,
            label="Anomalies"
        )

    plt.title("Temperature Anomalies")
    plt.xlabel("Date")
    plt.ylabel("Average Temperature")
    plt.legend()
    plt.tight_layout()
    _prepare_output(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_forecast_plot(historical_series: pd.Series, forecast_df: pd.DataFrame, output_path: str) -> None:
    # Only show recent historical window for lighter plotting
    hist = historical_series.tail(60)

    plt.figure(figsize=(9, 4), dpi=90)
    plt.plot(hist.index, hist.values, linewidth=1.5, label="Historical Monthly Temperature")
    plt.plot(forecast_df["date"], forecast_df["forecast_temp"], linewidth=1.5, label="Forecasted Temperature")
    plt.title("Monthly Temperature Forecast")
    plt.xlabel("Date")
    plt.ylabel("Average Temperature")
    plt.legend()
    plt.tight_layout()
    _prepare_output(output_path)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()