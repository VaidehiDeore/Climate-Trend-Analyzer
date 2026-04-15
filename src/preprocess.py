import pandas as pd


def preprocess_climate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess climate dataset.
    """

    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Remove rows with invalid dates
    df = df.dropna(subset=["date"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert numeric columns
    numeric_cols = ["avg_temp", "min_temp", "max_temp", "precipitation", "humidity"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing values (fill with median)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Feature Engineering
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["month_name"] = df["date"].dt.month_name()

    # Season logic
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Autumn"

    df["season"] = df["month"].apply(get_season)

    # Rolling averages (trend smoothing)
    df["temp_30d_avg"] = df["avg_temp"].rolling(window=30, min_periods=1).mean()
    df["rain_30d_avg"] = df["precipitation"].rolling(window=30, min_periods=1).mean()

    return df