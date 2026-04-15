import pandas as pd


def generate_business_insights(
    yearly_temp: pd.DataFrame,
    yearly_rain: pd.DataFrame,
    temp_anomalies: pd.DataFrame,
) -> list[str]:
    """
    Generate simple business/research insights from project outputs.
    """

    insights = []

    temp_start = yearly_temp["avg_temp"].iloc[0]
    temp_end = yearly_temp["avg_temp"].iloc[-1]
    temp_change = temp_end - temp_start

    if temp_change > 0:
        insights.append(
            f"Average yearly temperature increased by {temp_change:.2f} units from the first year to the last year."
        )
    else:
        insights.append(
            f"Average yearly temperature decreased by {abs(temp_change):.2f} units from the first year to the last year."
        )

    rain_start = yearly_rain["precipitation"].iloc[0]
    rain_end = yearly_rain["precipitation"].iloc[-1]
    rain_change = rain_end - rain_start

    if rain_change > 0:
        insights.append(
            f"Total yearly precipitation increased by {rain_change:.2f} units from the first year to the last year."
        )
    else:
        insights.append(
            f"Total yearly precipitation decreased by {abs(rain_change):.2f} units from the first year to the last year."
        )

    insights.append(
        f"The project detected {len(temp_anomalies)} temperature anomaly event(s) using z-score based detection."
    )

    hottest_year = yearly_temp.loc[yearly_temp["avg_temp"].idxmax(), "year"]
    hottest_temp = yearly_temp["avg_temp"].max()
    insights.append(
        f"The hottest year in the dataset is {int(hottest_year)} with an average temperature of {hottest_temp:.2f}."
    )

    wettest_year = yearly_rain.loc[yearly_rain["precipitation"].idxmax(), "year"]
    wettest_rain = yearly_rain["precipitation"].max()
    insights.append(
        f"The wettest year in the dataset is {int(wettest_year)} with total precipitation of {wettest_rain:.2f}."
    )

    return insights