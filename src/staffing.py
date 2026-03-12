import math
from cProfile import label

import pandas as pd
import matplotlib.pyplot as plt

from models import seasonal_naive_forecast
from forecasting import load_data, train_test_split_time_series

def calculate_required_agents(
        forecasted_demand: pd.Series,
        agent_capacity_per_hour: int = 8,
        buffer_multiplier: float = 1.15,
        minimum_agents: int = 1

) -> pd.Series:
    """
    Convert forecasted hourly demand into recommended staffing levels
    Parameters:
    - forecasted_demand: predicted demand for each hour
    - agent_capacity_per_hour: how many contacts one agent can handle per hour
    - buffer_multiplier: extra staffing cushion to protect service levels
    - minimum_agents: minimum number of agents required each hour
    """
    staffing_levels = []

    for demand in forecasted_demand:
        required_agents = math.ceil((demand / agent_capacity_per_hour) * buffer_multiplier)
        required_agents = max(required_agents, minimum_agents)
        staffing_levels.append(required_agents)

    return pd.Series(staffing_levels, index=forecasted_demand.index)

def smooth_staffing_levels(
        staffing_levels: pd.Series,
        max_change_per_hour: int = 1
) -> pd.Series:
    """
    Smooth staffing recommendations so they do not change too sharply
    between consecutive hours.

    Parameters:
     - staffing_levels: raw staffing recommendations
     - max_change_per_hour: maximum allowed increase or decrease between hours
    """

    smoothed = [staffing_levels.iloc[0]]

    for i in range(1, len(staffing_levels)):
        prev = smoothed[-1]
        target = staffing_levels.iloc[i]

        if target > prev:
            adjusted = min(prev + max_change_per_hour, target)
        elif target < prev:
            adjusted = max(prev - max_change_per_hour, target)
        else:
            adjusted = prev

        smoothed.append(adjusted)

    return pd.Series(smoothed, index=staffing_levels.index)

def build_staffing_plan(
        test: pd.DataFrame,
        forecast: pd.Series,
        staffing_recommendation: pd.Series,
        smoothed_staffing: pd.Series
) -> pd.DataFrame:
    """
        Combine timestamps, actual demand, forecasted demand,
        and staffing recommendations into one table.
        """
    staffing_df = pd.DataFrame({
        "timestamp": test["timestamp"],
        "actual_demand": test["demand"],
        "forecasted_demand": forecast,
        "recommended_agents": staffing_recommendation,
        "smoothed_agents": smoothed_staffing
    })

    staffing_df["hour"] = staffing_df["timestamp"].dt.hour
    staffing_df["day_of_week"] = staffing_df["timestamp"].dt.dayofweek

    return staffing_df

def plot_staffing_plan(staffing_df: pd.DataFrame) -> None:
    """
    Plot forecasted demand and recommended staffing over time
    """
    plt.figure(figsize=(14,6))

    plt.plot(staffing_df["timestamp"], staffing_df["forecasted_demand"], label="Forecasted Demand")
    plt.plot(staffing_df["timestamp"], staffing_df["recommended_agents"], label="Recommended Agents", linestyle="--")
    plt.plot(staffing_df["timestamp"], staffing_df["smoothed_agents"], label="Smoothed Agents", linewidth=2)

    plt.title("Forecasted Demand and Recommended Staffing")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand / Agents")

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    train, test = train_test_split_time_series(df)
    forecast = seasonal_naive_forecast(train,test,season_length=24)

    staffing_recommendation = calculate_required_agents(
        forecasted_demand=forecast,
        agent_capacity_per_hour=8,
        buffer_multiplier=1.15,
        minimum_agents=2
        )

    smoothed_staffing = smooth_staffing_levels(
        staffing_levels=staffing_recommendation,
        max_change_per_hour=1
    )


    staffing_df = build_staffing_plan(test, forecast, staffing_recommendation, smoothed_staffing=smoothed_staffing)

    print(staffing_df.head(10))
    print("\nSummary of recommended staffing: ")
    print(staffing_df["recommended_agents"].describe())

    print("\nSummary of smoothed staffing:")
    print(staffing_df["smoothed_agents"].describe())


    plot_staffing_plan(staffing_df)

