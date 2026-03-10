from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pyarrow import DataType


def load_data(file_path: str = "data/hourly_demand.csv") -> pd.DataFrame:
    """
    Load the generated demand dataset.
    parse_dates converts the timestamp column from a string
    into a true datetime object so we can perform time-series operations.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    return df

def plot_demand_over_time(df: pd.DataFrame):
    """
    Plot the full demand time series
    """
    plt.figure(figsize=(14,6))

    plt.plot(df["timestamp"], df["demand"])

    plt.title("Hourly Demand Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand")

    plt.tight_layout()
    plt.show()

def plot_rolling_average(df: pd.DataFrame):
    """
    Plot the demand with a rolling average to smooth noise.
    """

    df["rolling_24h"] = df["demand"].rolling(window=24).mean()

    plt.figure(figsize=(14,6))

    plt.plot(df["timestamp"], df["demand"], alpha=0.3, label="Raw demand")
    plt.plot(df["timestamp"], df["rolling_24h"], color="red", label="24h rolling mean")

    plt.title("Demand with 24-Hour Rolling Average")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_hourly_pattern(df: pd.DataFrame):
    """
    Show average demand by hour of day
    """
    hourly_avg = df.groupby("hour")["demand"].mean()

    plt.figure(figsize=(10,5))

    hourly_avg.plot(kind="bar")

    plt.title("Average Demand by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Demand")

    plt.tight_layout()
    plt.show()

def plot_weekly_pattern(df: pd.DataFrame):
    """
    Show average demand by day of week
    """
    weekly_avg = df.groupby("day_of_week")["demand"].mean()

    plt.figure(figsize=(8,5))

    weekly_avg.plot(kind="bar")

    plt.title("Average Demand by Day of Week")
    plt.xlabel("Day of Week (0=Mon)")
    plt.ylabel("Average Demand")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    plot_demand_over_time(df)
    plot_rolling_average(df)
    plot_hourly_pattern(df)
    plot_weekly_pattern(df)

    # Test data columns in console
    # print("First rows of dataset:\n")
    # print(df.head())
    #
    # print("\nSummary statistics:\n")
    # print(df.describe())

