import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import naive_forecast, seasonal_naive_forecast, rolling_mean_forecast


def load_data(file_path: str = "data/hourly_demand.csv") -> pd.DataFrame:
    """
    Load the demand dataset and parse timestamps as datetime objects.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    return df


def train_test_split_time_series(
    df: pd.DataFrame,
    test_size: int = 24 * 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into chronological train and test sets.

    test_size is the number of final rows reserved for testing.
    Default: 7 days of hourly data = 168 rows.
    """
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test


def calculate_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Mean Absolute Error (MAE):
    average absolute difference between actual and predicted values.
    """
    return np.mean(np.abs(actual - predicted))


def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Root Mean Squared Error (RMSE):
    square errors, average them, then take the square root.
    RMSE penalizes larger errors more than MAE.
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))


def plot_forecast_comparison(
    test: pd.DataFrame,
    forecasts: dict[str, pd.Series]
) -> None:
    """
    Plot actual demand and multiple forecast series on the same chart.
    """
    plt.figure(figsize=(14, 6))

    plt.plot(test["timestamp"], test["demand"], label="Actual", linewidth=2)

    for model_name, prediction_series in forecasts.items():
        plt.plot(test["timestamp"], prediction_series, label=model_name)

    plt.title("Forecast Comparison")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_data()
    train, test = train_test_split_time_series(df)

    forecasts = {
        "Naive": naive_forecast(train, test),
        "Seasonal Naive": seasonal_naive_forecast(train, test, season_length=24),
        "Rolling Mean (24h)": rolling_mean_forecast(train, test, window=24)
    }

    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}\n")

    for model_name, prediction_series in forecasts.items():
        mae = calculate_mae(test["demand"], prediction_series)
        rmse = calculate_rmse(test["demand"], prediction_series)

        print(model_name)
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}\n")

    plot_forecast_comparison(test, forecasts)