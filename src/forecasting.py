from cProfile import label
from operator import index

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path: str = "data/hourly_demand.csv") -> pd.DataFrame:
    """
    Load the demand dataset and parse timestamps as datetime objects
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    return df

def train_test_split_time_series(df:pd.DataFrame, test_size: int = 24 *  7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into chronological train and test sets
    test_size is the number of final rows reserved for testing
    Default: 7 days of hourly data = 168 rows
    """
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test

def seasonal_naive_forecast(train: pd.DataFrame, test: pd.DataFrame, season_length: int = 24) -> pd.Series:
    """
    Forecast each test value using the value from the same hour in the previous seasonal cycle
    For hourly data with daily seasonality, season_length = 24
    """
    history = pd.concat([train["demand"], test["demand"]]).reset_index(drop=True)

    predictions = []

    test_start_idx = len(train)
    test_end_idx = len(train) + len(test)

    for i in range(test_start_idx, test_end_idx):
        predicted_value = history[i - season_length]
        predictions.append(predicted_value)

    return pd.Series(predictions, index=test.index)

def calculate_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Mean Absolute Error MAE):
    average absolute difference between actual and predicted values
    """
    return np.mean(np.abs(actual - predicted))
def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Root Mean Squared Error (RMSE):
    square errors, average them, then take the square root
    - penalizes larger errors more than MAE
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))

def plot_forecast(test: pd.DataFrame, predictions: pd.Series) -> None:
    """
    Plot actual demand vs forecasted demand on the test period
    """
    plt.figure(figsize=(14,6))
    plt.plot(test["timestamp"], test["demand"], label="Actual")
    plt.plot(test["timestamp"], predictions, label="Seasonal Naive Forecast")

    plt.title("Actual vs Forecasted Demand")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand")

    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    train, test = train_test_split_time_series(df)
    predictions = seasonal_naive_forecast(train, test, season_length = 24)

    mae = calculate_mae(test["demand"], predictions)
    rmse = calculate_rmse(test["demand"], predictions)

    print("Train size: ", len(train))
    print("Test size: ", len(test))
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    plot_forecast(test, predictions)