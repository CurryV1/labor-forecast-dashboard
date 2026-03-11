from operator import index

import pandas as pd

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

def naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """
    Forecast using the previous observed value
    """
    history = pd.concat([train["demand"], test["demand"]]).reset_index(drop=True)

    predictions = []

    test_start_idx = len(train)
    test_end_idx = len(train) + len(test)

    for i in range(test_start_idx, test_end_idx):
        predicted_value = history[i - 1]
        predictions.append(predicted_value)

    return pd.Series(predictions, index=test.index)

def rolling_mean_forecast(
        train: pd.DataFrame,
        test: pd.DataFrame,
        window: int = 24
) -> pd.Series:
    """
    Forecast using the average of the previous window observation
    """
    history = pd.concat([train["demand"], test["demand"]]).reset_index(drop=True)

    predictions = []

    test_start_idx = len(train)
    test_end_idx = len(train) + len(test)

    for i in range(test_start_idx, test_end_idx):
        predicted_value = sum(history[-window:]) / window
        predictions.append(predicted_value)

    return pd.Series(predictions, index=test.index)

