from pathlib import Path
import numpy as np
import pandas as pd

def generate_hourly_demand_data(days: int = 90, seed: int = 42) -> pd.DataFrame:
    """
     Generate synthetic hourly demand data with realistic time-series behavior:
    - hourly seasonality
    - weekly seasonality
    - trend over time
    - random noise
    """

    # Create a random number generator so the results are reproducible.
    rng = np.random.default_rng(seed)

    # Create a sequence of hourly timestamps
    start_date = pd.Timestamp("2025-01-01")
    periods = days * 24
    timestamps = pd.date_range(start = start_date, periods=periods, freq="h")

    # Build initial Dataframe
    df = pd.DataFrame({"timestamp": timestamps})

    # Extract time-based features from the timestamps
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    # Set a baseline demand level that everything else will build on
    base_demand = 20

    # Model hourly demand patterns
    # Example logic:
    # - overnight is quiet
    # - morning picks up
    # - midday peaks
    # - evening slows down
    def get_hourly_pattern(hour: int) -> int:
        if 0 <= hour <= 5:
            return -8
        if 6 <= hour <= 9:
            return 2
        if 10 <= hour <= 13:
            return 12
        if 14 <= hour <= 17:
            return 8
        if 18 <= hour <= 21:
            return 3
        return -2

    df["hourly_seasonality"] = df["hour"].apply(get_hourly_pattern)
    # Model weekly demand patterns
    # Weekends may be lighter or heavier depending on the business
    # Here, weekdays are slightly busier
    df["weekly_seasonality"] = df["is_weekend"].apply(lambda x: -3 if x == 1 else 3)

    # Add a gradual upward trend over time
    # This simulates growing demand over the full date range
    df["trend"] = np.linspace(0, 6, periods)

    # Add random noise so the data is not perfectly predictable
    df["noise"] = rng.normal(loc=0, scale=3, size=periods)

    # Combine all components into final demand
    df["demand"] = (
        base_demand
        + df["hourly_seasonality"]
        + df["weekly_seasonality"]
        + df["trend"]
        + df["noise"]
    ).round().astype(int)

    # Prevent impossible negative or zero demand values
    df["demand"] = df["demand"].clip(lower=1)

    return df

def save_data(df: pd.DataFrame, output_path: str = "data/hourly_demand.csv") -> None:
    """
    Save the generated dataset to a csv file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    demand_df = generate_hourly_demand_data(days=90, seed=42)
    save_data(demand_df)

    print(demand_df.head())
    print("\nSummary statistics:")
    print(demand_df["demand"].describe())
    print("\nSaved dataset to data/hourly_demand.csv")




