from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow imports from project root so src can be imported as a package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.forecasting import load_data, train_test_split_time_series
from src.models import seasonal_naive_forecast
from src.staffing import (
    calculate_required_agents,
    smooth_staffing_levels,
    build_staffing_plan,
)

st.set_page_config(
    page_title="Labor Forecasting Dashboard",
    layout="wide",
)


@st.cache_data
def prepare_dashboard_data(
    agent_capacity_per_hour: int,
    buffer_multiplier: float,
    minimum_agents: int,
    max_change_per_hour: int,
) -> pd.DataFrame:
    """
    Load demand data, generate forecast, compute staffing recommendations,
    smooth staffing levels, and return a dashboard-ready DataFrame.
    """
    df = load_data()
    train, test = train_test_split_time_series(df)

    forecast = seasonal_naive_forecast(train, test, season_length=24)

    staffing_recommendation = calculate_required_agents(
        forecasted_demand=forecast,
        agent_capacity_per_hour=agent_capacity_per_hour,
        buffer_multiplier=buffer_multiplier,
        minimum_agents=minimum_agents,
    )

    smoothed_staffing = smooth_staffing_levels(
        staffing_levels=staffing_recommendation,
        max_change_per_hour=max_change_per_hour,
    )

    staffing_df = build_staffing_plan(
        test=test,
        forecast=forecast,
        staffing_recommendation=staffing_recommendation,
        smoothed_staffing=smoothed_staffing,
    )

    return staffing_df


def plot_dashboard_chart(staffing_df: pd.DataFrame) -> None:
    """
    Render the main interactive Plotly chart.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=staffing_df["timestamp"],
            y=staffing_df["forecasted_demand"],
            mode="lines",
            name="Forecasted Demand",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=staffing_df["timestamp"],
            y=staffing_df["recommended_agents"],
            mode="lines",
            name="Recommended Agents",
            line=dict(dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=staffing_df["timestamp"],
            y=staffing_df["smoothed_agents"],
            mode="lines",
            name="Smoothed Agents",
        )
    )

    fig.update_layout(
        title="Forecasted Demand vs Staffing Recommendations",
        xaxis_title="Timestamp",
        yaxis_title="Demand / Agents",
        template="plotly_white",
        legend_title="Series",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Labor Forecasting & Staffing Dashboard")
    st.markdown(
        """
        Interactive call center workforce planning dashboard that:

        - forecasts hourly demand using a seasonal baseline model
        - converts demand into staffing recommendations
        - smooths staffing changes for more realistic schedules
        - estimates labor cost under configurable assumptions
        """
    )

    st.markdown(
        """
        ### Workforce Forecasting Simulator

        This dashboard simulates a call center workforce planning system by:

        - Forecasting hourly demand using time-series models
        - Translating forecasts into staffing requirements
        - Applying smoothing rules to produce realistic schedules

        Use the controls on the left to explore how operational assumptions impact staffing needs.
        """
    )

    st.sidebar.header("Scenario Controls")

    agent_capacity = st.sidebar.slider(
        "Agent Capacity per Hour",
        min_value=4,
        max_value=15,
        value=8,
        step=1,
    )

    buffer_multiplier = st.sidebar.slider(
        "Buffer Multiplier",
        min_value=1.00,
        max_value=1.50,
        value=1.15,
        step=0.01,
    )

    minimum_agents = st.sidebar.slider(
        "Minimum Agents",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

    max_change_per_hour = st.sidebar.slider(
        "Max Staffing Change per Hour",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
    )

    hourly_cost_per_agent = st.sidebar.slider(
        "Hourly Cost per Agent ($)",
        min_value=10,
        max_value=60,
        value=25,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### Model Assumptions

        **Agent Capacity:**  
        Calls handled per agent per hour.

        **Buffer Multiplier:**  
        Extra staffing buffer for variability.

        **Minimum Agents:**  
        Minimum operational staffing level.

        **Max Staffing Change:**  
        Limits how quickly staffing can change between hours.
        """
    )

    staffing_df = prepare_dashboard_data(
        agent_capacity_per_hour=agent_capacity,
        buffer_multiplier=buffer_multiplier,
        minimum_agents=minimum_agents,
        max_change_per_hour=max_change_per_hour,
    )

    avg_forecast = staffing_df["forecasted_demand"].mean()
    avg_agents = staffing_df["recommended_agents"].mean()
    avg_smoothed_agents = staffing_df["smoothed_agents"].mean()
    peak_smoothed_agents = staffing_df["smoothed_agents"].max()
    total_staff_hours = staffing_df["smoothed_agents"].sum()
    estimated_total_cost = total_staff_hours * hourly_cost_per_agent

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Forecasted Demand", f"{avg_forecast:.2f}")
    c2.metric("Avg Recommended Agents", f"{avg_agents:.2f}")
    c3.metric("Avg Smoothed Agents", f"{avg_smoothed_agents:.2f}")
    c4.metric("Peak Smoothed Agents", f"{peak_smoothed_agents}")
    c5.metric("Estimated Labor Cost", f"${estimated_total_cost:,.0f}")

    st.subheader("Forecast and Staffing Overview")
    plot_dashboard_chart(staffing_df)

    st.markdown(
        f"Estimated labor cost assumes **${hourly_cost_per_agent}/hour per agent** "
        f"across the displayed planning window."
    )

    st.subheader("Staffing Plan Preview")
    st.dataframe(staffing_df, use_container_width=True)

    csv_data = staffing_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download staffing plan CSV",
        data=csv_data,
        file_name="call_center_staffing_plan.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()