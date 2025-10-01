"""
Comprehensive Streamlit Dashboard for HH2025TB
=============================================

This Streamlit application provides a wide range of interactive visualisations
based on the full 2025 Household Travel Survey dataset for Tbilisi.  It
supersedes the simpler dashboard delivered previously and includes many more
analytical views.  The app is organised into thematic sections (selected
via a drop‑down in the sidebar) covering socio‑demographics, mode choice,
trip characteristics, mobility behaviour and special charts.  Users can
filter the dataset by trip purpose, age category and sex.  All figures
update reactively in response to the filters and the selected section.

The following charts are included:

* **Socio‑Demographics & Household**
  - *Household size distribution*: bar chart of how many people live in each
    household.
  - *Age–sex pyramid*: horizontal bar chart showing the distribution of trips
    by age category and sex (male values are plotted to the left and female
    values to the right).
  - *Car ownership vs household size*: grouped bar chart showing the
    distribution of car ownership by household size.
  - *Car ownership vs income*: grouped bar chart illustrating how vehicle
    ownership varies across household income quartiles.

* **Mode Choice**
  - *Overall mode share*: pie chart of the percentage of trips by each
    transport mode.
  - *Mode share by age group*: stacked bar chart showing the modal split
    within each age category.
  - *Mode share by income group*: stacked bar chart comparing modal split
    across income quartiles.
  - *Mode share by car ownership*: stacked bar chart comparing modal
    split for households with no car, one car or more than one car.
  - *Mode share by employment status*: stacked bar chart showing how
    employed, students, retirees and others travel.

* **Trip Characteristics**
  - *Trip duration distribution*: histogram of trip durations (minutes).
  - *Trip distance distribution*: histogram of estimated trip distances (km).
  - *Duration vs distance*: scatter plot relating trip duration to distance,
    coloured by trip purpose.
  - *Average trip time by mode and purpose*: grouped bar chart showing the
    mean duration for each combination of mode and purpose.

* **Mobility Behaviour & Equity**
  - *Trips per capita by income group*: bar chart of the average number of
    trips per person in each income quartile.
  - *Trip count distribution per person*: bar chart showing how many
    people made 1, 2, 3 or more trips on the survey day.
  - *Trip rate by employment status*: bar chart of the average number of
    trips per person for each employment category.

* **Special Charts**
  - *Purpose → Mode Sankey*: Sankey diagram illustrating flows from trip
    purposes to transport modes.
  - *Mode preference by age group*: radar chart comparing the modal split
    across several age categories.
  - *Mode preference by employment status*: radar chart comparing the
    modal split across employment groups.

To run this app locally, install the required packages and execute:

.. code-block:: bash

    pip install streamlit pandas plotly
    streamlit run streamlit_app_tb.py

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the survey data from a CSV and set categorical types.

    Parameters
    ----------
    path: str
        Path to the CSV file containing the trip‑level dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with typed categorical columns.
    """
    df = pd.read_csv(path)
    cat_cols = [
        "purpose",
        "mode",
        "sex",
        "employment",
        "car_ownership",
        "age",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


@st.cache_data
def compute_household_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unique household summary table from the trip data.

    Each household appears once with attributes for number of persons,
    household income and car ownership.  Trip counts are ignored here;
    they are aggregated later as needed.

    Parameters
    ----------
    df: pd.DataFrame
        Trip‑level DataFrame.

    Returns
    -------
    pd.DataFrame
        Household summary with one row per household.
    """
    house_cols = ["household_id", "num_persons", "household_income", "car_ownership"]
    house_df = df[house_cols].drop_duplicates().copy()
    # Ensure numeric types for numeric columns
    house_df["num_persons"] = pd.to_numeric(house_df["num_persons"], errors="coerce")
    house_df["household_income"] = pd.to_numeric(house_df["household_income"], errors="coerce")
    return house_df


@st.cache_data
def compute_person_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trips to the person level.

    Computes the number of trips per person and attaches household income,
    car ownership and employment status.  This is used for trip rate
    analyses in the mobility behaviour section.

    Parameters
    ----------
    df: pd.DataFrame
        Trip‑level DataFrame.

    Returns
    -------
    pd.DataFrame
        Person‑level summary.
    """
    # Derive person attributes from the first trip record for each person
    person_attrs = ["person_id", "household_id", "employment", "sex", "age"]
    person_df = df[person_attrs].drop_duplicates(subset="person_id").copy()
    # Count number of trips per person
    trip_counts = df.groupby("person_id").size().reset_index(name="trips")
    person_df = person_df.merge(trip_counts, on="person_id", how="left")
    # Attach household income and car ownership
    household_attrs = df[["household_id", "household_income", "car_ownership"]].drop_duplicates()
    person_df = person_df.merge(household_attrs, on="household_id", how="left")
    # Convert numeric columns appropriately
    person_df["household_income"] = pd.to_numeric(person_df["household_income"], errors="coerce")
    return person_df


def plot_household_size_distribution(house_df: pd.DataFrame) -> go.Figure:
    """Return a bar chart of household size distribution."""
    counts = house_df["num_persons"].value_counts().sort_index()
    fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        labels={"x": "Household size", "y": "Number of households"},
        title="Household Size Distribution",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_age_sex_pyramid(df: pd.DataFrame) -> go.Figure:
    """Return a horizontal bar chart for age–sex pyramid.

    The function counts the number of trips by age and sex.  Male counts
    are plotted on the left (negative values) and female counts on the
    right.  If only one sex category is present the corresponding
    bar will be omitted.
    """
    if df.empty:
        return go.Figure()
    pivot = df.groupby(["age", "sex"]).size().reset_index(name="count")
    pivot = pivot.pivot(index="age", columns="sex", values="count").fillna(0)
    # Sort ages according to the categorical order for consistent display
    age_order = df["age"].astype(str).unique().tolist()
    pivot = pivot.reindex(age_order)
    male_col = None
    female_col = None
    for col in pivot.columns:
        if str(col).lower().startswith("m"):
            male_col = col
        elif str(col).lower().startswith("f"):
            female_col = col
    male_counts = pivot[male_col] if male_col is not None else pd.Series(0, index=pivot.index)
    female_counts = pivot[female_col] if female_col is not None else pd.Series(0, index=pivot.index)
    # Negate male counts for left side
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pivot.index.tolist(),
        x=-male_counts.values,
        name="Male",
        orientation="h",
    ))
    fig.add_trace(go.Bar(
        y=pivot.index.tolist(),
        x=female_counts.values,
        name="Female",
        orientation="h",
    ))
    fig.update_layout(
        title="Age–Sex Pyramid (Trip Counts)",
        xaxis=dict(title="Number of trips", tickformat=".0f"),
        yaxis=dict(title="Age group"),
        barmode="overlay",
        legend_title="Sex",
    )
    # Ensure symmetrical axis limits
    max_count = max(male_counts.max(), female_counts.max())
    fig.update_xaxes(range=[-max_count * 1.1, max_count * 1.1])
    return fig


def plot_car_vs_house_size(house_df: pd.DataFrame) -> go.Figure:
    """Return a grouped bar chart of car ownership by household size."""
    counts = (
        house_df.groupby(["num_persons", "car_ownership"]).size().reset_index(name="count")
    )
    # Convert numeric household size to string for categorical x axis
    counts["num_persons"] = counts["num_persons"].astype(int).astype(str)
    fig = px.bar(
        counts,
        x="num_persons",
        y="count",
        color="car_ownership",
        barmode="group",
        labels={"num_persons": "Household size", "count": "Number of households", "car_ownership": "Car ownership"},
        title="Car Ownership by Household Size",
    )
    return fig


def plot_car_vs_income(house_df: pd.DataFrame) -> go.Figure:
    """Return a grouped bar chart of car ownership by income quartile."""
    # Drop rows with missing income
    income_df = house_df.dropna(subset=["household_income"]).copy()
    if income_df.empty:
        return go.Figure()
    # Create income quartile categories
    income_df["income_bin"] = pd.qcut(income_df["household_income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    counts = income_df.groupby(["income_bin", "car_ownership"]).size().reset_index(name="count")
    fig = px.bar(
        counts,
        x="income_bin",
        y="count",
        color="car_ownership",
        barmode="group",
        labels={"income_bin": "Income quartile", "count": "Number of households", "car_ownership": "Car ownership"},
        title="Car Ownership by Income Quartile",
    )
    return fig


def compute_mode_share(df: pd.DataFrame, group_field: str) -> pd.DataFrame:
    """Compute mode share percentages for each category of a given field.

    Parameters
    ----------
    df: pd.DataFrame
        Filtered trip‑level DataFrame.
    group_field: str
        Column name by which to group the data (e.g. "age", "car_ownership").

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns [group_field, 'mode', 'share'] containing
        the share (in percent) of trips by mode within each category.
    """
    if df.empty or group_field not in df.columns:
        return pd.DataFrame(columns=[group_field, "mode", "share"])
    counts = df.groupby([group_field, "mode"]).size().reset_index(name="count")
    total = df.groupby(group_field).size().reset_index(name="total")
    merged = counts.merge(total, on=group_field, how="left")
    merged["share"] = merged["count"] / merged["total"]
    return merged


def plot_mode_share_by_field(df: pd.DataFrame, group_field: str, title: str) -> go.Figure:
    """Return a stacked bar chart of mode share by a given field."""
    share_df = compute_mode_share(df, group_field)
    if share_df.empty:
        return go.Figure()
    fig = px.bar(
        share_df,
        x=group_field,
        y="share",
        color="mode",
        title=title,
        labels={group_field: group_field.replace("_", " ").title(), "share": "Mode share"},
    )
    fig.update_layout(yaxis_tickformat=".0%")
    return fig


def plot_duration_distribution(df: pd.DataFrame) -> go.Figure:
    """Return a histogram of trip durations."""
    fig = px.histogram(
        df,
        x="duration_min",
        nbins=30,
        title="Trip Duration Distribution",
        labels={"duration_min": "Duration (min)", "count": "Number of trips"},
    )
    return fig


def plot_distance_distribution(df: pd.DataFrame) -> go.Figure:
    """Return a histogram of trip distances."""
    fig = px.histogram(
        df,
        x="distance_km",
        nbins=30,
        title="Trip Distance Distribution",
        labels={"distance_km": "Distance (km)", "count": "Number of trips"},
    )
    return fig


def plot_duration_vs_distance(df: pd.DataFrame) -> go.Figure:
    """Return a scatter plot of duration vs distance coloured by purpose."""
    fig = px.scatter(
        df,
        x="distance_km",
        y="duration_min",
        color="purpose",
        labels={"distance_km": "Distance (km)", "duration_min": "Duration (min)", "purpose": "Purpose"},
        title="Duration vs Distance",
        hover_data=["mode", "employment"],
    )
    return fig


def plot_avg_time_by_mode_purpose(df: pd.DataFrame) -> go.Figure:
    """Return a grouped bar chart of average trip duration by mode and purpose."""
    if df.empty:
        return go.Figure()
    stats = df.groupby(["mode", "purpose"]).agg(mean_duration=("duration_min", "mean")).reset_index()
    fig = px.bar(
        stats,
        x="mode",
        y="mean_duration",
        color="purpose",
        barmode="group",
        labels={"mode": "Mode", "mean_duration": "Average duration (min)", "purpose": "Purpose"},
        title="Average Trip Duration by Mode and Purpose",
    )
    return fig


def plot_trips_per_capita_by_income(person_df: pd.DataFrame) -> go.Figure:
    """Return a bar chart of average trips per person by income quartile."""
    df = person_df.dropna(subset=["household_income"]).copy()
    if df.empty:
        return go.Figure()
    df["income_bin"] = pd.qcut(df["household_income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    stats = df.groupby("income_bin")["trips"].mean().reset_index(name="avg_trips")
    fig = px.bar(
        stats,
        x="income_bin",
        y="avg_trips",
        title="Trips per Capita by Income Quartile",
        labels={"income_bin": "Income quartile", "avg_trips": "Average trips per person"},
    )
    return fig


def plot_trip_count_distribution(person_df: pd.DataFrame) -> go.Figure:
    """Return a bar chart of the distribution of trip counts per person."""
    counts = person_df["trips"].value_counts().sort_index()
    # Group 4 or more trips into a single category labelled "4+"
    # We'll create a new series with keys 0,1,2,3, '4+'
    distribution = {}
    for trips, count in counts.items():
        if trips >= 4:
            distribution["4+"] = distribution.get("4+", 0) + count
        else:
            distribution[str(int(trips))] = count
    # Ensure categories appear in order
    order = ["0", "1", "2", "3", "4+"]
    counts_ordered = [distribution.get(cat, 0) for cat in order]
    fig = px.bar(
        x=order,
        y=counts_ordered,
        title="Distribution of Trips per Person",
        labels={"x": "Number of trips", "y": "Number of people"},
    )
    return fig


def plot_trip_rate_by_employment(person_df: pd.DataFrame) -> go.Figure:
    """Return a bar chart of average trips per person by employment status."""
    stats = person_df.groupby("employment")["trips"].mean().reset_index(name="avg_trips")
    fig = px.bar(
        stats,
        x="employment",
        y="avg_trips",
        title="Trips per Capita by Employment Status",
        labels={"employment": "Employment status", "avg_trips": "Average trips per person"},
        color="employment",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig


def plot_sankey_purpose_mode(df: pd.DataFrame) -> go.Figure:
    """Return a Sankey diagram of flows from trip purposes to modes."""
    if df.empty:
        return go.Figure()
    # Identify unique purposes and modes
    purposes = df["purpose"].astype(str).unique().tolist()
    modes = df["mode"].astype(str).unique().tolist()
    # Create a mapping from node label to index
    labels = purposes + modes
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    # Compute link values
    flows = df.groupby(["purpose", "mode"]).size().reset_index(name="value")
    source_indices = [label_to_idx[p] for p in flows["purpose"]]
    target_indices = [label_to_idx[m] for m in flows["mode"]]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=flows["value"].tolist(),
        ),
    )])
    fig.update_layout(title="Trip Purpose → Mode Sankey Diagram")
    return fig


def plot_radar_mode_by_group(df: pd.DataFrame, group_field: str, top_groups: int = 4) -> go.Figure:
    """Return a radar chart showing mode preferences by group.

    Parameters
    ----------
    df: pd.DataFrame
        Trip‑level DataFrame.
    group_field: str
        Column to group by (e.g. 'age', 'employment').
    top_groups: int
        Maximum number of groups to plot (if more groups exist, the first
        ``top_groups`` categories are taken in sorted order).
    """
    # Compute mode share
    share_df = compute_mode_share(df, group_field)
    if share_df.empty:
        return go.Figure()
    # Determine top groups
    categories = sorted(df[group_field].astype(str).unique().tolist())[:top_groups]
    modes = sorted(df["mode"].astype(str).unique().tolist())
    fig = go.Figure()
    for group in categories:
        row = share_df[share_df[group_field] == group]
        # Ensure all modes present; missing modes get zero share
        shares = []
        for mode in modes:
            val = row[row["mode"] == mode]["share"].iloc[0] if not row[row["mode"] == mode].empty else 0.0
            shares.append(val)
        shares.append(shares[0])  # Close the loop for radar chart
        fig.add_trace(go.Scatterpolar(
            r=shares,
            theta=modes + [modes[0]],
            name=str(group),
        ))
    fig.update_layout(
        title=f"Mode Preferences by {group_field.replace('_', ' ').title()}",
        polar=dict(radialaxis=dict(visible=True, tickformat=".0%")),
        showlegend=True,
    )
    return fig


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="HH2025TB Mobility Dashboard", layout="wide")
    st.title("HH2025TB Household Travel Survey Dashboard")

    # Load full dataset
    DATA_PATH = "sample_data_full.csv"
    try:
        data = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Data file '{DATA_PATH}' not found. Please ensure the CSV is present in the application directory.")
        return

    # Sidebar filters and section selection
    st.sidebar.header("Filters")
    available_purposes = sorted(data["purpose"].cat.categories.tolist())
    available_sexes = sorted(data["sex"].cat.categories.tolist())
    available_ages = sorted(data["age"].astype(str).unique().tolist())

    selected_purposes = st.sidebar.multiselect(
        "Trip Purpose", options=available_purposes, default=[],
        help="Select one or more purposes to filter trips. Leave empty for all."
    )
    selected_ages = st.sidebar.multiselect(
        "Age Categories", options=available_ages, default=available_ages,
        help="Select one or more age categories to filter trips."
    )
    selected_sexes = st.sidebar.multiselect(
        "Sex", options=available_sexes, default=[],
        help="Select one or more sex categories to filter trips."
    )
    section = st.sidebar.selectbox(
        "Select Analysis Section",
        [
            "Socio‑Demographics & Household",
            "Mode Choice",
            "Trip Characteristics",
            "Mobility Behaviour & Equity",
            "Special Charts",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.write("Loaded dataset contains", len(data), "trip records.")

    # Apply filters to trips
    df_filtered = data.copy()
    if selected_purposes:
        df_filtered = df_filtered[df_filtered["purpose"].isin(selected_purposes)]
    if selected_ages:
        df_filtered = df_filtered[df_filtered["age"].isin(selected_ages)]
    if selected_sexes:
        df_filtered = df_filtered[df_filtered["sex"].isin(selected_sexes)]

    # Create person and household summaries for behaviour analyses
    house_df = compute_household_table(df_filtered if not df_filtered.empty else data)
    person_df = compute_person_table(df_filtered if not df_filtered.empty else data)

    # Dispatch to appropriate section
    if section == "Socio‑Demographics & Household":
        st.header("Socio‑Demographics & Household")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_household_size_distribution(house_df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_age_sex_pyramid(df_filtered), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_car_vs_house_size(house_df), use_container_width=True)
        with col4:
            st.plotly_chart(plot_car_vs_income(house_df), use_container_width=True)

    elif section == "Mode Choice":
        st.header("Mode Choice Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Overall mode share
            mode_counts = (
                df_filtered.groupby("mode").size().reset_index(name="count")
                if not df_filtered.empty else data.groupby("mode").size().reset_index(name="count")
            )
            fig_mode = px.pie(
                mode_counts,
                names="mode",
                values="count",
                title="Overall Mode Share",
            )
            st.plotly_chart(fig_mode, use_container_width=True)
        with col2:
            st.plotly_chart(plot_mode_share_by_field(df_filtered, "age", "Mode Share by Age Group"), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_mode_share_by_field(df_filtered, "household_income", "Mode Share by Income Quartile"), use_container_width=True)
        with col4:
            st.plotly_chart(plot_mode_share_by_field(df_filtered, "car_ownership", "Mode Share by Car Ownership"), use_container_width=True)
        # Employment chart separate row
        st.plotly_chart(plot_mode_share_by_field(df_filtered, "employment", "Mode Share by Employment Status"), use_container_width=True)

    elif section == "Trip Characteristics":
        st.header("Trip Characteristics")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_duration_distribution(df_filtered), use_container_width=True)
        with col2:
            st.plotly_chart(plot_distance_distribution(df_filtered), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_duration_vs_distance(df_filtered), use_container_width=True)
        with col4:
            st.plotly_chart(plot_avg_time_by_mode_purpose(df_filtered), use_container_width=True)

    elif section == "Mobility Behaviour & Equity":
        st.header("Mobility Behaviour & Equity")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_trips_per_capita_by_income(person_df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_trip_count_distribution(person_df), use_container_width=True)
        st.plotly_chart(plot_trip_rate_by_employment(person_df), use_container_width=True)

    elif section == "Special Charts":
        st.header("Special Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_sankey_purpose_mode(df_filtered), use_container_width=True)
        with col2:
            st.plotly_chart(plot_radar_mode_by_group(df_filtered, "age"), use_container_width=True)
        # Additional radar by employment can be displayed below
        st.plotly_chart(plot_radar_mode_by_group(df_filtered, "employment"), use_container_width=True)


if __name__ == "__main__":
    main()