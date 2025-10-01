"""
Comprehensive Dash Dashboard for HH2025TB
=======================================

This Dash application presents a rich set of mobility analyses for the 2025
Household Travel Survey.  It mirrors the functionality of the Streamlit
dashboard but uses Dash and Bootstrap for layout and interactivity.  The
application is organised into tabs corresponding to different analytical
themes: Socio‑Demographics & Household, Mode Choice, Trip Characteristics,
Mobility Behaviour & Equity, and Special Charts.  A row of filters
(trip purpose, age category and sex) allows users to subset the data
interactively.  Charts update automatically when filters or tabs change.

Note: This app reads the full survey data from ``sample_data_full.csv`` and
does not impose any sampling limitation.

To run the app locally, install the required packages and execute:

.. code-block:: bash

    pip install dash dash-bootstrap-components pandas plotly
    python dash_app_tb.py

The server will start on http://127.0.0.1:8050/ by default.

"""

import os

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cat_cols = ["purpose", "mode", "sex", "employment", "car_ownership", "age"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def compute_household_table(df: pd.DataFrame) -> pd.DataFrame:
    house_df = df[["household_id", "num_persons", "household_income", "car_ownership"]].drop_duplicates().copy()
    house_df["num_persons"] = pd.to_numeric(house_df["num_persons"], errors="coerce")
    house_df["household_income"] = pd.to_numeric(house_df["household_income"], errors="coerce")
    return house_df


def compute_person_table(df: pd.DataFrame) -> pd.DataFrame:
    person_attrs = ["person_id", "household_id", "employment", "sex", "age"]
    person_df = df[person_attrs].drop_duplicates(subset="person_id").copy()
    trip_counts = df.groupby("person_id").size().reset_index(name="trips")
    person_df = person_df.merge(trip_counts, on="person_id", how="left")
    household_attrs = df[["household_id", "household_income", "car_ownership"]].drop_duplicates()
    person_df = person_df.merge(household_attrs, on="household_id", how="left")
    person_df["household_income"] = pd.to_numeric(person_df["household_income"], errors="coerce")
    return person_df


def plot_household_size_distribution(house_df: pd.DataFrame) -> go.Figure:
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
    if df.empty:
        return go.Figure()
    pivot = df.groupby(["age", "sex"]).size().reset_index(name="count").pivot(index="age", columns="sex", values="count").fillna(0)
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
    fig = go.Figure()
    fig.add_trace(go.Bar(y=pivot.index.tolist(), x=-male_counts.values, name="Male", orientation="h"))
    fig.add_trace(go.Bar(y=pivot.index.tolist(), x=female_counts.values, name="Female", orientation="h"))
    fig.update_layout(
        title="Age–Sex Pyramid (Trip Counts)",
        xaxis=dict(title="Number of trips"),
        yaxis=dict(title="Age group"),
        barmode="overlay",
        legend_title="Sex",
    )
    max_count = max(male_counts.max(), female_counts.max()) if len(male_counts) else 0
    fig.update_xaxes(range=[-max_count * 1.1, max_count * 1.1])
    return fig


def plot_car_vs_house_size(house_df: pd.DataFrame) -> go.Figure:
    counts = house_df.groupby(["num_persons", "car_ownership"]).size().reset_index(name="count")
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
    income_df = house_df.dropna(subset=["household_income"]).copy()
    if income_df.empty:
        return go.Figure()
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
    if df.empty or group_field not in df.columns:
        return pd.DataFrame(columns=[group_field, "mode", "share"])
    counts = df.groupby([group_field, "mode"]).size().reset_index(name="count")
    total = df.groupby(group_field).size().reset_index(name="total")
    merged = counts.merge(total, on=group_field, how="left")
    merged["share"] = merged["count"] / merged["total"]
    return merged


def plot_mode_share_by_field(df: pd.DataFrame, group_field: str, title: str) -> go.Figure:
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
    fig = px.histogram(
        df,
        x="duration_min",
        nbins=30,
        title="Trip Duration Distribution",
        labels={"duration_min": "Duration (min)", "count": "Number of trips"},
    )
    return fig


def plot_distance_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="distance_km",
        nbins=30,
        title="Trip Distance Distribution",
        labels={"distance_km": "Distance (km)", "count": "Number of trips"},
    )
    return fig


def plot_duration_vs_distance(df: pd.DataFrame) -> go.Figure:
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
    counts = person_df["trips"].value_counts().sort_index()
    distribution = {}
    for trips, count in counts.items():
        if trips >= 4:
            distribution["4+"] = distribution.get("4+", 0) + count
        else:
            distribution[str(int(trips))] = count
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
    if df.empty:
        return go.Figure()
    purposes = df["purpose"].astype(str).unique().tolist()
    modes = df["mode"].astype(str).unique().tolist()
    labels = purposes + modes
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    flows = df.groupby(["purpose", "mode"]).size().reset_index(name="value")
    source_indices = [label_to_idx[p] for p in flows["purpose"]]
    target_indices = [label_to_idx[m] for m in flows["mode"]]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels),
        link=dict(source=source_indices, target=target_indices, value=flows["value"].tolist()),
    )])
    fig.update_layout(title="Trip Purpose → Mode Sankey Diagram")
    return fig


def plot_radar_mode_by_group(df: pd.DataFrame, group_field: str, top_groups: int = 4) -> go.Figure:
    share_df = compute_mode_share(df, group_field)
    if share_df.empty:
        return go.Figure()
    categories = sorted(df[group_field].astype(str).unique().tolist())[:top_groups]
    modes = sorted(df["mode"].astype(str).unique().tolist())
    fig = go.Figure()
    for group in categories:
        row = share_df[share_df[group_field] == group]
        shares = []
        for mode in modes:
            val = row[row["mode"] == mode]["share"].iloc[0] if not row[row["mode"] == mode].empty else 0.0
            shares.append(val)
        shares.append(shares[0])
        fig.add_trace(go.Scatterpolar(r=shares, theta=modes + [modes[0]], name=str(group)))
    fig.update_layout(
        title=f"Mode Preferences by {group_field.replace('_', ' ').title()}",
        polar=dict(radialaxis=dict(visible=True, tickformat=".0%")),
        showlegend=True,
    )
    return fig


# Load global data
DATA_PATH = "sample_data_full.csv"
try:
    data = load_data(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Data file '{DATA_PATH}' not found. Please ensure it is present.")

house_df_full = compute_household_table(data)
person_df_full = compute_person_table(data)

# Prepare filter options
available_purposes = sorted(data["purpose"].cat.categories.tolist())
available_sexes = sorted(data["sex"].cat.categories.tolist())
available_ages = sorted(data["age"].astype(str).unique().tolist())

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "HH2025TB Mobility Dashboard"


def serve_layout() -> html.Div:
    return dbc.Container([
        dbc.Row([dbc.Col(html.H2("HH2025TB Household Travel Survey Dashboard"), width=12)], className="my-3"),
        # Filter controls
        dbc.Row([
            dbc.Col([
                html.Label("Trip Purpose", className="form-label"),
                dcc.Dropdown(
                    id="purpose-filter",
                    options=[{"label": p, "value": p} for p in available_purposes],
                    multi=True,
                    placeholder="Select purpose(s)",
                ),
            ], md=4),
            dbc.Col([
                html.Label("Age Categories", className="form-label"),
                dcc.Dropdown(
                    id="age-filter",
                    options=[{"label": a, "value": a} for a in available_ages],
                    multi=True,
                    value=available_ages,
                    placeholder="Select age category(ies)",
                ),
            ], md=4),
            dbc.Col([
                html.Label("Sex", className="form-label"),
                dcc.Dropdown(
                    id="sex-filter",
                    options=[{"label": s, "value": s} for s in available_sexes],
                    multi=True,
                    placeholder="Select sex category(ies)",
                ),
            ], md=4),
        ], className="mb-3"),
        # Tabs for sections
        dcc.Tabs(
            id="section-tabs",
            value="socio",
            children=[
                dcc.Tab(label="Socio‑Demographics", value="socio"),
                dcc.Tab(label="Mode Choice", value="mode"),
                dcc.Tab(label="Trip Characteristics", value="trip"),
                dcc.Tab(label="Mobility Behaviour", value="behaviour"),
                dcc.Tab(label="Special Charts", value="special"),
            ],
        ),
        html.Div(id="tab-content", className="mt-3"),
    ], fluid=True)


app.layout = serve_layout


@app.callback(
    Output("tab-content", "children"),
    Input("section-tabs", "value"),
    Input("purpose-filter", "value"),
    Input("age-filter", "value"),
    Input("sex-filter", "value"),
)
def render_tab_content(tab, selected_purposes, selected_ages, selected_sexes):
    # Apply filters to data
    df_filtered = data.copy()
    if selected_purposes:
        df_filtered = df_filtered[df_filtered["purpose"].isin(selected_purposes)]
    if selected_ages:
        df_filtered = df_filtered[df_filtered["age"].isin(selected_ages)]
    if selected_sexes:
        df_filtered = df_filtered[df_filtered["sex"].isin(selected_sexes)]
    # Recompute aggregated tables based on filtered data
    house_df = compute_household_table(df_filtered if not df_filtered.empty else data)
    person_df = compute_person_table(df_filtered if not df_filtered.empty else data)
    if tab == "socio":
        # Four charts arranged in two rows
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_household_size_distribution(house_df)), md=6),
                dbc.Col(dcc.Graph(figure=plot_age_sex_pyramid(df_filtered)), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_car_vs_house_size(house_df)), md=6),
                dbc.Col(dcc.Graph(figure=plot_car_vs_income(house_df)), md=6),
            ]),
        ], fluid=True)
    elif tab == "mode":
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=px.pie(
                    df_filtered.groupby("mode").size().reset_index(name="count")
                    if not df_filtered.empty else data.groupby("mode").size().reset_index(name="count"),
                    names="mode",
                    values="count",
                    title="Overall Mode Share",
                )), md=6),
                dbc.Col(dcc.Graph(figure=plot_mode_share_by_field(df_filtered, "age", "Mode Share by Age Group")), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_mode_share_by_field(df_filtered, "household_income", "Mode Share by Income Quartile")), md=6),
                dbc.Col(dcc.Graph(figure=plot_mode_share_by_field(df_filtered, "car_ownership", "Mode Share by Car Ownership")), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_mode_share_by_field(df_filtered, "employment", "Mode Share by Employment Status")), md=12),
            ]),
        ], fluid=True)
    elif tab == "trip":
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_duration_distribution(df_filtered)), md=6),
                dbc.Col(dcc.Graph(figure=plot_distance_distribution(df_filtered)), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_duration_vs_distance(df_filtered)), md=6),
                dbc.Col(dcc.Graph(figure=plot_avg_time_by_mode_purpose(df_filtered)), md=6),
            ]),
        ], fluid=True)
    elif tab == "behaviour":
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_trips_per_capita_by_income(person_df)), md=6),
                dbc.Col(dcc.Graph(figure=plot_trip_count_distribution(person_df)), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_trip_rate_by_employment(person_df)), md=12),
            ]),
        ], fluid=True)
    elif tab == "special":
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_sankey_purpose_mode(df_filtered)), md=6),
                dbc.Col(dcc.Graph(figure=plot_radar_mode_by_group(df_filtered, "age")), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_radar_mode_by_group(df_filtered, "employment")), md=12),
            ]),
        ], fluid=True)
    else:
        return html.Div()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)