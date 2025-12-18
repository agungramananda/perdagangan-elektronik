import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st

st.set_page_config(page_title="Online Retail BI", layout="wide")

DATA_PATH = Path(__file__).parent / "Online Retail.xlsx"
MODEL_PATH = Path(__file__).parent / "model" / "prophet_model.pkl"

@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_excel(path, parse_dates=["InvoiceDate"])

    df = df.dropna(subset=["InvoiceNo", "InvoiceDate", "StockCode", "Description", "UnitPrice", "Quantity"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["CustomerID"] = df["CustomerID"].astype("Int64")

    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDateOnly"] = df["InvoiceDate"].dt.date
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["InvoiceWeek"] = df["InvoiceDate"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    df["Hour"] = df["InvoiceDate"].dt.hour
    return df


@st.cache_resource(show_spinner=True)
def load_model(path: Path) -> Prophet:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as file:
        model = pickle.load(file)
    return model

def render_header():
    st.title("Dashboard Penjualan")
    st.caption("Implementasi BI")


def render_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")

    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()
    date_range = st.sidebar.date_input(
        "Invoice date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    countries = sorted(df["Country"].dropna().unique())
    selected_countries = st.sidebar.multiselect("Countries", countries, default=countries)

    return date_range, selected_countries


def apply_filters(df: pd.DataFrame, date_range, countries):
    start_date, end_date = date_range if isinstance(date_range, tuple) else (date_range, date_range)
    mask = (
        (df["InvoiceDateOnly"] >= start_date)
        & (df["InvoiceDateOnly"] <= end_date)
        & (df["Country"].isin(countries))
    )
    return df[mask]


def render_kpis(df: pd.DataFrame):
    revenue = df["Revenue"].sum()
    invoices = df["InvoiceNo"].nunique()
    customers = df["CustomerID"].nunique()
    aov = revenue / invoices if invoices else 0

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Revenue", f"£{revenue:,.0f}")
    kpi_cols[1].metric("Invoices", f"{invoices:,}")
    kpi_cols[2].metric("Customers", f"{customers:,}")
    kpi_cols[3].metric("Avg Order Value", f"£{aov:,.2f}")


def render_time_series(df: pd.DataFrame):
    monthly = df.groupby("InvoiceMonth", as_index=False)["Revenue"].sum()
    fig = px.line(monthly, x="InvoiceMonth", y="Revenue", markers=True, title="Revenue by Month")
    fig.update_layout(yaxis_title="Revenue", xaxis_title="Month")
    st.plotly_chart(fig, width='stretch')


def render_country_bar(df: pd.DataFrame):
    country_rev = df.groupby("Country", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    fig = px.bar(country_rev, x="Revenue", y="Country", orientation="h", title="Revenue by Country")
    fig.update_layout(yaxis_title="Country", xaxis_title="Revenue", height=500)
    st.plotly_chart(fig, width='stretch')


def render_top_products(df: pd.DataFrame):
    top_n = st.slider("Top N products", min_value=5, max_value=25, value=10, step=1)
    prod_rev = (
        df.groupby(["StockCode", "Description"], as_index=False)["Revenue"].sum()
        .sort_values("Revenue", ascending=False)
        .head(top_n)
    )
    fig = px.bar(prod_rev, x="Revenue", y="Description", orientation="h", title="Top Products by Revenue")
    fig.update_layout(yaxis_title="Product", xaxis_title="Revenue", height=600)
    st.plotly_chart(fig, width='stretch')


def render_hourly(df: pd.DataFrame):
    hourly = df.groupby("Hour", as_index=False)["Revenue"].sum()
    fig = px.bar(hourly, x="Hour", y="Revenue", title="Revenue by Hour of Day")
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(fig, width='stretch')


def render_detail_table(df: pd.DataFrame):
    st.subheader("Filtered data sample")
    st.dataframe(df.head(200))

def render_forecast(df: pd.DataFrame, model: Prophet):
    st.subheader("Prediksi Penjualan")
    horizon_days = st.slider("Prediksi (hari)", min_value=30, max_value=180, value=60, step=15)

    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future)
    forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_display["ds"], y=forecast_display["yhat"], name="Prediksi", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=forecast_display["ds"],
            y=forecast_display["yhat_upper"],
            name="Upper",
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
   
    fig.update_layout(title="Prediksi Penjualan Harian", xaxis_title="Date", yaxis_title="Sales")

    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Detail Prediksi")
    forecast_display_renamed = forecast_display.copy()
    forecast_display_renamed.columns = ["Tanggal", "Prediksi Penjualan", "Batas Bawah", "Batas Atas"]
    st.dataframe(forecast_display_renamed.tail(15))


def main():
    try:
        data = load_data(DATA_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    render_header()
    date_range, selected_countries = render_filters(data)
    filtered = apply_filters(data, date_range, selected_countries)

    if filtered.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    render_kpis(filtered)

    col1, col2 = st.columns(2)
    with col1:
        render_time_series(filtered)
    with col2:
        render_country_bar(filtered)

    col3, col4 = st.columns(2)
    with col3:
        render_top_products(filtered)
    with col4:
        render_hourly(filtered)

    render_forecast(data, model)
    render_detail_table(filtered)


if __name__ == "__main__":
    main()
