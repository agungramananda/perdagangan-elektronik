import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st

st.set_page_config(page_title="Perdagangan Elektronik", layout="wide")

DATA_PATH = Path(__file__).parent / "Online Retail.xlsx"
MODEL_PATH = Path(__file__).parent / "model" / "prophet_model.pkl"
TOP_10_PRODUCT_CODES = ['85123A', '22423', '85099B', '84879', '47566', '20725', '22720', 'POST', '23203', '20727']

@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Berkas data tidak ditemukan: {path}")

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
        raise FileNotFoundError(f"Berkas model tidak ditemukan: {path}")

    with open(path, "rb") as file:
        model = pickle.load(file)
    return model

def render_header():
    st.title("Dasbor Penjualan")
    st.caption("Toko Ritel")


def render_filters(df: pd.DataFrame):
    st.sidebar.header("Filter")

    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()
    date_range = st.sidebar.date_input(
        "Rentang tanggal invoice",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    countries = sorted(df["Country"].dropna().unique())
    selected_countries = st.sidebar.multiselect("Negara", countries, default=countries)

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
    kpi_cols[0].metric("Pendapatan", f"£{revenue:,.0f}")
    kpi_cols[1].metric("Invoices", f"{invoices:,}")
    kpi_cols[2].metric("Pelanggan", f"{customers:,}")
    kpi_cols[3].metric("Nilai Pesanan Rata-rata", f"£{aov:,.2f}")


def render_time_series(df: pd.DataFrame):
    monthly = df.groupby("InvoiceMonth", as_index=False)["Revenue"].sum()
    fig = px.line(monthly, x="InvoiceMonth", y="Revenue", markers=True, title="Pendapatan per Bulan")
    fig.update_layout(yaxis_title="Pendapatan", xaxis_title="Bulan")
    st.plotly_chart(fig, width='stretch')


def render_country_bar(df: pd.DataFrame):
    country_rev = df.groupby("Country", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    fig = px.bar(country_rev, x="Revenue", y="Country", orientation="h", title="Pendapatan per Negara")
    fig.update_layout(yaxis_title="Negara", xaxis_title="Pendapatan", height=500)
    st.plotly_chart(fig, width='stretch')


def render_top_products(df: pd.DataFrame):
    top_n = st.slider("Top N produk", min_value=5, max_value=25, value=10, step=1)
    prod_rev = (
        df.groupby(["StockCode", "Description"], as_index=False)["Revenue"].sum()
        .sort_values("Revenue", ascending=False)
        .head(top_n)
    )
    fig = px.bar(prod_rev, x="Revenue", y="Description", orientation="h", title="Produk Teratas Berdasarkan Pendapatan")
    fig.update_layout(yaxis_title="Produk", xaxis_title="Pendapatan", height=600)
    st.plotly_chart(fig, width='stretch')


def render_hourly(df: pd.DataFrame):
    hourly = df.groupby("Hour", as_index=False)["Revenue"].sum()
    fig = px.bar(hourly, x="Hour", y="Revenue", title="Pendapatan per Jam")
    fig.update_layout(xaxis=dict(dtick=1, title="Jam"), yaxis_title="Pendapatan")
    st.plotly_chart(fig, width='stretch')


def render_detail_table(df: pd.DataFrame):
    st.subheader("Data Produk")
    st.dataframe(df.head(200))

def load_product_model(stock_code: str) -> Prophet | None:
    product_model_path = Path(__file__).parent / "model" / "top_10_products_model" / f"prophet_model_{stock_code}.pkl"
    if not product_model_path.exists():
        return None
    with open(product_model_path, "rb") as file:
        return pickle.load(file)


def render_forecast(df: pd.DataFrame, model: Prophet):
    st.subheader("Prediksi Penjualan Keseluruhan")
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
            name="Batas atas",
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.update_layout(title="Prediksi Penjualan Harian", xaxis_title="Tanggal", yaxis_title="Penjualan")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detail Prediksi")
    forecast_display_renamed = forecast_display[["ds", "yhat"]].copy()
    forecast_display_renamed.columns = ["Tanggal", "Prediksi Penjualan"]
    st.dataframe(forecast_display_renamed)


def render_visualization_page(data: pd.DataFrame):
    render_header()
    date_range, selected_countries = render_filters(data)
    filtered = apply_filters(data, date_range, selected_countries)

    if filtered.empty:
        st.warning("Tidak ada data untuk filter terpilih.")
        return

    render_kpis(filtered)

    tab_trend, tab_country, tab_products, tab_hourly = st.tabs(
        ["Tren Bulanan", "Negara Pembeli", "Produk Teratas", "Penjualan Per Jam"]
    )

    with tab_trend:
        render_time_series(filtered)
    with tab_country:
        render_country_bar(filtered)
    with tab_products:
        render_top_products(filtered)
    with tab_hourly:
        render_hourly(filtered)

    render_detail_table(filtered)


def render_product_forecasts(df: pd.DataFrame, horizon_days: int = 60):
    st.subheader("Prediksi Produk Teratas")
    
    top_products = df[df["StockCode"].isin(TOP_10_PRODUCT_CODES)].groupby(
        ["StockCode", "Description"], as_index=False
    )["Revenue"].sum().sort_values("Revenue", ascending=False)
    
    if top_products.empty:
        st.warning("Tidak ada produk untuk diprediksi.")
        return
    
    tabs = st.tabs([f"{row['Description'][:30]}..." if len(row['Description']) > 30 else row['Description'] 
                    for _, row in top_products.iterrows()])
    
    for idx, (tab, (_, product_row)) in enumerate(zip(tabs, top_products.iterrows())):
        with tab:
            stock_code = product_row["StockCode"]
            description = product_row["Description"]
            
            product_model = load_product_model(stock_code)
            
            if product_model is None:
                st.warning(f"Model untuk produk {description} ({stock_code}) tidak tersedia.")
                continue
            
            future = product_model.make_future_dataframe(periods=horizon_days, freq="D")
            forecast = product_model.predict(future)
            forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_display["ds"],
                y=forecast_display["yhat"],
                name="Prediksi",
                mode="lines"
            ))
            
            fig.update_layout(
                title=f"Prediksi {description}",
                xaxis_title="Tanggal",
                yaxis_title="Penjualan",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Lihat detail prediksi"):
                forecast_table = forecast_display[["ds", "yhat"]].copy()
                forecast_table.columns = ["Tanggal", "Prediksi Penjualan"]
                st.dataframe(forecast_table, use_container_width=True)


def render_prediction_page(data: pd.DataFrame, model: Prophet):
    st.title("Prediksi Penjualan")
    st.caption("Forecasting menggunakan model Prophet")
    
    render_forecast(data, model)
    
    st.divider()
    
    render_product_forecasts(data)


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

    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih halaman", ("Visualisasi", "Prediksi"))

    if page == "Visualisasi":
        render_visualization_page(data)
    else:
        render_prediction_page(data, model)


if __name__ == "__main__":
    main()
