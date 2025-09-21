# weather_climatology_app_bingley_forecast.py

import streamlit as st
import pandas as pd
import requests
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Bingley Agro Climatology", layout="wide")

# -----------------
# Utility functions
# -----------------

def fetch_openmeteo_archive(lat, lon, start, end):
    """Fetch hourly ET, leaf wetness, and radiation from Open-Meteo archive API"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": [
            "evapotranspiration",
            "leaf_wetness_probability",
            "shortwave_radiation"
        ],
        "timezone": "UTC"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_openmeteo_forecast(lat, lon):
    """Fetch next 16 days hourly forecast"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "evapotranspiration",
            "leaf_wetness_probability",
            "shortwave_radiation"
        ],
        "timezone": "UTC",
        "forecast_days": 16
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def compute_daily_metrics(df):
    """Aggregate hourly data into daily ET, Leaf Wetness, and DLI"""
    df["date"] = df["time"].dt.date

    # ET: sum of hourly ET (mm)
    daily_et = df.groupby("date")["evapotranspiration"].sum()

    # Leaf Wetness: average daily probability (0-1)
    daily_lw = df.groupby("date")["leaf_wetness_probability"].mean()

    # DLI: convert shortwave radiation (W/m²) → mol/m²/day
    sw = df.groupby("date")["shortwave_radiation"].sum()  # Wh/m²/day
    sw_j = sw * 3600  # convert Wh/m² to J/m²
    dli = (sw_j * 4.6) / 1e6  # mol/m²/day

    daily = pd.DataFrame({
        "ET": daily_et,
        "LeafWetness": daily_lw,
        "DLI": dli
    })
    daily.index = pd.to_datetime(daily.index)
    daily["doy"] = daily.index.dayofyear
    daily["year"] = daily.index.year
    return daily


def build_climatology(daily, metric, years=10):
    """Compute climatology envelopes for given metric over last N years"""
    cutoff_year = daily.index.year.max() - years
    hist = daily[daily["year"] > cutoff_year]

    clim = hist.groupby("doy")[metric].agg(["min", "max", "mean"])
    return clim


def plot_climatology(daily, clim, metric, today=None, forecast=None):
    """Plot this year's daily values + forecast vs climatology envelopes"""
    if today is None:
        today = dt.date.today()

    current_year = daily[daily["year"] == today.year].set_index("doy")[metric]

    today_doy = today.timetuple().tm_yday
    doys = np.arange(today_doy - 183, today_doy + 183 + 1)
    doys = np.mod(doys - 1, 365) + 1  # wrap around year

    clim_window = clim.loc[clim.index.isin(doys)].dropna()
    current_window = current_year.loc[current_year.index.isin(doys)].dropna()

    x = clim_window.index.values  # numeric DOY

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(x, clim_window["min"].values, clim_window["max"].values,
                    alpha=0.2, label="10y range")
    ax.plot(x, clim_window["mean"].values, label="10y mean", color="blue")
    ax.plot(current_window.index.values, current_window.values,
            label=f"{today.year}", color="red")

    # Overlay forecast (if provided)
    if forecast is not None:
        f_df = compute_daily_metrics(forecast)
        f_df = f_df[f_df.index.year == today.year]
        f_series = f_df.set_index("doy")[metric].dropna()
        ax.plot(f_series.index.values, f_series.values,
                "--", color="orange", label="Forecast")

    ax.axvline(today_doy, color="k", linestyle="--", label="Today")
    ax.set_title(f"{metric} Climatology vs {today.year} – Bingley, Yorkshire")
    ax.set_xlabel("Day of Year (1–365)")
    ax.set_ylabel(metric)
    ax.legend()
    return fig


# -----------------
# Streamlit UI
# -----------------

st.title("🌱 Bingley Agro Climatology Dashboard")
st.write("Climatology of evapotranspiration, leaf wetness, and DLI for Bingley, Yorkshire.")

# Fixed coordinates for Bingley
lat, lon = 53.8486, -1.8370

# Sidebar
years = st.sidebar.slider("Climatology window (years)", 5, 20, 10)
today = dt.date.today()

if st.sidebar.button("Fetch & Analyze"):
    with st.spinner("Fetching and processing data for Bingley..."):
        start = (today - dt.timedelta(days=365*years+30)).isoformat()
        end = today.isoformat()

        # Archive (history)
        df_hist = fetch_openmeteo_archive(lat, lon, start, end)
        daily_hist = compute_daily_metrics(df_hist)

        # Forecast (next 16 days)
        df_fore = fetch_openmeteo_forecast(lat, lon)

        for metric in ["ET", "LeafWetness", "DLI"]:
            st.subheader(metric)
            clim = build_climatology(daily_hist, metric, years)
            fig = plot_climatology(daily_hist, clim, metric, today, forecast=df_fore)
            st.pyplot(fig)

        st.success("Analysis complete for Bingley ✅")
