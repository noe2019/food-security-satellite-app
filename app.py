import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

from data import DISTRICTS, LOCKDOWN, simulate

st.set_page_config(page_title="Food Security – KwaZulu-Natal", page_icon="🌾", layout="wide")
st.title("🌾 Food Security Monitoring with Satellite Data – KwaZulu-Natal")
st.caption("Effects of drought and COVID-19 lockdowns on smallholder vs commercial agriculture using NDVI and rainfall · simulated MODIS/CHIRPS-like series")


@st.cache_data
def load():
    df = simulate()
    df["month"] = df.date.dt.month
    clim = df.groupby(["district", "system", "month"]).agg(ndvi_clim=("ndvi", "mean"), ndvi_sd=("ndvi", "std"),
                                                           rain_clim=("rainfall_mm", "mean"))
    df = df.join(clim, on=["district", "system", "month"])
    df["ndvi_anomaly"] = (df.ndvi - df.ndvi_clim) / df.ndvi_sd
    df["rain_anomaly_pct"] = df.rainfall_mm / df.rain_clim - 1
    return df


df = load()
with st.sidebar:
    st.header("Selection")
    dist = st.selectbox("District", ["All districts"] + list(DISTRICTS))
    st.info("Drought periods: Jun 2015–Jun 2016 (El Niño) and Jan–Oct 2019.\n\nLockdown: 27 Mar – 30 Jun 2020.")
    st.markdown("[Portfolio](https://noe2019.github.io/DA-Portfolio-Website/)")

f = df if dist == "All districts" else df[df.district == dist]
ts = f.groupby(["date", "system"]).agg(ndvi=("ndvi", "mean"), anomaly=("ndvi_anomaly", "mean"), rain=("rainfall_mm", "mean")).reset_index()

t1, t2, t3 = st.tabs(["Time series", "District map", "Impact model"])
with t1:
    fig = px.line(ts, x="date", y="ndvi", color="system", title=f"Monthly NDVI – {dist}")
    for a, b, lbl in [("2015-06-01", "2016-06-30", "Drought"), ("2019-01-01", "2019-10-31", "Dry spell"), (LOCKDOWN[0], LOCKDOWN[1], "Lockdown")]:
        fig.add_vrect(x0=a, x1=b, fillcolor="orange" if lbl != "Lockdown" else "red", opacity=.12, line_width=0, annotation_text=lbl)
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(ts[ts.system == "Smallholder"], x="date", y="rain", title="Mean monthly rainfall (mm)"), use_container_width=True)
    b.plotly_chart(px.line(ts, x="date", y="anomaly", color="system", title="Standardized NDVI anomaly").add_hline(y=-1, line_dash="dash"),
                   use_container_width=True)
with t2:
    month = st.select_slider("Month", options=[d.strftime("%Y-%m") for d in sorted(df.date.unique())], value="2020-05")
    sys_ = st.radio("Farming system", ["Smallholder", "Commercial"], horizontal=True)
    m = df[(df.date == pd.Timestamp(month + "-01")) & (df.system == sys_)]
    fig = px.scatter_mapbox(m, lat="lat", lon="lon", color="ndvi_anomaly", size=[18] * len(m), hover_name="district",
                            hover_data={"ndvi": ":.3f", "rainfall_mm": True, "ndvi_anomaly": ":.2f", "lat": False, "lon": False},
                            color_continuous_scale="RdYlGn", range_color=[-3, 3], zoom=5.6, height=560,
                            center={"lat": -28.9, "lon": 30.9}, mapbox_style="open-street-map", title=f"NDVI anomaly – {sys_} – {month}")
    st.plotly_chart(fig, use_container_width=True)
with t3:
    st.write("Linear model per farming system: **NDVI anomaly ~ rainfall anomaly (lagged 1 month) + lockdown**")
    out = []
    for s, g in f.sort_values("date").groupby("system"):
        g = g.assign(rain_lag=g.groupby("district").rain_anomaly_pct.shift(1)).dropna(subset=["rain_lag"])
        lr = LinearRegression().fit(g[["rain_lag", "lockdown"]], g.ndvi_anomaly)
        out.append({"system": s, "effect of +10% rainfall (σ)": lr.coef_[0] * .1, "lockdown effect (σ)": lr.coef_[1],
                    "R²": lr.score(g[["rain_lag", "lockdown"]], g.ndvi_anomaly)})
    res = pd.DataFrame(out).set_index("system")
    st.dataframe(res.style.format("{:+.3f}"), use_container_width=True)
    st.plotly_chart(go.Figure([go.Bar(x=res.index, y=res["lockdown effect (σ)"], marker_color=["#2E75B6", "#C0392B"][: len(res)])])
                    .update_layout(title="Estimated lockdown effect on NDVI anomaly", yaxis_title="Standard deviations"), use_container_width=True)
    st.caption("Smallholder farms show a stronger lockdown and drought response (less irrigation, restricted access to inputs and markets).")
