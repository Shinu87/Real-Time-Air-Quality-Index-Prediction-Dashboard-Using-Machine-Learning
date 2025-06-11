import streamlit as st
import pandas as pd
import joblib
import requests
import folium
import plotly.express as px
from streamlit_folium import st_folium
from datetime import datetime, timezone
import pytz
import numpy as np

# -------------------- App Setup --------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

# -------------------- Page Style --------------------
st.markdown("""
<style>
    body {
        background-color: #eef2f5;
    }
    .stButton button {
        background-color: #26a69a;
        color: white;
        font-weight: 600;
        border-radius: 8px;
    }
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        font-size: 0.85rem;
        color: #666;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_trained_model():
    try:
        return joblib.load("best_model_pipeline.pkl")
    except Exception as err:
        st.error(f"Model loading failed: {err}")
        return None

model_pipeline = load_trained_model()

# -------------------- Region-District Data --------------------
@st.cache_data
def get_districts_by_region():
    return {
        "Vidarbha": ["Nagpur", "Amravati", "Chandrapur", "Yavatmal", "Akola"],
        "Marathwada": ["Aurangabad", "Beed", "Latur", "Nanded", "Parbhani"],
        "Western Maharashtra": ["Pune", "Kolhapur", "Sangli", "Satara"],
        "Konkan": ["Mumbai", "Thane", "Raigad", "Ratnagiri"],
        "Northern Maharashtra": ["Nashik", "Dhule", "Jalgaon"]
    }

# -------------------- Fetch AQ Data --------------------
def get_aq_data_from_api(city):
    try:
        geo_api = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_resp = requests.get(geo_api).json()

        if not geo_resp.get("results"):
            return None, None, None

        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]

        aq_api = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        aq_data = requests.get(aq_api).json()
        hour_data = aq_data.get("hourly", {})
        times = hour_data.get("time", [])

        if not times:
            return None, [lat, lon], None

        now = datetime.now(timezone.utc)
        nearest_time_idx = int(np.argmin([abs((datetime.fromisoformat(t).replace(tzinfo=timezone.utc) - now).total_seconds()) for t in times]))

        vals = {
            "PM2.5": hour_data["pm2_5"][nearest_time_idx],
            "PM10": hour_data["pm10"][nearest_time_idx],
            "NO2": hour_data["nitrogen_dioxide"][nearest_time_idx],
            "SO2": hour_data["sulphur_dioxide"][nearest_time_idx],
            "CO": hour_data["carbon_monoxide"][nearest_time_idx] / 1000,  # µg/m³ to mg/m³
            "O3": hour_data["ozone"][nearest_time_idx],
        }

        return vals, [lat, lon], times[nearest_time_idx]
    except Exception as e:
        st.error(f"Failed to get API data: {e}")
        return None, None, None

# -------------------- Sidebar --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933884.png", width=90)
    st.title("Air Quality Monitor")
    method = st.radio("Select Data Source:", ["Manual Entry", "Use Real-Time Data"])

# -------------------- Title --------------------
st.title("Air Quality Index Estimator")
st.write("Check your local air quality using real-time data or manual inputs.")

# -------------------- Inputs --------------------
input_df, coords, time_used = None, None, None

if method == "Manual Entry":
    st.subheader("Manual Pollution Values")
    col1, col2, col3 = st.columns(3)

    with col1:
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 400.0, 35.0)
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 400.0, 45.0)
    with col2:
        no2 = st.slider("NO2 (µg/m³)", 0.0, 200.0, 25.0)
        so2 = st.slider("SO2 (µg/m³)", 0.0, 200.0, 10.0)
    with col3:
        co = st.slider("CO (mg/m³)", 0.0, 10.0, 1.5)
        o3 = st.slider("O3 (µg/m³)", 0.0, 300.0, 30.0)

    input_df = pd.DataFrame([[pm25, pm10, no2, so2, co, o3]], columns=POLLUTANTS)

else:
    st.subheader("Live District Readings")
    region_map = get_districts_by_region()
    reg_col, dist_col = st.columns(2)

    with reg_col:
        region = st.selectbox("Select Region", list(region_map.keys()))
    with dist_col:
        district = st.selectbox("Select District", region_map[region])

    if district:
        data_vals, coords, time_used = get_aq_data_from_api(district)
        if data_vals:
            input_df = pd.DataFrame([[data_vals[p] for p in POLLUTANTS]], columns=POLLUTANTS)
            st.success("Fetched latest values successfully.")
            m_cols = st.columns(len(POLLUTANTS))
            for i, p in enumerate(POLLUTANTS):
                m_cols[i].metric(p, f"{data_vals[p]:.2f}")
        else:
            st.error("Could not retrieve live data.")

st.markdown("---")

st.info("Note: CO values are displayed in mg/m³, while others are in µg/m³.")

# -------------------- Predict AQI --------------------
if st.button("Predict AQI"):
    if model_pipeline and input_df is not None:
        try:
            aqi_val = model_pipeline.predict(input_df)[0]
            st.session_state["aqi"] = aqi_val
            st.session_state["location"] = coords
            st.session_state["time_used"] = time_used
        except Exception as ex:
            st.error(f"Error in prediction: {ex}")

# -------------------- Results --------------------
if "aqi" in st.session_state:
    st.subheader("Predicted AQI")
    pred = st.session_state["aqi"]
    loc = st.session_state.get("location")
    ts = st.session_state.get("time_used")

    st.markdown(f"<h3 style='text-align:center;'>AQI Value: <span style='color:crimson;'>{pred:.2f}</span></h3>", unsafe_allow_html=True)

    if ts:
        ist_time = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).astimezone(pytz.timezone("Asia/Kolkata"))
        st.write(f"Data Time (IST): {ist_time.strftime('%d %b %Y %H:%M')}")

    for lower, upper, label, msg, note in [
        (0, 50, "Good", "Air quality is healthy.", "Enjoy outdoor activities."),
        (51, 100, "Moderate", "Acceptable for most.", "Sensitive individuals may feel mild effects."),
        (101, 150, "Unhealthy for Sensitive Groups", "May affect sensitive people.", "Consider staying indoors."),
        (151, 200, "Unhealthy", "Harmful to all individuals.", "Limit outdoor exposure."),
        (201, 300, "Very Unhealthy", "Health warnings possible.", "Strongly advised to stay indoors."),
        (301, float("inf"), "Hazardous", "Emergency conditions.", "Avoid all outdoor activity."),
    ]:
        if lower <= pred <= upper:
            st.info(f"Level: {label}")
            st.markdown(f"<div class='metric-box'><b>Message:</b> {msg}<br><b>Advice:</b> {note}</div>", unsafe_allow_html=True)
            break

    if loc:
        st.markdown("### Location Map")
        folium_map = folium.Map(location=loc, zoom_start=9)
        folium.Marker(loc, tooltip=f"AQI: {pred:.2f}").add_to(folium_map)
        st_folium(folium_map, width=700, height=450)

# -------------------- Charts --------------------
if input_df is not None:
    st.markdown("### Pollutant Levels")
    col1, col2 = st.columns(2)
    chart_df = pd.DataFrame({
        "Pollutant": POLLUTANTS,
        "Value": input_df.iloc[0].values
    })

    with col1:
        st.plotly_chart(px.pie(chart_df, names="Pollutant", values="Value", hole=0.35), use_container_width=True)
    with col2:
        bar_fig = px.bar(chart_df, x="Pollutant", y="Value", text="Value", color="Pollutant")
        bar_fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(bar_fig, use_container_width=True)

# -------------------- Footer --------------------
st.markdown("""
---
<div class="footer">
    Built with 🧠 using Streamlit • Data from <a href="https://open-meteo.com/" target="_blank">Open-Meteo</a>
</div>
""", unsafe_allow_html=True)
