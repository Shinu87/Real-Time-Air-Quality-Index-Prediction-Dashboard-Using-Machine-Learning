import streamlit as st
import pandas as pd
import joblib
import requests
import folium
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
from datetime import datetime, timezone
import pytz
import numpy as np
import gdown
import os

# -------------------- App Setup --------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    html, body, [data-testid="stApp"] {
        background: linear-gradient(to bottom, #e6f0fa, #f0f7ff);
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #1e293b;
    }
    .stButton>button {
        background: linear-gradient(to right, #10b981, #059669);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #059669, #047857);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid #10b981;
    }
    .metric-box {
        background: #f8fafc;
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        text-align: center;
    }
    .footer {
        text-align: center;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 2rem;
        padding: 1rem 0;
        border-top: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: #f1f5f9;
        padding: 1.5rem;
        border-right: 2px solid #cbd5e1;
        color: #1e3a8a;
    }
    [data-testid="stSidebar"] img {
        display: block;
        margin: 0 auto 1rem auto;
        border-radius: 50%;
        padding: 0.5rem;
        background: #e6f0fa;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p {
        color: #1e3a8a;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #ef4444;
    }
    .stSlider > div > div > div > div {
        background-color: #10b981;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #cbd5e1;
    }
    .stMetric {
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.75rem;
    }
    /* Ensure full-width map */
    [data-testid="stFullScreenFrame"] {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .folium-map-container {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 auto !important;
        padding: 0 !important;
    }
    .leaflet-container {
        width: 100% !important;
        max-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_trained_model():
    model_path = "best_model_pipeline.pkl"
    file_id = "1PYyd5qYIHQzjwDOUhLV1Vz5iypp1daIq"
    url = "https://drive.google.com/uc?export=download&id=1PYyd5qYIHQzjwDOUhLV1Vz5iypp1daIq"

    try:
        if not os.path.exists(model_path):
            gdown.download(url, model_path, quiet=False)
        return joblib.load(model_path)
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
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933884.png", width=80)
    st.markdown("<h2 style='text-align: center;'>Air Quality Monitor 🌍</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #1e3a8a;'>Select your input method</p>", unsafe_allow_html=True)
    method = st.radio("", ["Manual Entry", "Real-Time Data"], label_visibility="collapsed")
    st.markdown("<hr style='border-color: #e2e8f0;'>", unsafe_allow_html=True)

# -------------------- Main Content --------------------
st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem;'>Air Quality Index Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; margin-bottom: 2rem;'>Monitor local air quality with real-time data or manual inputs</p>", unsafe_allow_html=True)

# -------------------- Inputs --------------------
input_df, coords, time_used = None, None, None

st.markdown("<div class='card'><h3>🔧 Input Options</h3>", unsafe_allow_html=True)
if method == "Manual Entry":
    st.markdown("<h4>Manual Pollution Values</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 400.0, 35.0, help="Fine particles ≤2.5 µm")
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 400.0, 45.0)
    with col2:
        no2 = st.slider("NO2 (µg/m³)", 0.0, 200.0, 25.0)
        so2 = st.slider("SO2 (µg/m³)", 0.0, 200.0, 10.0)
    with col3:
        co = st.slider("CO (mg/m³)", 0.0, 10.0, 1.5)
        o3 = st.slider("O3 (µg/m³)", 0.0, 300.0, 30.0)

    input_df = pd.DataFrame([[pm25, pm10, no2, so2, co, o3]], columns=POLLUTANTS)

else:
    st.markdown("<h4>Live District Readings</h4>", unsafe_allow_html=True)
    region_map = get_districts_by_region()
    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Select Region", list(region_map.keys()), help="Choose a region in Maharashtra")
    with col2:
        district = st.selectbox("Select District", region_map[region], help="Choose a district")

    if district:
        data_vals, coords, time_used = get_aq_data_from_api(district)
        if data_vals:
            input_df = pd.DataFrame([[data_vals[p] for p in POLLUTANTS]], columns=POLLUTANTS)
            st.success("✅ Fetched latest air quality data")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            m_cols = st.columns(3)
            for i, p in enumerate(POLLUTANTS):
                with m_cols[i % 3]:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h4>{p}</h4>
                        <p style='font-size: 1.2rem; color: #1e293b;'>{data_vals[p]:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("❌ Could not retrieve live data.")

st.markdown("</div><hr>", unsafe_allow_html=True)
st.info("ℹ️ CO values are in mg/m³; others are in µg/m³.")

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
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    pred = st.session_state["aqi"]
    loc = st.session_state.get("location")
    ts = st.session_state.get("time_used")

    st.markdown(f"""
    <div class='metric-box'>
        <h3 style='color:#b91c1c;'>Predicted AQI: {pred:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)

    if ts:
        ist_time = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc).astimezone(pytz.timezone("Asia/Kolkata"))
        st.markdown(f"<p>Data Time (IST): {ist_time.strftime('%d %b %Y %H:%M')}</p>", unsafe_allow_html=True)

    for lower, upper, label, msg, note, color in [
        (0, 50, "Good", "Air quality is healthy.", "Enjoy outdoor activities.", "#10b981"),
        (51, 100, "Moderate", "Acceptable for most.", "Sensitive individuals may feel mild effects.", "#3b82f6"),
        (101, 150, "Unhealthy for Sensitive Groups", "May affect sensitive people.", "Consider staying indoors.", "#f59e0b"),
        (151, 200, "Unhealthy", "Harmful to all individuals.", "Limit outdoor exposure.", "#ef4444"),
        (201, 300, "Very Unhealthy", "Health warnings possible.", "Strongly advised to stay indoors.", "#7c3aed"),
        (301, float("inf"), "Hazardous", "Emergency conditions.", "Avoid all outdoor activity.", "#4b0082"),
    ]:
        if lower <= pred <= upper:
            st.markdown(f"""
            <div class='metric-box' style='border-left: 4px solid {color};'>
                <h4 style='color: {color};'>Level: {label}</h4>
                <p><b>Message:</b> {msg}</p>
                <p><b>Advice:</b> {note}</p>
            </div>
            """, unsafe_allow_html=True)
            break

    if loc:
        st.markdown("<h3>🌐 Location Map</h3>", unsafe_allow_html=True)
        folium_map = folium.Map(location=loc, zoom_start=10, tiles="CartoDB Positron")
        folium.Marker(
            loc,
            tooltip=f"AQI: {pred:.2f}",
            popup=f"AQI: {pred:.2f}",
            icon=folium.Icon(color="blue", icon="cloud")
        ).add_to(folium_map)
        st_folium(folium_map, width=None, height=500, returned_objects=[])
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Charts --------------------
if input_df is not None:
    st.markdown("<div class='card'><h3>📊 Pollutant Composition</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    chart_df = pd.DataFrame({
        "Pollutant": POLLUTANTS,
        "Value": input_df.iloc[0].values
    })

    with col1:
        pie_fig = px.pie(
            chart_df,
            names="Pollutant",
            values="Value",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        pie_fig.update_traces(textinfo="percent+label", pull=[0.05]*len(POLLUTANTS))
        pie_fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), title="Pollutant Distribution")
        st.plotly_chart(pie_fig, use_container_width=True)

    with col2:
        bar_fig = px.bar(
            chart_df,
            x="Pollutant",
            y="Value",
            text="Value",
            color="Pollutant",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        bar_fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        bar_fig.update_layout(
            yaxis_title="Concentration",
            margin=dict(t=0, b=0, l=0, r=0),
            showlegend=False,
            title="Pollutant Levels"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with col3:
        radar_fig = go.Figure(data=go.Scatterpolar(
            r=chart_df["Value"],
            theta=chart_df["Pollutant"],
            fill='toself',
            line=dict(color='#10b981')
        ))
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True),
                angularaxis=dict(direction="clockwise")
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            showlegend=False,
            title="Pollutant Radar"
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("""
<div class='footer'>
    Built with 🧠 using Streamlit • Powered by <a href="https://open-meteo.com/" target="_blank" style='color: #10b981;'>Open-Meteo</a>
</div>
""", unsafe_allow_html=True)
