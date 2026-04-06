import streamlit as st
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Earthquake Dashboard", page_icon="🌍", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("earthquake_model.pkl")

st.markdown("<h1 style='text-align: center;'>🌍 Earthquake Intelligence Dashboard</h1>", unsafe_allow_html=True)

# ---------------- CONTINENT SELECT ----------------
continent = st.selectbox(
    "🌍 Select Continent",
    ["Global", "Asia", "Europe", "Africa", "North America", "South America", "Oceania"]
)

# ---------------- BOUNDS ----------------
def get_bounds(continent):
    if continent == "Asia":
        return 5, 55, 60, 150
    elif continent == "Europe":
        return 35, 70, -10, 40
    elif continent == "Africa":
        return -35, 35, -20, 55
    elif continent == "North America":
        return 10, 75, -170, -50
    elif continent == "South America":
        return -60, 15, -90, -30
    elif continent == "Oceania":
        return -50, 0, 110, 180
    else:
        return None

# ---------------- MULTIPLE EARTHQUAKES ----------------
def get_multiple_earthquakes(continent):
    if continent == "Global":
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    else:
        minlat, maxlat, minlon, maxlon = get_bounds(continent)
        url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&minlatitude={minlat}&maxlatitude={maxlat}&minlongitude={minlon}&maxlongitude={maxlon}&orderby=time"

    data = requests.get(url).json()

    eq_list = []
    for feature in data['features'][:50]:
        props = feature['properties']
        coords = feature['geometry']['coordinates']

        eq_list.append({
            "lat": coords[1],
            "lon": coords[0],
            "mag": props['mag']
        })

    return pd.DataFrame(eq_list)

# ---------------- AUTO REFRESH ----------------
refresh = st.checkbox("🔄 Auto Refresh (every 10 sec)")

if refresh:
    time.sleep(10)
    st.rerun()

# ================= MAP =================
st.markdown("## 🌍 Live Earthquake Map")

eq_df = get_multiple_earthquakes(continent)

st.map(eq_df)

# ================= ALERT SYSTEM =================
high_eq = eq_df[eq_df['mag'] >= 6]

if not high_eq.empty:
    st.error(f"🚨 ALERT: {len(high_eq)} strong earthquakes detected!")
else:
    st.success("✅ No dangerous earthquakes right now")

# ================= REAL DATA GRAPH =================
st.markdown("## 📊 Earthquake Data Insights")

df = pd.read_csv("earthquake_1995-2023.csv")

fig, ax = plt.subplots()
ax.hist(df['magnitude'], bins=20)

ax.set_title("Magnitude Distribution (Real Data)")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Frequency")

st.pyplot(fig)

# ================= MANUAL PREDICTION =================
st.markdown("---")
st.markdown("## 🔮 Predict Earthquake Magnitude")

c1, c2 = st.columns(2)

with c1:
    cdi = st.number_input("CDI", 0, 10, 5)
    mmi = st.number_input("MMI", 0, 10, 5)
    tsunami = st.selectbox("Tsunami", [0, 1])
    sig = st.number_input("Significance", 0, 2000, 500)
    nst = st.number_input("NST", 0, 500, 100)

with c2:
    dmin = st.number_input("dmin", 0.0, 10.0, 0.5)
    gap = st.number_input("gap", 0.0, 360.0, 30.0)
    depth = st.number_input("Depth", 0.0, 700.0, 100.0)
    latitude = st.number_input("Latitude", -90.0, 90.0, 10.0)
    longitude = st.number_input("Longitude", -180.0, 180.0, 70.0)

year = st.number_input("Year", 1995, 2030, 2023)
month = st.number_input("Month", 1, 12, 5)
day = st.number_input("Day", 1, 31, 10)

if st.button("🚀 Predict Magnitude"):

    data = {
        'cdi': cdi,
        'mmi': mmi,
        'tsunami': tsunami,
        'sig': sig,
        'nst': nst,
        'dmin': dmin,
        'gap': gap,
        'depth': depth,
        'latitude': latitude,
        'longitude': longitude,
        'year': year,
        'month': month,
        'day': day,
        'alert_green': 1,
        'net_us': 1,
        'magType_mb': 1,
        'continent_Asia': 1,
        'country_India': 1
    }

    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)[0]

    if prediction >= 7:
        st.error(f"🚨 HIGH RISK: {prediction:.2f}")
    elif prediction >= 5:
        st.warning(f"⚠️ MODERATE: {prediction:.2f}")
    else:
        st.success(f"✅ LOW: {prediction:.2f}")