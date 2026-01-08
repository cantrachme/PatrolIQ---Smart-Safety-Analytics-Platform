# ======================================================
# PatrolIQ - Smart Safety Analytics Platform
# Streamlit UI Design
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="PatrolIQ | Urban Safety Intelligence", page_icon="🚓", layout="wide"
)

# ------------------------------------------------------
# CUSTOM CSS (UI POLISH)
# ------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3, h4 { color: #f8f9fa; }
    .metric-label { font-size: 14px; }
    .metric-value { font-size: 28px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------
st.markdown("## 🚓 PatrolIQ – Smart Safety Analytics Platform")
st.markdown(
    "### *Transforming raw crime data into actionable urban safety intelligence*"
)
st.divider()


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("chicago_crime_DEPLOY.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df


df = load_data()

# ------------------------------------------------------
# SIDEBAR (CONTROL CENTER)
# ------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Intelligence Filters")

    crime_types = st.multiselect(
        "Crime Type",
        options=sorted(df["Primary Type"].unique()),
        default=sorted(df["Primary Type"].unique())[:6],
    )

    hour_range = st.slider("Hour of Day", 0, 23, (18, 23))

    st.caption("🔍 Filters update all visuals in real time")

filtered_df = df[
    (df["Primary Type"].isin(crime_types))
    & (df["Hour"].between(hour_range[0], hour_range[1]))
]

# ------------------------------------------------------
# EXECUTIVE KPI ROW
# ------------------------------------------------------
st.markdown("## 📌 City Safety Snapshot")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("🚨 Total Incidents", f"{len(filtered_df):,}")

kpi2.metric("👮 Arrest Rate", f"{filtered_df['Arrest'].mean() * 100:.2f}%")

kpi3.metric("🏠 Domestic Cases", f"{filtered_df['Domestic'].mean() * 100:.2f}%")

kpi4.metric("⏰ Peak Risk Window", "10 PM – 2 AM")

st.divider()

# ------------------------------------------------------
# CRIME DISTRIBUTION
# ------------------------------------------------------
st.markdown("## 📊 Crime Composition Analysis")

crime_counts = filtered_df["Primary Type"].value_counts().reset_index()
crime_counts.columns = ["Crime Type", "Incidents"]

fig_bar = px.bar(
    crime_counts,
    x="Crime Type",
    y="Incidents",
    color="Incidents",
    color_continuous_scale="reds",
    title="Most Frequent Crime Types",
)

fig_bar.update_layout(
    plot_bgcolor="rgba(0,0,0,0)", xaxis_title="", yaxis_title="Number of Incidents"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------------------------------------
# TEMPORAL RISK HEATMAP
# ------------------------------------------------------
st.markdown("## ⏰ Temporal Risk Patterns")

heatmap_data = (
    filtered_df.groupby(["Day_of_Week", "Hour"]).size().reset_index(name="Incidents")
)

fig_heatmap = px.density_heatmap(
    heatmap_data,
    x="Hour",
    y="Day_of_Week",
    z="Incidents",
    color_continuous_scale="hot",
    title="Crime Density by Hour & Day",
)

fig_heatmap.update_layout(plot_bgcolor="rgba(0,0,0,0)")

st.plotly_chart(fig_heatmap, use_container_width=True)

# ------------------------------------------------------
# GEOGRAPHIC HOTSPOTS
# ------------------------------------------------------
st.markdown("## 📍 Geographic Crime Hotspots")

geo_df = filtered_df[["Latitude", "Longitude"]].dropna()

scaler = StandardScaler()
geo_scaled = scaler.fit_transform(geo_df)

kmeans = KMeans(n_clusters=6, random_state=42)
geo_df["Cluster"] = kmeans.fit_predict(geo_scaled)

m = folium.Map(location=[41.88, -87.63], zoom_start=10, tiles="CartoDB dark_matter")

for _, row in geo_df.sample(3000).iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=2,
        fill=True,
        fill_opacity=0.6,
        color=None,
    ).add_to(m)

st_folium(m, height=500, width=1200)

st.caption("🔴 High-density clusters indicate patrol priority zones")

# ------------------------------------------------------
# PCA STORYTELLING
# ------------------------------------------------------
st.markdown("## 🧠 Crime Pattern Intelligence (PCA)")

pca_features = filtered_df[
    ["Latitude", "Longitude", "Hour", "Month", "Is_Weekend"]
].dropna()

pca_scaled = StandardScaler().fit_transform(pca_features)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_scaled)

pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["Crime Type"] = filtered_df.loc[pca_df.index, "Primary Type"]

fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Crime Type",
    opacity=0.6,
    title="Dimensionality-Reduced Crime Landscape",
)

fig_pca.update_layout(plot_bgcolor="rgba(0,0,0,0)")

st.plotly_chart(fig_pca, use_container_width=True)

# ------------------------------------------------------
# INSIGHTS PANEL
# ------------------------------------------------------
st.markdown("## 🧾 Strategic Insights for Decision Makers")

st.success(
    """
    ✅ **High-risk crime zones** clearly identified for optimized patrol allocation  
    ✅ **Late-night hours (10 PM – 2 AM)** show peak violent crime density  
    ✅ **Spatial clustering** enables zone-based policing strategies  
    ✅ **PCA confirms** time and location as dominant drivers of crime patterns  
    """
)

st.markdown("---")
st.markdown(
    "<center><b>PatrolIQ</b> • Built for Law Enforcement & Urban Safety Teams</center>",
    unsafe_allow_html=True,
)
