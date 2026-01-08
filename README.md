# PatrolIQ---Smart-Safety-Analytics-Platform
Python • Streamlit Cloud Deployment • Machine Learning • Data Analysis • Unsupervised Learning • Clustering Algorithms • Dimensionality Reduction • Geographic Data Analysis • Data Visualization • MLflow



# 🚓 PatrolIQ – Smart Safety Analytics Platform

## 📌 Project Overview
**PatrolIQ** is a smart urban safety analytics platform designed to help law enforcement agencies make **data-driven policing and resource allocation decisions**. The platform leverages **unsupervised machine learning**, **geospatial analytics**, and **dimensionality reduction techniques** to analyze large-scale crime data and uncover hidden crime patterns.

Built using **500,000 crime records from the Chicago Crime Dataset**, PatrolIQ identifies crime hotspots, temporal crime patterns, and high-risk zones through clustering and visualization. The insights generated enable proactive policing, optimized patrol planning, and improved public safety outcomes.

---

## 🎯 Project Objectives
- Analyze large-scale urban crime data using unsupervised learning  
- Identify geographic crime hotspots and high-risk areas  
- Discover temporal crime patterns for proactive policing  
- Reduce complex high-dimensional crime data into interpretable visual representations  
- Track and compare unsupervised models using MLflow  
- Deploy a production-ready safety analytics platform using Streamlit Cloud  

---

## 🚀 Key Features

### 🗺️ Geographic Crime Hotspot Analysis
- Identifies high-crime zones using latitude and longitude data  
- Detects dense crime regions and removes noise/outliers  
- Visualizes crime clusters on interactive city maps  
- Highlights areas requiring increased police presence  

### ⏰ Temporal Crime Pattern Analysis
- Discovers time-based crime patterns (hourly, daily, seasonal)  
- Identifies peak crime hours and high-risk periods  
- Compares weekday vs weekend crime behavior  
- Enables time-based patrol and resource planning  

---

### 🧹 Data Processing & Feature Engineering
- Processed **500,000 crime records** sampled from a 8.7B dataset  
- Cleaned missing values, inconsistencies, and invalid locations  
- Engineered advanced features including:
  - Hour of day, day of week, month, season  
  - Weekend indicators  
  - Crime severity scores  
- Normalized geographic coordinates for distance-based modeling  

---

### 🤖 Unsupervised Machine Learning

#### Clustering Algorithms
- **K-Means Clustering** – Identifies distinct crime concentration zones  
- **DBSCAN** – Density-based hotspot detection with noise handling  
- **Hierarchical Clustering** – Reveals nested relationships between crime zones  

Models are evaluated using:
- Silhouette Score  
- Davies–Bouldin Index  
- Elbow Method  

The best-performing clustering model is selected for deployment.

#### Temporal Clustering
- Groups similar crime-time behaviors  
- Identifies late-night, rush-hour, and seasonal crime patterns  
- Creates temporal crime profiles for different incident types  

---

### 📉 Dimensionality Reduction & Visualization
- **PCA (Principal Component Analysis)**:
  - Reduces 22+ features into 2–3 components  
  - Preserves over 70% of data variance  
  - Identifies key drivers of crime patterns  
- **t-SNE**:
  - Generates 2D visualizations of complex crime data  
  - Clearly separates crime clusters and patterns  
- Interactive visual exploration of reduced feature spaces  

---

### 📈 MLflow Experiment Tracking
- Tracks all clustering and dimensionality reduction experiments  
- Logs:
  - Algorithm parameters  
  - Evaluation metrics  
  - Model artifacts  
- Enables model comparison and version control  
- Supports reproducibility and structured experimentation  

---

### 🖥️ Streamlit Web Application
- Interactive analytics platform  
- Geographic crime heatmaps with cluster overlays  
- Temporal crime dashboards  
- Interactive PCA and t-SNE visualizations  
- Integrated MLflow monitoring for model performance  

---

## 🏗️ System Architecture
Chicago Crime Dataset → Data Cleaning & Preprocessing → Feature Engineering & EDA → Clustering Analysis (Geographic & Temporal) → Dimensionality Reduction (PCA, t-SNE) → MLflow Experiment Tracking → Streamlit Application → Streamlit Cloud Deployment
---

## 🧰 Technologies Used
- Python  
- Pandas & NumPy  
- Unsupervised Machine Learning  
- K-Means, DBSCAN, Hierarchical Clustering  
- PCA, t-SNE  
- Geographic & Temporal Data Analysis  
- MLflow (Experiment Tracking)  
- Streamlit & Streamlit Cloud  
- Data Visualization  
- Git & GitHub  

---

## 🗄️ Dataset Description
- **Dataset:** Chicago Crime Dataset (Public Data)  
- **Records Used:** 500,000 crime incidents  
- **Total Features:** 22+ original and engineered variables  
- **Crime Categories:** 33 distinct crime types  
- **Geographic Coverage:** Chicago districts, wards, and community areas  
- Includes spatial, temporal, administrative, and crime-status attributes  

---

## 🧠 Learning Outcomes
- Large-scale urban data preprocessing and analysis  
- Practical application of unsupervised machine learning  
- Geographic and temporal crime analytics  
- Dimensionality reduction for complex datasets  
- MLflow-based experiment tracking and model comparison  
- Building and deploying real-world data science applications  
- Applying analytics to public safety and urban planning problems  

---

