# =====================================================
# PatrolIQ - MLflow Training Pipeline (FAST & ROBUST)
# =====================================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
DATA_PATH = "chicago_crime_sample_500k_clean.csv"
TRAIN_SAMPLE_SIZE = 50_000
EVAL_SAMPLE_SIZE = 15_000
HIER_SAMPLE_SIZE = 10_000
RANDOM_STATE = 42

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
df = pd.read_csv(DATA_PATH)

# -----------------------------------------------------
# FEATURE CREATION (OPTION B – INSIDE ML PIPELINE)
# -----------------------------------------------------

# Parse date safely
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# Day of week numeric
if "Day_of_Week_Num" not in df.columns:
    df["Day_of_Week_Num"] = df["Date"].dt.dayofweek

# Crime Severity
if "Crime_Severity_Score" not in df.columns:
    severity_map = {
        "HOMICIDE": 10,
        "CRIMINAL SEXUAL ASSAULT": 9,
        "ROBBERY": 8,
        "AGGRAVATED ASSAULT": 8,
        "ASSAULT": 7,
        "BURGLARY": 6,
        "MOTOR VEHICLE THEFT": 6,
        "THEFT": 5,
        "CRIMINAL DAMAGE": 4,
        "OTHER OFFENSE": 3,
        "NARCOTICS": 3,
        "PUBLIC PEACE VIOLATION": 2,
    }

    df["Crime_Severity_Score"] = df["Primary Type"].map(severity_map).fillna(3)

# -----------------------------------------------------
# FEATURE MATRICES
# -----------------------------------------------------
geo_features = df[["Latitude", "Longitude"]]

temporal_features = df[["Hour", "Day_of_Week_Num", "Month", "Is_Weekend"]]

pca_features = df[
    [
        "Latitude",
        "Longitude",
        "Hour",
        "Day_of_Week_Num",
        "Month",
        "Is_Weekend",
        "Crime_Severity_Score",
    ]
]

# -----------------------------------------------------
# SCALING
# -----------------------------------------------------
scaler_geo = StandardScaler()
geo_scaled = scaler_geo.fit_transform(geo_features)

scaler_pca = StandardScaler()
pca_scaled = scaler_pca.fit_transform(pca_features)

# -----------------------------------------------------
# SAMPLING (CRITICAL FOR SPEED & MEMORY)
# -----------------------------------------------------
np.random.seed(RANDOM_STATE)

train_idx = np.random.choice(len(df), TRAIN_SAMPLE_SIZE, replace=False)
eval_idx = np.random.choice(len(df), EVAL_SAMPLE_SIZE, replace=False)

geo_train = geo_scaled[train_idx]
geo_eval = geo_scaled[eval_idx]
pca_train = pca_scaled[train_idx]

# -----------------------------------------------------
# MLFLOW SETUP
# -----------------------------------------------------
mlflow.set_experiment("PatrolIQ_Crime_Clustering_FAST")

with mlflow.start_run(run_name="Crime_Clustering_Pipeline"):
    # =================================================
    # 1️⃣ K-MEANS CLUSTERING
    # =================================================
    k = 6
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    kmeans.fit(geo_train)

    kmeans_eval_labels = kmeans.predict(geo_eval)

    sil = silhouette_score(geo_eval, kmeans_eval_labels)
    dbi = davies_bouldin_score(geo_eval, kmeans_eval_labels)

    mlflow.log_param("kmeans_k", k)
    mlflow.log_metric("kmeans_silhouette", sil)
    mlflow.log_metric("kmeans_davies_bouldin", dbi)

    mlflow.sklearn.log_model(kmeans, "kmeans_model")

    # =================================================
    # 2️⃣ DBSCAN (SAMPLED)
    # =================================================
    dbscan = DBSCAN(eps=0.3, min_samples=100)
    db_labels = dbscan.fit_predict(geo_train)

    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    noise_ratio = np.mean(db_labels == -1)

    mlflow.log_param("dbscan_eps", 0.3)
    mlflow.log_param("dbscan_min_samples", 100)
    mlflow.log_metric("dbscan_clusters", n_clusters)
    mlflow.log_metric("dbscan_noise_ratio", noise_ratio)

    # =================================================
    # 3️⃣ HIERARCHICAL CLUSTERING (SAMPLED)
    # =================================================
    hier_data = geo_train[:HIER_SAMPLE_SIZE]
    linkage_matrix = linkage(hier_data, method="ward")

    mlflow.log_param("hierarchical_method", "ward")
    mlflow.log_param("hierarchical_sample_size", HIER_SAMPLE_SIZE)

    # =================================================
    # 4️⃣ PCA
    # =================================================
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(pca_train)

    explained_var = pca.explained_variance_ratio_.sum()

    mlflow.log_param("pca_components", 2)
    mlflow.log_metric("pca_explained_variance", explained_var)

    mlflow.sklearn.log_model(pca, "pca_model")

print("✅ MLflow tracking completed successfully (FAST & STABLE)")
