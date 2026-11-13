import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Global Development Clustering",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# Load saved ML components
# -----------------------------
@st.cache_resource
def load_models():
    with open(BASE_DIR / "std_sca.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(BASE_DIR / "pca.pkl", "rb") as f:
        pca = pickle.load(f)

    with open(BASE_DIR / "mean_shift.pkl", "rb") as f:
        model = pickle.load(f)

    return scaler, pca, model


@st.cache_data
def load_training_data():
    # The notebook uses this filename. The fallback makes the app
    # tolerant of the common spelling used in the local project folder.
    candidates = [
        BASE_DIR / "World_development_mesurement.xlsx",
        BASE_DIR / "World_development_measurement.xlsx",
    ]

    for path in candidates:
        if path.exists():
            return pd.read_excel(path)

    xlsx_files = list(BASE_DIR.glob("*.xlsx"))
    if len(xlsx_files) == 1:
        return pd.read_excel(xlsx_files[0])

    raise FileNotFoundError(
        "World_development_mesurement.xlsx was not found in the app folder."
    )


def preprocess_like_notebook(df):
    data = df.copy()

    # Same columns removed in the notebook.
    data.drop(columns=["Ease of Business", "Number of Records"], errors="ignore", inplace=True)

    # Same object-to-numeric conversion used in the notebook.
    money_cols = [
        "Business Tax Rate",
        "GDP",
        "Health Exp/Capita",
        "Tourism Inbound",
        "Tourism Outbound",
    ]

    for col in money_cols:
        if col in data.columns:
            data[col] = (
                data[col]
                .astype("string")
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
            )
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # The notebook scales all numeric columns; Country is excluded automatically.
    num_cols = data.select_dtypes(include=["float64", "int64", "number"]).columns

    # Same median-imputation strategy as the notebook.
    for col in num_cols:
        data[col]=pd.to_numeric(data[col], errors="coerce").astype("float64")
        data[col] = data[col].fillna(data[col].median())

    return data, list(num_cols)


st.title("🌍 Global Development Clustering")
st.write(
    "This application uses the trained Mean Shift clustering model from the "
    "Global Development project."
)

st.info(
    "Pipeline used by the original notebook: median imputation → StandardScaler "
    "→ PCA (2 components) → Mean Shift."
)

try:
    scaler, pca, mean_shift = load_models()
    df = load_training_data()
except Exception as e:
    st.error(f"Could not load the project files: {e}")
    st.stop()

processed_df, feature_columns = preprocess_like_notebook(df)

# Make sure the deployment preprocessing produces exactly the same feature order
# as the saved scaler expects.
expected_features = list(getattr(scaler, "feature_names_in_", feature_columns))

missing = [c for c in expected_features if c not in processed_df.columns]
extra = [c for c in processed_df.columns if c not in expected_features]

if missing:
    st.error(f"Missing model features: {missing}")
    st.stop()

X = processed_df[expected_features]
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)
clusters = mean_shift.predict(X_pca)

# Attach predictions to the original rows.
result_df = df.copy()
result_df["Cluster"] = clusters

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Project Information")
st.sidebar.metric("Rows", len(df))
st.sidebar.metric("Original columns", df.shape[1])
st.sidebar.metric("PCA components", 2)
st.sidebar.write("Best model: **Mean Shift**")
st.sidebar.write("Silhouette Score: **0.5204**")

# -----------------------------
# Overview
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Countries / records", len(result_df))
col2.metric("Clusters found", len(np.unique(clusters)))
col3.metric("Best Silhouette Score", "0.5204")

st.subheader("📊 Cluster Distribution")
cluster_counts = result_df["Cluster"].value_counts().sort_index()
st.bar_chart(cluster_counts)

# -----------------------------
# Country lookup
# -----------------------------
st.subheader("🔎 Find a Country's Cluster")

if "Country" in result_df.columns:
    countries = sorted(result_df["Country"].dropna().astype(str).unique())
    selected_country = st.selectbox("Select a country", countries)

    country_rows = result_df[
        result_df["Country"].astype(str) == selected_country
    ].copy()

    if not country_rows.empty:
        st.success(
            f"{selected_country} belongs to **Cluster "
            f"{int(country_rows.iloc[0]['Cluster'])}**."
        )

        display_cols = ["Country", "Cluster"]
        st.dataframe(country_rows[display_cols], use_container_width=True)

# -----------------------------
# Full results
# -----------------------------
st.subheader("📋 Clustering Results")

filter_cluster = st.selectbox(
    "Filter by cluster",
    ["All"] + [str(int(c)) for c in sorted(np.unique(clusters))]
)

if filter_cluster == "All":
    filtered = result_df
else:
    filtered = result_df[result_df["Cluster"] == int(filter_cluster)]

st.dataframe(filtered, use_container_width=True)

# -----------------------------
# Download
# -----------------------------
csv = result_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download clustering results (CSV)",
    data=csv,
    file_name="global_development_clustered.csv",
    mime="text/csv",
)

st.caption(
    "Model artifacts: std_sca.pkl, pca.pkl, mean_shift.pkl. "
    "Predictions use the same preprocessing structure as the supplied notebook."
)
