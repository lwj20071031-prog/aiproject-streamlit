import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Student Clustering", layout="wide")
st.title("Student Clustering and Learning Profile Assignment (PCA version)")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Upload your CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Data Preview")
st.dataframe(df.head())

default_cols = ["math_test1", "math_test2", "map_overall", "map_number", "map_algebra", "map_geometry"]
feature_cols = st.multiselect(
    "Select numeric columns to use",
    options=list(df.columns),
    default=[c for c in default_cols if c in df.columns]
)

if len(feature_cols) < 2:
    st.error("Select at least 2 feature columns.")
    st.stop()

X_raw = df[feature_cols].to_numpy()
X_imputed = SimpleImputer(strategy="mean").fit_transform(X_raw)
X = StandardScaler().fit_transform(X_imputed)

latent = PCA(n_components=2, random_state=0).fit_transform(X)
df["z1"], df["z2"] = latent[:, 0], latent[:, 1]

st.subheader("2D Skill Space (PCA)")
fig1 = plt.figure(figsize=(6, 6))
plt.scatter(df["z1"], df["z2"])
plt.xlabel("z1"); plt.ylabel("z2")
st.pyplot(fig1)

k = st.slider("Number of clusters", 2, 10, 3)
df["cluster"] = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(latent)

st.subheader("Clusters")
fig2 = plt.figure(figsize=(6, 6))
for c in range(k):
    pts = df[df["cluster"] == c]
    plt.scatter(pts["z1"], pts["z2"], label=f"Cluster {c}")
plt.legend()
st.pyplot(fig2)

df["_overall_proxy"] = df[feature_cols].mean(axis=1, numeric_only=True)
ranked = df.groupby("cluster")["_overall_proxy"].mean().sort_values().index.tolist()

labels_pool = ["Foundations Group", "On-level Group", "Advanced Group", "High Flyers", "Support Needed",
               "Accelerated", "Core", "Extension", "Remedial", "Enrichment"]
label_map = {cl: labels_pool[i] if i < len(labels_pool) else f"Group {i+1}" for i, cl in enumerate(ranked)}
df["learning_profile"] = df["cluster"].map(label_map)

suggest = {
    "Foundations Group": "Basic practice worksheet with visual models.",
    "On-level Group": "Grade-level practice with some problem-solving tasks.",
    "Advanced Group": "Challenge tasks and project-based learning.",
}
df["assignment_suggestion"] = df["learning_profile"].map(suggest).fillna("Differentiated practice based on needs.")

cols_show = (["student_id"] if "student_id" in df.columns else []) + \
            ["cluster", "learning_profile", "assignment_suggestion", "z1", "z2"] + feature_cols
st.subheader("Preview")
st.dataframe(df[cols_show].head(50))

out_csv = df.drop(columns=["_overall_proxy"], errors="ignore").to_csv(index=False)
st.download_button("Download CSV", out_csv, "students_with_profiles.csv", "text/csv")
