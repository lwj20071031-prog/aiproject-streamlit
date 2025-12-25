import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Student Grouping", layout="wide")
st.title("Student Grouping (MAP percentile + Unit Tests)")

st.caption("Upload CSV (UTF-8 recommended). Groups are labeled Group 1 (best) → Group N.")


# -------------------------
# Grade + Class picker
# -------------------------
GRADE_CLASS_COUNT = {
    "JG1": 2,
    "SG1": 4,
    "Grade 2": 4,
    "Grade 3": 5,
    "Grade 4": 5,
    "Grade 5": 6,
}

st.sidebar.header("Class Info")
selected_grade = st.sidebar.selectbox("Select grade", list(GRADE_CLASS_COUNT.keys()))
selected_class = st.sidebar.selectbox(
    "Select class",
    [f"Class {i}" for i in range(1, GRADE_CLASS_COUNT[selected_grade] + 1)]
)

# Optional: add these to every row so exported CSV includes them
run_grade = selected_grade
run_class = f"{selected_grade} {selected_class}"


# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload your CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
work = df.copy()


# -------------------------
# Column detection helpers
# -------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

cols_norm = {c: norm(c) for c in work.columns}

# Find student id column (optional)
student_id_col = None
for c, n in cols_norm.items():
    if n in ["student_id", "student id", "id"]:
        student_id_col = c
        break

# Find MAP column: contains both "map" and "math"
map_candidates = [c for c, n in cols_norm.items() if ("map" in n and "math" in n)]
map_col = map_candidates[0] if map_candidates else None

# Find unit tests 1..10: "math unit test 1" ... "math unit test 10"
unit_pairs = []
for c, n in cols_norm.items():
    m = re.search(r"math\s*unit\s*test\s*(\d+)", n)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 10:
            unit_pairs.append((num, c))
unit_test_cols = [c for _, c in sorted(unit_pairs)]

st.subheader("Detected columns")
st.write("student_id:", student_id_col)
st.write("MAP:", map_col)
st.write("Unit tests:", unit_test_cols)

if map_col is None and len(unit_test_cols) == 0:
    st.error("No MAP column found and no 'math unit test N' columns found (1..10). Check your headers.")
    st.stop()

# If no student_id, create one from row number (safe)
if student_id_col is None:
    work["student_id"] = [f"Student_{i+1:03d}" for i in range(len(work))]
    student_id_col = "student_id"


# -------------------------
# Convert MAP to percentile (0..100) — MAP only
# -------------------------
map_feature = None
if map_col is not None:
    map_vals = pd.to_numeric(work[map_col], errors="coerce")
    work["map_percentile"] = map_vals.rank(pct=True) * 100.0
    map_feature = "map_percentile"


# -------------------------
# Feature list (model uses MAP percentile + unit tests)
# -------------------------
feature_cols = []
if map_feature is not None:
    feature_cols.append(map_feature)
feature_cols += unit_test_cols

# Ensure numeric
for c in feature_cols:
    work[c] = pd.to_numeric(work[c], errors="coerce")

if len(feature_cols) < 2:
    st.error("Need at least 2 numeric features (e.g., MAP + at least one unit test).")
    st.stop()


# -------------------------
# Settings (k + weights)
# -------------------------
st.subheader("Grouping settings")

k = st.slider("Number of groups", min_value=2, max_value=10, value=3)

# Weights: user controls MAP weight %, remaining split across unit tests
if map_feature is not None and len(unit_test_cols) > 0:
    map_weight_pct = st.slider("MAP weight (%)", 0, 100, 60)
    map_w = map_weight_pct / 100.0
    remaining = 1.0 - map_w
    unit_each = remaining / len(unit_test_cols)

    weights = np.array([map_w] + [unit_each] * len(unit_test_cols), dtype=float)

    MAP_weight_pct_used = map_weight_pct
    UnitTests_total_weight_pct_used = (remaining * 100.0)
    UnitTest_weight_each_pct_used = (unit_each * 100.0)

elif map_feature is not None:
    st.info("Only MAP detected (no unit tests). MAP weight = 100%.")
    weights = np.array([1.0], dtype=float)

    MAP_weight_pct_used = 100.0
    UnitTests_total_weight_pct_used = 0.0
    UnitTest_weight_each_pct_used = 0.0

else:
    st.info("MAP not found. Using unit tests only (equal weights).")
    unit_each = 1.0 / len(unit_test_cols)
    weights = np.array([unit_each] * len(unit_test_cols), dtype=float)

    MAP_weight_pct_used = 0.0
    UnitTests_total_weight_pct_used = 100.0
    UnitTest_weight_each_pct_used = (unit_each * 100.0)

weights = weights / weights.sum()

with st.expander("Show feature weights used"):
    w_table = pd.DataFrame({"feature_used": feature_cols, "weight_%": (weights * 100).round(2)})
    st.dataframe(w_table, use_container_width=True)


# -------------------------
# Preprocess
# -------------------------
X_raw = work[feature_cols].to_numpy(dtype=float)
X_imputed = SimpleImputer(strategy="mean").fit_transform(X_raw)
Z = StandardScaler().fit_transform(X_imputed)

# Apply sqrt(weights) so KMeans distance is weighted properly
Z_w = Z * np.sqrt(weights)


# -------------------------
# KMeans clustering
# -------------------------
km = KMeans(n_clusters=k, random_state=0, n_init=10)
work["cluster"] = km.fit_predict(Z_w)


# -------------------------
# Per-student influence % (MAP vs Unit tests)
# Based on contribution to squared distance to assigned centroid
# -------------------------
centroids = km.cluster_centers_  # in weighted z-space
row_centroid = centroids[work["cluster"].to_numpy()]
diff = (Z_w - row_centroid)
feat_contrib = diff ** 2
total = feat_contrib.sum(axis=1, keepdims=True)
total[total == 0] = 1.0
feat_contrib_pct = feat_contrib / total * 100.0

contrib_df = pd.DataFrame(feat_contrib_pct, columns=[f"{c} influence %" for c in feature_cols])

if map_feature is not None:
    map_infl = contrib_df[f"{map_feature} influence %"].to_numpy()
    work["MAP_influence_%"] = np.round(map_infl, 2)
    work["UnitTests_influence_%"] = np.round(100.0 - map_infl, 2)
else:
    work["MAP_influence_%"] = 0.0
    work["UnitTests_influence_%"] = 100.0


# -------------------------
# Label clusters: Group 1 = best
# Rank clusters by average "level" (higher is better)
# -------------------------
work["_level_proxy"] = work[feature_cols].mean(axis=1, numeric_only=True)

cluster_order_best_to_worst = (
    work.groupby("cluster")["_level_proxy"]
    .mean()
    .sort_values(ascending=False)   # BEST first
    .index
    .tolist()
)

cluster_to_groupnum = {cl: i + 1 for i, cl in enumerate(cluster_order_best_to_worst)}
work["Group"] = work["cluster"].map(cluster_to_groupnum).astype(int)


# -------------------------
# Add run metadata columns
# -------------------------
work["grade"] = run_grade
work["class"] = run_class

# Add the weights as columns (so exported CSV shows the decision settings)
work["MAP_weight_%"] = round(MAP_weight_pct_used, 2)
work["UnitTests_total_weight_%"] = round(UnitTests_total_weight_pct_used, 2)
work["UnitTest_weight_each_%"] = round(UnitTest_weight_each_pct_used, 2)


# -------------------------
# Output table (NO z1/z2, NO assignment_suggestion, NO graphs)
# -------------------------
st.subheader("Results")

show_cols = [
    "grade", "class",
    student_id_col,
    "Group", "cluster",
]

# Show raw MAP + percentile (if exists)
if map_col is not None:
    show_cols.append(map_col)
if "map_percentile" in work.columns:
    show_cols.append("map_percentile")

# Unit tests
show_cols += [c for c in unit_test_cols if c in work.columns]

# Weights used + influence
show_cols += [
    "MAP_weight_%",
    "UnitTests_total_weight_%",
    "UnitTest_weight_each_%",
    "MAP_influence_%",
    "UnitTests_influence_%",
]

# Sort with Group 1 at top

out_table = work[show_cols].sort_values(["Group", student_id_col]).reset_index(drop=True)
st.dataframe(out_table, use_container_width=True)

with st.expander("Advanced: per-feature influence %"):
    st.dataframe(pd.concat([out_table, contrib_df.round(2)], axis=1), use_container_width=True)


# -------------------------
# Download
# -------------------------
download_df = pd.concat([work.drop(columns=["_level_proxy"], errors="ignore"), contrib_df.round(2)], axis=1)
st.download_button(
    "Download CSV (students_with_groups.csv)",
    data=download_df.to_csv(index=False),
    file_name="students_with_groups.csv",
    mime="text/csv"
)
