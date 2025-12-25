import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

APP_VERSION = "v5 (grade label + test selector + group cap + no cluster shown)"

st.set_page_config(page_title="Student Grouping", layout="wide")
st.title("Student Grouping")
st.caption(f"App version: {APP_VERSION}")

# -------------------------
# Grade picker (used in group names)
# -------------------------
GRADE_OPTIONS = ["JG1", "SG1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]
st.sidebar.header("Grade")
selected_grade = st.sidebar.selectbox("Select grade", GRADE_OPTIONS)

# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader("Upload CSV (UTF-8 recommended)", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
work = df.copy()

# -------------------------
# Helpers: column detection
# -------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

cols_norm = {c: norm(c) for c in work.columns}

# student id (optional)
student_id_col = None
for c, n in cols_norm.items():
    if n in ["student_id", "student id", "id"]:
        student_id_col = c
        break

if student_id_col is None:
    work["student_id"] = [f"S{i+1:03d}" for i in range(len(work))]
    student_id_col = "student_id"

# MAP column: contains both "map" and "math"
map_candidates = [c for c, n in cols_norm.items() if ("map" in n and "math" in n)]
map_col = map_candidates[0] if map_candidates else None

# unit tests 1..10: "math unit test 1"... etc
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
    st.error("I couldn't find a MAP column or any 'math unit test N' columns (1..10). Check your headers.")
    st.stop()

# -------------------------
# Let user choose which tests to use
# -------------------------
st.subheader("Select which score columns to use")
options = []
if map_col is not None:
    options.append(map_col)
options += unit_test_cols

selected_score_cols = st.multiselect(
    "Choose score columns (numeric only)",
    options=options,
    default=options
)

if len(selected_score_cols) < 2:
    st.error("Select at least 2 score columns (e.g., MAP + one unit test).")
    st.stop()

use_map = (map_col is not None and map_col in selected_score_cols)
selected_unit_tests = [c for c in selected_score_cols if c != map_col]

# -------------------------
# Build model features
# - MAP -> percentile (0..100)
# - Unit tests remain raw
# -------------------------
model_features = []

if use_map:
    map_vals = pd.to_numeric(work[map_col], errors="coerce")
    work["map_percentile"] = map_vals.rank(pct=True) * 100.0
    model_features.append("map_percentile")

for c in selected_unit_tests:
    work[c] = pd.to_numeric(work[c], errors="coerce")
model_features += selected_unit_tests

if len(model_features) < 2:
    st.error("Not enough numeric features after processing.")
    st.stop()

# -------------------------
# Controls: number of groups + max group size cap (%)
# -------------------------
st.subheader("Grouping controls")
n_students = len(work)

k = st.slider("How many groups?", min_value=2, max_value=10, value=3)

cap_pct = st.slider(
    "Max size per group (% of class). 0% = no limit. (up to 40%)",
    min_value=0, max_value=40, value=0
)

# -------------------------
# Weights: MAP vs Unit Tests
# (If MAP is selected, user controls MAP %, rest split across unit tests)
# -------------------------
st.subheader("Weights (decision influence)")

if use_map and len(selected_unit_tests) > 0:
    map_weight_pct = st.slider("MAP weight (%)", 0, 100, 60)
    map_w = map_weight_pct / 100.0
    remaining = 1.0 - map_w
    unit_each = remaining / len(selected_unit_tests)
    weights = np.array([map_w] + [unit_each] * len(selected_unit_tests), dtype=float)
else:
    # either MAP-only, or unit-tests-only
    weights = np.ones(len(model_features), dtype=float)

weights = weights / weights.sum()

with st.expander("Show feature weights"):
    st.dataframe(
        pd.DataFrame({"feature_used": model_features, "weight_%": (weights * 100).round(2)}),
        use_container_width=True
    )

# -------------------------
# Preprocess: impute + standardize + apply weights
# -------------------------
X_raw = work[model_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
X_imputed = SimpleImputer(strategy="mean").fit_transform(X_raw)
Z = StandardScaler().fit_transform(X_imputed)

# Weighted space for KMeans distances
Z_w = Z * np.sqrt(weights)

# -------------------------
# KMeans (fit centroids)
# -------------------------
km = KMeans(n_clusters=k, random_state=0, n_init=10)
km.fit(Z_w)
centroids = km.cluster_centers_

# Distances: n x k
dists = np.linalg.norm(Z_w[:, None, :] - centroids[None, :, :], axis=2)

# -------------------------
# Optional: capacity-constrained assignment
# cap_pct = 0 -> normal nearest centroid
# cap_pct > 0 -> greedy assignment with capacity per group
# -------------------------
if cap_pct == 0:
    assigned = np.argmin(dists, axis=1)
else:
    cap = int(np.ceil((cap_pct / 100.0) * n_students))
    cap = max(cap, 1)

    if k * cap < n_students:
        st.error(
            f"Impossible settings: {k} groups × max {cap} students/group = {k*cap}, "
            f"but class size is {n_students}. Increase the % cap or increase number of groups."
        )
        st.stop()

    # "Hard" students first: small margin between 1st and 2nd choice
    sorted_idx = np.argsort(np.sort(dists, axis=1)[:, 1] - np.sort(dists, axis=1)[:, 0])

    remaining = np.array([cap] * k, dtype=int)
    assigned = np.full(n_students, -1, dtype=int)

    for i in sorted_idx:
        for g in np.argsort(dists[i]):  # try nearest groups first
            if remaining[g] > 0:
                assigned[i] = g
                remaining[g] -= 1
                break

    if np.any(assigned == -1):
        st.error("Unexpected assignment failure (shouldn't happen). Try raising the cap %.")
        st.stop()

work["_cluster_internal"] = assigned  # internal only (not shown)

# -------------------------
# Group ranking: Group 1 = best
# Use weighted standardized proxy (higher = better)
# -------------------------
level_proxy = (Z * weights).sum(axis=1)  # weighted standardized score
work["_level_proxy"] = level_proxy

group_means = (
    work.groupby("_cluster_internal")["_level_proxy"]
    .mean()
    .sort_values(ascending=False)  # best first
)

cluster_to_groupnum = {cl: i + 1 for i, cl in enumerate(group_means.index.tolist())}
work["Group"] = work["_cluster_internal"].map(cluster_to_groupnum).astype(int)
work["Group Name"] = work["Group"].apply(lambda g: f"{selected_grade} Group {g}")

# -------------------------
# Influence %: MAP vs Unit tests for each student (distance contribution)
# -------------------------
row_centroid = centroids[assigned]
diff = (Z_w - row_centroid)
feat_contrib = diff ** 2
total = feat_contrib.sum(axis=1, keepdims=True)
total[total == 0] = 1.0
feat_contrib_pct = feat_contrib / total * 100.0

contrib_df = pd.DataFrame(feat_contrib_pct, columns=[f"{c} influence %" for c in model_features])

if use_map:
    map_infl = contrib_df["map_percentile influence %"].to_numpy()
    work["MAP_influence_%"] = np.round(map_infl, 2)
    work["UnitTests_influence_%"] = np.round(100.0 - map_infl, 2)
else:
    work["MAP_influence_%"] = 0.0
    work["UnitTests_influence_%"] = 100.0

# -------------------------
# Output table (NO cluster shown)
# -------------------------
st.subheader("Results")

show_cols = [student_id_col, "Group Name"]

if use_map:
    show_cols += [map_col, "map_percentile"]

show_cols += selected_unit_tests

show_cols += ["MAP_influence_%", "UnitTests_influence_%"]

out_table = work[show_cols].sort_values(["Group Name", student_id_col]).reset_index(drop=True)
st.dataframe(out_table, use_container_width=True)

# Group size summary
st.subheader("Group sizes")
sizes = work.groupby("Group Name").size().reset_index(name="students")
st.dataframe(sizes.sort_values("Group Name"), use_container_width=True)

# Download
download_df = pd.concat(
    [work.drop(columns=["_cluster_internal", "_level_proxy"], errors="ignore"), contrib_df.round(2)],
    axis=1
)

st.download_button(
    "Download CSV",
    data=download_df.to_csv(index=False),
    file_name="students_with_groups.csv",
    mime="text/csv"
)
