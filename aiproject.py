import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------
# Page + Professional UI
# -------------------------
st.set_page_config(page_title="Grouping Studio", layout="wide")

st.markdown(
    """
    <style>
      /* Hide Streamlit chrome */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* Professional neutral background */
      .stApp{
        background: linear-gradient(180deg, #FAFAFA 0%, #F5F5F7 100%);
      }

      /* Layout */
      .block-container{
        padding-top: 1.1rem;
        padding-bottom: 2.2rem;
        max-width: 1180px;
      }

      /* Top bar */
      .topbar{
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:1rem;
        padding: 0.95rem 1.05rem;
        border-radius: 16px;
        background: rgba(255,255,255,.85);
        border: 1px solid rgba(0,0,0,.08);
        box-shadow: 0 14px 40px rgba(0,0,0,.07);
        backdrop-filter: blur(10px);
        margin-bottom: 0.9rem;
      }
      .brand{
        display:flex; align-items:center; gap:.75rem;
      }
      .mark{
        width: 34px; height: 34px; border-radius: 10px;
        background: linear-gradient(135deg, #111827, #0B0B0F);
        box-shadow: 0 10px 22px rgba(0,0,0,.20);
      }
      .title{
        margin:0;
        font-size: 1.1rem;
        font-weight: 900;
        color: rgba(17,24,39,.92);
        letter-spacing: .2px;
      }
      .subtitle{
        margin:.1rem 0 0;
        font-size: .9rem;
        color: rgba(17,24,39,.60);
      }
      .rule{
        padding: .3rem .7rem;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,.10);
        background: rgba(255,255,255,.9);
        color: rgba(17,24,39,.86);
        font-weight: 800;
        font-size: .78rem;
        white-space: nowrap;
      }

      /* Sections */
      .card{
        border-radius: 16px;
        padding: 1.05rem 1.05rem;
        background: rgba(255,255,255,.86);
        border: 1px solid rgba(0,0,0,.08);
        box-shadow: 0 14px 42px rgba(0,0,0,.06);
        backdrop-filter: blur(8px);
        margin-bottom: .9rem;
      }
      .h{
        margin: 0 0 .7rem 0;
        font-size: 1.02rem;
        font-weight: 900;
        color: rgba(17,24,39,.92);
      }
      .muted{ color: rgba(17,24,39,.62); font-size: .92rem; }
      .tiny{ color: rgba(17,24,39,.52); font-size: .84rem; }

      /* Pills */
      .pills{display:flex; flex-wrap:wrap; gap:.55rem; margin-top:.15rem;}
      .pill{
        display:flex; align-items:center; gap:.5rem;
        padding: .4rem .65rem;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,.10);
        background: rgba(255,255,255,.95);
        box-shadow: 0 10px 22px rgba(0,0,0,.04);
        font-weight: 800;
        font-size: .85rem;
        color: rgba(17,24,39,.86);
      }
      .dot{
        width: 9px; height: 9px; border-radius: 50%;
        background: #111827;
        opacity: .85;
      }

      /* Inputs */
      .stSelectbox div[data-baseweb="select"] > div,
      .stMultiSelect div[data-baseweb="select"] > div,
      .stFileUploader div{
        border-radius: 14px !important;
      }
      .stDownloadButton button{
        border-radius: 999px !important;
        padding: .65rem 1.05rem !important;
        font-weight: 900 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="mark"></div>
        <div>
          <div class="title">Grouping Studio</div>
          <div class="subtitle">Upload → select scores → configure grouping → export roster</div>
        </div>
      </div>
      <div class="rule">Group 1 = highest • MAP → percentile (0–100)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Grade (grade-only)
# -------------------------
GRADE_CLASS_COUNT = {
    "JG1": 2,
    "SG1": 4,
    "Grade 2": 4,
    "Grade 3": 5,
    "Grade 4": 5,
    "Grade 5": 6,
}

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Grade</div>", unsafe_allow_html=True)

selected_grade = st.selectbox(
    "Grade",
    list(GRADE_CLASS_COUNT.keys()),
    format_func=lambda g: f"{g} ({GRADE_CLASS_COUNT[g]} classes)",
    key="grade_main",
)
st.markdown(
    f"<div class='muted'>Classes in {selected_grade}: <b>{GRADE_CLASS_COUNT[selected_grade]}</b></div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Upload
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Upload</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV (UTF-8 recommended)", type=["csv"])
st.markdown("<div class='tiny'>Excel → Save As → CSV UTF-8 recommended.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
work = df.copy()

# -------------------------
# Column detection
# -------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

cols_norm = {c: norm(c) for c in work.columns}

student_id_col = None
for c, n in cols_norm.items():
    if n in ["student_id", "student id", "id"]:
        student_id_col = c
        break
if student_id_col is None:
    work["student_id"] = [f"S{i+1:03d}" for i in range(len(work))]
    student_id_col = "student_id"

map_candidates = [c for c, n in cols_norm.items() if ("map" in n and "math" in n)]
map_col = map_candidates[0] if map_candidates else None

unit_pairs = []
for c, n in cols_norm.items():
    m = re.search(r"math\s*unit\s*test\s*(\d+)", n)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 10:
            unit_pairs.append((num, c))
unit_test_cols = [c for _, c in sorted(unit_pairs)]

if map_col is None and len(unit_test_cols) == 0:
    st.error("No MAP column and no 'math unit test N' columns (1..10). Check your CSV headers.")
    st.stop()

n_students = len(work)

# -------------------------
# Snapshot
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Snapshot</div>", unsafe_allow_html=True)
st.markdown("<div class='pills'>", unsafe_allow_html=True)

pills = [
    f"{n_students} students",
    f"MAP: {'Found' if map_col else 'Not found'}",
    f"Unit tests detected: {len(unit_test_cols)}",
    f"{selected_grade} ({GRADE_CLASS_COUNT[selected_grade]} classes)",
]
for label in pills:
    st.markdown(f"<div class='pill'><span class='dot'></span>{label}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Score selection (DEFAULT EMPTY)
# Works with ONE score alone:
# - If exactly one unit test is selected => ignore MAP automatically
# -------------------------
options = []
if map_col is not None:
    options.append(map_col)
options += unit_test_cols

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Select scores</div>", unsafe_allow_html=True)
selected_score_cols = st.multiselect(
    "Start empty — select 1 or more score columns",
    options=options,
    default=[],
)
st.markdown("<div class='tiny'>If you select only 1 unit test, grouping uses only that test (MAP ignored).</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if len(selected_score_cols) < 1:
    st.info("Select at least 1 score column to continue.")
    st.stop()

only_one_selected = (len(selected_score_cols) == 1)
selected_only = selected_score_cols[0] if only_one_selected else None

use_map = (map_col is not None and map_col in selected_score_cols)
selected_unit_tests = [c for c in selected_score_cols if c != map_col]

if only_one_selected and (selected_only != map_col):
    use_map = False
    selected_unit_tests = [selected_only]

# -------------------------
# Controls (k default = 0)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Grouping settings</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    k = st.slider("Number of groups (1–10)", 0, 10, 0)
with c2:
    cap_pct = st.slider("Max group size (% of class)", 0, 40, 0, help="0% = no limit")
with c3:
    if use_map and len(selected_unit_tests) > 0:
        map_weight_pct = st.slider("MAP weight (%)", 0, 100, 0)
    else:
        map_weight_pct = 0
        st.caption("MAP weight appears only when MAP + unit tests are selected.")

st.markdown("</div>", unsafe_allow_html=True)

if k == 0:
    st.info("Choose how many groups (1–10) to continue.")
    st.stop()

if k > n_students:
    st.error(f"Groups ({k}) cannot be greater than students ({n_students}).")
    st.stop()

# -------------------------
# Build model features
# MAP -> percentile 0..100
# Unit tests raw
# Works with ONE feature
# -------------------------
model_features = []
if use_map:
    map_vals = pd.to_numeric(work[map_col], errors="coerce")
    work["map_percentile"] = map_vals.rank(pct=True) * 100.0
    model_features.append("map_percentile")

for c in selected_unit_tests:
    work[c] = pd.to_numeric(work[c], errors="coerce")
model_features += selected_unit_tests

if len(model_features) < 1:
    st.error("No usable numeric features found.")
    st.stop()

# Weights
if use_map and len(selected_unit_tests) > 0:
    map_w = map_weight_pct / 100.0
    remaining = 1.0 - map_w
    unit_each = remaining / len(selected_unit_tests)
    weights = np.array([map_w] + [unit_each] * len(selected_unit_tests), dtype=float)
else:
    weights = np.ones(len(model_features), dtype=float)
weights = weights / weights.sum()

# -------------------------
# Preprocess + KMeans
# -------------------------
X_raw = work[model_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
X_imputed = SimpleImputer(strategy="mean").fit_transform(X_raw)
Z = StandardScaler().fit_transform(X_imputed)
Z_w = Z * np.sqrt(weights)

km = KMeans(n_clusters=k, random_state=0, n_init=10)
km.fit(Z_w)
centroids = km.cluster_centers_

dists = np.linalg.norm(Z_w[:, None, :] - centroids[None, :, :], axis=2)

# Assignment with optional cap
if cap_pct == 0:
    assigned = np.argmin(dists, axis=1)
else:
    cap = int(np.ceil((cap_pct / 100.0) * n_students))
    cap = max(cap, 1)
    if k * cap < n_students:
        st.error(
            f"Impossible: {k} groups × max {cap}/group = {k*cap}, but class size is {n_students}. "
            f"Increase cap % or increase groups."
        )
        st.stop()

    sorted_idx = np.argsort(np.sort(dists, axis=1)[:, 1] - np.sort(dists, axis=1)[:, 0])
    remaining = np.array([cap] * k, dtype=int)
    assigned = np.full(n_students, -1, dtype=int)

    for i in sorted_idx:
        for g in np.argsort(dists[i]):
            if remaining[g] > 0:
                assigned[i] = g
                remaining[g] -= 1
                break

    if np.any(assigned == -1):
        st.error("Assignment failed unexpectedly. Try increasing cap %.")
        st.stop()

work["_cluster_internal"] = assigned  # never shown

# Rank clusters: Group 1 best
level_proxy = (Z * weights).sum(axis=1)
work["_level_proxy"] = level_proxy

order_best = (
    work.groupby("_cluster_internal")["_level_proxy"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)
cluster_to_groupnum = {cl: i + 1 for i, cl in enumerate(order_best)}
work["Group"] = work["_cluster_internal"].map(cluster_to_groupnum).astype(int)

work["Group Name"] = work["Group"].apply(lambda g: f"{selected_grade} • Group {g}")

# Influence (MAP vs Unit tests)
row_centroid = centroids[assigned]
diff = (Z_w - row_centroid)
feat_contrib = diff ** 2
total = feat_contrib.sum(axis=1, keepdims=True)
total[total == 0] = 1.0
feat_contrib_pct = feat_contrib / total * 100.0

if use_map:
    map_infl = feat_contrib_pct[:, 0]
    work["MAP_influence_%"] = np.round(map_infl, 2)
    work["UnitTests_influence_%"] = np.round(100.0 - map_infl, 2)
else:
    work["MAP_influence_%"] = 0.0
    work["UnitTests_influence_%"] = 100.0

# -------------------------
# Results
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Results</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='muted'><b>Groups:</b> {k} &nbsp;&nbsp; "
    f"<b>Selected scores:</b> {len(selected_score_cols)} &nbsp;&nbsp; "
    f"<b>Grade:</b> {selected_grade}</div>",
    unsafe_allow_html=True,
)

show_cols = [student_id_col, "Group Name"]
if use_map:
    show_cols += [map_col, "map_percentile"]
show_cols += selected_unit_tests
show_cols += ["MAP_influence_%", "UnitTests_influence_%"]

out_table = work[show_cols].sort_values(["Group Name", student_id_col]).reset_index(drop=True)
st.dataframe(out_table, use_container_width=True, height=560)
st.markdown("</div>", unsafe_allow_html=True)

# Group sizes
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Group sizes</div>", unsafe_allow_html=True)
sizes = work.groupby("Group Name").size().reset_index(name="students")
st.dataframe(sizes.sort_values("Group Name"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Export
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Export</div>", unsafe_allow_html=True)
download_df = work[show_cols].copy()
st.download_button(
    "Download CSV",
    data=download_df.to_csv(index=False),
    file_name="students_with_groups.csv",
    mime="text/csv",
    use_container_width=True
)
st.markdown("</div>", unsafe_allow_html=True)
