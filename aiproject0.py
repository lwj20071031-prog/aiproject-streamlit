import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------
# Helpers: reset UI state cleanly
# -------------------------
def _reset_widget_keys(keys: list[str]) -> None:
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

def reset_app_state(keep_grade: bool = True) -> None:
    keys_to_reset = [
        "selected_scores",
        "k_groups",
        "limit_group_size",
        "cap_pct",
        "weights_table",
    ]
    if not keep_grade:
        keys_to_reset.append("grade_main")
    _reset_widget_keys(keys_to_reset)
    st.session_state["current_file_sig"] = None

# -------------------------
# Page + Professional UI
# -------------------------
st.set_page_config(page_title="Grouping Studio", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      .stApp{ background: linear-gradient(180deg, #FAFAFA 0%, #F5F5F7 100%); }
      .block-container{ padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1180px; }

      .topbar{
        display:flex; align-items:center; justify-content:space-between; gap:1rem;
        padding: 0.95rem 1.05rem; border-radius: 16px;
        background: rgba(255,255,255,.88); border: 1px solid rgba(0,0,0,.08);
        box-shadow: 0 14px 40px rgba(0,0,0,.07); backdrop-filter: blur(10px);
        margin-bottom: 0.9rem;
      }
      .brand{display:flex; align-items:center; gap:.75rem;}
      .mark{
        width: 34px; height: 34px; border-radius: 10px;
        background: linear-gradient(135deg, #111827, #0B0B0F);
        box-shadow: 0 10px 22px rgba(0,0,0,.20);
      }
      .title{ margin:0; font-size: 1.1rem; font-weight: 900; color: rgba(17,24,39,.92); letter-spacing: .2px; }
      .subtitle{ margin:.1rem 0 0; font-size: .9rem; color: rgba(17,24,39,.60); }
      .rule{
        padding: .3rem .7rem; border-radius: 999px;
        border: 1px solid rgba(0,0,0,.10); background: rgba(255,255,255,.92);
        color: rgba(17,24,39,.86); font-weight: 800; font-size: .78rem; white-space: nowrap;
      }
      .card{
        border-radius: 16px; padding: 1.05rem 1.05rem;
        background: rgba(255,255,255,.88); border: 1px solid rgba(0,0,0,.08);
        box-shadow: 0 14px 42px rgba(0,0,0,.06); backdrop-filter: blur(8px);
        margin-bottom: .9rem;
      }
      .h{ margin: 0 0 .7rem 0; font-size: 1.02rem; font-weight: 900; color: rgba(17,24,39,.92); }
      .muted{ color: rgba(17,24,39,.62); font-size: .92rem; }
      .tiny{ color: rgba(17,24,39,.52); font-size: .84rem; }

      .stSelectbox div[data-baseweb="select"] > div,
      .stMultiSelect div[data-baseweb="select"] > div,
      .stFileUploader div{
        border-radius: 14px !important;
      }
      .stDownloadButton button, .stButton button{
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
          <div class="subtitle">Upload → select scores → set weights → group → export</div>
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
# Upload + Remove file
# -------------------------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "current_file_sig" not in st.session_state:
    st.session_state["current_file_sig"] = None

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Upload</div>", unsafe_allow_html=True)

up_c1, up_c2 = st.columns([3, 1])
with up_c1:
    uploaded_file = st.file_uploader(
        "Upload CSV (UTF-8 recommended)",
        type=["csv"],
        key=f"uploader_{st.session_state['uploader_key']}",
    )
    st.markdown("<div class='tiny'>Excel → Save As → CSV UTF-8 recommended.</div>", unsafe_allow_html=True)

with up_c2:
    if st.button("Remove file", use_container_width=True):
        st.session_state["uploader_key"] += 1
        reset_app_state(keep_grade=True)
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

file_sig = (uploaded_file.name, getattr(uploaded_file, "size", None))
if st.session_state["current_file_sig"] != file_sig:
    reset_app_state(keep_grade=True)
    st.session_state["current_file_sig"] = file_sig

df = pd.read_csv(uploaded_file)
work = df.copy()
n_students_total = len(work)

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

# -------------------------
# Score selection (DEFAULT EMPTY)
# 1-score mode still works:
# - If exactly one unit test selected => ignore MAP automatically
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
    key="selected_scores",
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
# Create MAP percentile feature if needed
# -------------------------
if use_map:
    map_vals = pd.to_numeric(work[map_col], errors="coerce")
    work["map_percentile"] = map_vals.rank(pct=True) * 100.0

# -------------------------
# Per-test weighting system (editable table)
# - Auto-normalizes to 100%
# - If only 1 score selected -> 100% automatically
# -------------------------
feature_display = []
feature_keys = []

if use_map:
    feature_display.append("MAP (percentile within uploaded file)")
    feature_keys.append("map_percentile")

for c in selected_unit_tests:
    feature_display.append(c)
    feature_keys.append(c)

num_features = len(feature_keys)
if num_features < 1:
    st.error("No usable scores selected.")
    st.stop()

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Weights (per selected score)</div>", unsafe_allow_html=True)

if num_features == 1:
    st.info("Only one score selected → it automatically becomes 100%.")
    weights = np.array([1.0], dtype=float)
    st.dataframe(pd.DataFrame({"score": feature_display, "weight_%": [100.0]}), use_container_width=True)
else:
    default_df = pd.DataFrame(
        {"score": feature_display, "weight_%": [round(100.0 / num_features, 2)] * num_features}
    )

    w_df = st.data_editor(
        default_df,
        key="weights_table",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "score": st.column_config.TextColumn("Score", disabled=True),
            "weight_%": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=1.0),
        },
    )

    w_vals = pd.to_numeric(w_df["weight_%"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    w_sum = float(w_vals.sum())
    if w_sum <= 0:
        st.error("Total weight is 0. Please set at least one weight above 0.")
        st.stop()

    w_norm = (w_vals / w_sum) * 100.0
    w_norm = np.round(w_norm, 2)
    st.caption("Weights are automatically normalized to sum to 100%.")
    st.dataframe(pd.DataFrame({"score": feature_display, "normalized_weight_%": w_norm}), use_container_width=True)

    weights = (w_norm / 100.0).astype(float)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Grouping settings
# - k starts at 0
# - Optional group size limit toggle
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Grouping settings</div>", unsafe_allow_html=True)

c1, c2 = st.columns([1, 1])
with c1:
    k = st.slider("Number of groups (1–10)", 0, 10, 0, key="k_groups")
with c2:
    limit_group_size = st.checkbox("Limit max group size", value=False, key="limit_group_size")
    cap_pct = 0
    if limit_group_size:
        cap_pct = st.slider("Max % per group", 1, 40, 20, key="cap_pct")

st.markdown("</div>", unsafe_allow_html=True)

if k == 0:
    st.info("Choose how many groups (1–10) to continue.")
    st.stop()

# -------------------------
# Build model matrix
# IMPORTANT CHANGE: NO IMPUTATION
# - We drop students with missing values in ANY selected score feature.
# -------------------------
model_features = feature_keys

# force numeric (missing becomes NaN)
for c in model_features:
    work[c] = pd.to_numeric(work[c], errors="coerce")

X = work[model_features].to_numpy(dtype=float)
valid_mask = ~np.any(np.isnan(X), axis=1)

excluded = work.loc[~valid_mask, [student_id_col] + ([map_col] if map_col else []) + selected_unit_tests].copy()
valid_work = work.loc[valid_mask].copy()

n_valid = len(valid_work)
n_excluded = len(excluded)

if n_excluded > 0:
    st.warning(f"Excluded {n_excluded} student(s) due to missing selected score(s). (No imputation is used.)")

if n_valid == 0:
    st.error("All students were excluded (missing selected scores). Upload a cleaner file or select fewer columns.")
    st.stop()

if k > n_valid:
    st.error(f"Groups ({k}) cannot be greater than valid students ({n_valid}).")
    st.stop()

X_valid = valid_work[model_features].to_numpy(dtype=float)

# Standardize (z-score) on valid rows only
Z = StandardScaler().fit_transform(X_valid)

# Weighted distance for KMeans: multiply by sqrt(weights)
Z_w = Z * np.sqrt(weights)

# -------------------------
# KMeans
# -------------------------
km = KMeans(n_clusters=k, random_state=0, n_init=10)
km.fit(Z_w)

centroids = km.cluster_centers_
dists = np.linalg.norm(Z_w[:, None, :] - centroids[None, :, :], axis=2)

# Optional cap assignment
if cap_pct == 0:
    assigned = np.argmin(dists, axis=1)
else:
    cap = int(np.ceil((cap_pct / 100.0) * n_valid))
    cap = max(cap, 1)
    if k * cap < n_valid:
        st.error(
            f"Impossible: {k} groups × max {cap}/group = {k*cap}, but valid class size is {n_valid}. "
            f"Increase max % or disable the limit."
        )
        st.stop()

    sorted_idx = np.argsort(np.sort(dists, axis=1)[:, 1] - np.sort(dists, axis=1)[:, 0])
    remaining = np.array([cap] * k, dtype=int)
    assigned = np.full(n_valid, -1, dtype=int)

    for i in sorted_idx:
        for g in np.argsort(dists[i]):
            if remaining[g] > 0:
                assigned[i] = g
                remaining[g] -= 1
                break

    if np.any(assigned == -1):
        st.error("Assignment failed unexpectedly. Try disabling the limit or increasing max %.")
        st.stop()

valid_work["_cluster_internal"] = assigned  # internal only

# Rank clusters so Group 1 = highest
level_proxy = (Z * weights).sum(axis=1)
valid_work["_level_proxy"] = level_proxy

order_best = (
    valid_work.groupby("_cluster_internal")["_level_proxy"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)
cluster_to_groupnum = {cl: i + 1 for i, cl in enumerate(order_best)}
valid_work["Group"] = valid_work["_cluster_internal"].map(cluster_to_groupnum).astype(int)
valid_work["Group Name"] = valid_work["Group"].apply(lambda g: f"{selected_grade} • Group {g}")

# Influence (MAP vs Unit tests) per student
row_centroid = centroids[assigned]
diff = (Z_w - row_centroid)
feat_contrib = diff ** 2
total = feat_contrib.sum(axis=1, keepdims=True)
total[total == 0] = 1.0
feat_contrib_pct = feat_contrib / total * 100.0

if use_map:
    map_infl = feat_contrib_pct[:, 0]
    valid_work["MAP_influence_%"] = np.round(map_infl, 2)
    valid_work["UnitTests_influence_%"] = np.round(100.0 - map_infl, 2)
else:
    valid_work["MAP_influence_%"] = 0.0
    valid_work["UnitTests_influence_%"] = 100.0

# -------------------------
# Results (valid students only)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Results</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='muted'><b>Valid students:</b> {n_valid} / {n_students_total} &nbsp;&nbsp; "
    f"<b>Groups:</b> {k} &nbsp;&nbsp; "
    f"<b>Grade:</b> {selected_grade}</div>",
    unsafe_allow_html=True,
)

show_cols = [student_id_col, "Group Name"]
if use_map:
    show_cols += [map_col, "map_percentile"]
show_cols += selected_unit_tests
show_cols += ["MAP_influence_%", "UnitTests_influence_%"]

out_table = valid_work[show_cols].sort_values(["Group Name", student_id_col]).reset_index(drop=True)
st.dataframe(out_table, use_container_width=True, height=560)
st.markdown("</div>", unsafe_allow_html=True)

# Excluded preview (optional)
if n_excluded > 0:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h'>Excluded (missing selected scores)</div>", unsafe_allow_html=True)
    st.dataframe(excluded.reset_index(drop=True), use_container_width=True, height=240)
    st.markdown("</div>", unsafe_allow_html=True)

# Group sizes
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Group sizes</div>", unsafe_allow_html=True)
sizes = valid_work.groupby("Group Name").size().reset_index(name="students")
st.dataframe(sizes.sort_values("Group Name"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Export (valid only)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Export</div>", unsafe_allow_html=True)
download_df = valid_work[show_cols].copy()
st.download_button(
    "Download CSV (valid students)",
    data=download_df.to_csv(index=False),
    file_name="students_with_groups.csv",
    mime="text/csv",
    use_container_width=True
)
st.markdown("</div>", unsafe_allow_html=True)
