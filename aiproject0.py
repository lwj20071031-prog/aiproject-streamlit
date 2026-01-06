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
        "weights_table_0to10",
    ]
    if not keep_grade:
        keys_to_reset.append("grade_main")
    _reset_widget_keys(keys_to_reset)
    st.session_state["current_file_sig"] = None

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

# Cache only the CSV read (safe). Any control change still recomputes model.
@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))

# -------------------------
# Core compute (always recalculated on UI changes)
# -------------------------
def compute_groups(
    df: pd.DataFrame,
    selected_grade: str,
    student_id_col: str,
    map_col: str | None,
    unit_test_cols: list[str],
    selected_score_cols: list[str],
    k: int,
    cap_pct: int,
    use_map: bool,
    selected_unit_tests: list[str],
    weights_0to10: np.ndarray,
):
    work = df.copy()

    # MAP percentile if used
    if use_map and map_col is not None:
        map_vals = pd.to_numeric(work[map_col], errors="coerce")
        work["map_percentile"] = map_vals.rank(pct=True) * 100.0

    # Build features in model order
    feature_display, feature_keys = [], []
    if use_map and map_col is not None:
        feature_display.append("MAP (percentile within uploaded file)")
        feature_keys.append("map_percentile")
    for c in selected_unit_tests:
        feature_display.append(c)
        feature_keys.append(c)

    # Force numeric → NaN for invalid
    for c in feature_keys:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    X = work[feature_keys].to_numpy(dtype=float)
    valid_mask = ~np.any(np.isnan(X), axis=1)

    excluded_cols = [student_id_col]
    if map_col is not None:
        excluded_cols.append(map_col)
    excluded_cols += unit_test_cols

    excluded = work.loc[~valid_mask, excluded_cols].copy()
    valid_work = work.loc[valid_mask].copy()

    n_valid = len(valid_work)
    if n_valid == 0:
        raise ValueError("All students are missing selected score(s).")

    if k > n_valid:
        raise ValueError(f"Groups ({k}) cannot be greater than valid students ({n_valid}).")

    # Weights: 0..10 → 0..100 → normalize
    raw = np.array(weights_0to10, dtype=float)
    if raw.sum() <= 0:
        raise ValueError("All weights are 0. Set at least one score weight above 0.")
    raw_percent = raw * 10.0
    weights = (raw_percent / raw_percent.sum()).astype(float)

    # Standardize on valid only
    X_valid = valid_work[feature_keys].to_numpy(dtype=float)
    Z = StandardScaler().fit_transform(X_valid)

    # Weighted KMeans (correct way)
    Z_w = Z * np.sqrt(weights)

    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(Z_w)
    centroids = km.cluster_centers_

    dists = np.linalg.norm(Z_w[:, None, :] - centroids[None, :, :], axis=2)

    # Assignment with optional cap
    if cap_pct == 0:
        assigned = np.argmin(dists, axis=1)
    else:
        cap = int(np.ceil((cap_pct / 100.0) * n_valid))
        cap = max(cap, 1)
        if k * cap < n_valid:
            raise ValueError(
                f"Impossible: {k} groups × max {cap}/group = {k*cap}, but valid class size is {n_valid}."
            )

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
            raise ValueError("Assignment failed. Disable limit or increase max %.")

    valid_work["_cluster_internal"] = assigned

    # Rank clusters → Group 1 highest
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

    # Return
    weights_view = pd.DataFrame(
        {"score": feature_display, "final_weight_%": np.round(weights * 100.0, 2)}
    )
    return valid_work, excluded, weights_view, feature_keys


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Grouping Studio", layout="wide")

# Minimal clean style
st.markdown(
    """
    <style>
      #MainMenu {visibility:hidden;}
      footer {visibility:hidden;}
      header {visibility:hidden;}
      .stApp{ background: linear-gradient(180deg, #FAFAFA 0%, #F5F5F7 100%); }
      .block-container{ max-width: 1180px; padding-top: 1.1rem; }
      .card{ border-radius:16px; padding:1.05rem; background:rgba(255,255,255,.88);
             border:1px solid rgba(0,0,0,.08); box-shadow:0 14px 42px rgba(0,0,0,.06);
             backdrop-filter: blur(8px); margin-bottom:.9rem; }
      .h{ margin:0 0 .7rem 0; font-weight:900; font-size:1.02rem; color:rgba(17,24,39,.92);}
      .muted{ color:rgba(17,24,39,.62); font-size:.92rem; }
      .tiny{ color:rgba(17,24,39,.52); font-size:.84rem; }
      .stDownloadButton button, .stButton button{ border-radius:999px !important; font-weight:900 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='card'><div class='h'>Grouping Studio</div><div class='muted'>Any change updates results automatically.</div></div>", unsafe_allow_html=True)

GRADE_CLASS_COUNT = {"JG1": 2, "SG1": 4, "Grade 2": 4, "Grade 3": 5, "Grade 4": 5, "Grade 5": 6}

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Grade</div>", unsafe_allow_html=True)
selected_grade = st.selectbox(
    "Grade",
    list(GRADE_CLASS_COUNT.keys()),
    format_func=lambda g: f"{g} ({GRADE_CLASS_COUNT[g]} classes)",
    key="grade_main",
)
st.markdown("</div>", unsafe_allow_html=True)

# Upload / remove file
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "current_file_sig" not in st.session_state:
    st.session_state["current_file_sig"] = None

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Upload</div>", unsafe_allow_html=True)
u1, u2 = st.columns([3, 1])
with u1:
    uploaded = st.file_uploader(
        "Upload CSV (UTF-8 recommended)",
        type=["csv"],
        key=f"uploader_{st.session_state['uploader_key']}",
    )
with u2:
    if st.button("Remove file", use_container_width=True):
        st.session_state["uploader_key"] += 1
        reset_app_state(keep_grade=True)
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

file_sig = (uploaded.name, getattr(uploaded, "size", None))
if st.session_state["current_file_sig"] != file_sig:
    reset_app_state(keep_grade=True)
    st.session_state["current_file_sig"] = file_sig

# Read CSV (cached by bytes)
df = read_csv_cached(uploaded.getvalue())
work = df.copy()

# Detect columns
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
    st.error("No MAP column and no 'math unit test N' columns (1..10).")
    st.stop()

# Score selection
options = []
if map_col is not None:
    options.append(map_col)
options += unit_test_cols

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Select scores</div>", unsafe_allow_html=True)
selected_score_cols = st.multiselect(
    "Select 1+ scores (works with 1 score only)",
    options=options,
    default=[],
    key="selected_scores",
)
st.markdown("</div>", unsafe_allow_html=True)

if len(selected_score_cols) < 1:
    st.info("Select at least 1 score column to continue.")
    st.stop()

use_map = (map_col is not None and map_col in selected_score_cols)
selected_unit_tests = [c for c in selected_score_cols if c != map_col]

# Weights 0..10 per selected score
feature_display = (["MAP (percentile within uploaded file)"] if use_map else []) + selected_unit_tests
num_features = len(feature_display)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Weights (0–10 per score)</div>", unsafe_allow_html=True)
st.markdown("<div class='tiny'>0=0% • 1=10% • … • 10=100% (auto-normalized across selected scores)</div>", unsafe_allow_html=True)

if num_features == 1:
    weights_0to10 = np.array([10], dtype=int)
    st.dataframe(pd.DataFrame({"score": feature_display, "weight_0_to_10": [10]}), use_container_width=True)
else:
    default_df = pd.DataFrame({"score": feature_display, "weight_0_to_10": [10] * num_features})
    w_df = st.data_editor(
        default_df,
        key="weights_table_0to10",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "score": st.column_config.TextColumn("Score", disabled=True),
            "weight_0_to_10": st.column_config.NumberColumn("Weight (0–10)", min_value=0, max_value=10, step=1),
        },
    )
    weights_0to10 = pd.to_numeric(w_df["weight_0_to_10"], errors="coerce").fillna(0).to_numpy(dtype=int)

st.markdown("</div>", unsafe_allow_html=True)

# Grouping settings
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
# ALWAYS recompute from current UI values
# -------------------------
with st.spinner("Recalculating…"):
    try:
        valid_work, excluded, weights_view, feature_keys = compute_groups(
            df=work,
            selected_grade=selected_grade,
            student_id_col=student_id_col,
            map_col=map_col,
            unit_test_cols=unit_test_cols,
            selected_score_cols=selected_score_cols,
            k=k,
            cap_pct=cap_pct,
            use_map=use_map,
            selected_unit_tests=selected_unit_tests,
            weights_0to10=weights_0to10,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

# Show weights used
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Weights used</div>", unsafe_allow_html=True)
st.dataframe(weights_view, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Results table (no stale results, always current)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Results</div>", unsafe_allow_html=True)

show_cols = [student_id_col, "Group Name"]
if use_map and map_col is not None:
    show_cols += [map_col, "map_percentile"]
show_cols += selected_unit_tests

out_table = valid_work[show_cols].sort_values(["Group Name", student_id_col]).reset_index(drop=True)
st.dataframe(out_table, use_container_width=True, height=560)
st.markdown("</div>", unsafe_allow_html=True)

# Group sizes
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Group sizes</div>", unsafe_allow_html=True)
sizes = valid_work.groupby("Group Name").size().reset_index(name="students")
st.dataframe(sizes.sort_values("Group Name"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Excluded (optional)
if len(excluded) > 0:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h'>Excluded (missing selected scores)</div>", unsafe_allow_html=True)
    st.dataframe(excluded.reset_index(drop=True), use_container_width=True, height=240)
    st.markdown("</div>", unsafe_allow_html=True)

# Export
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Export</div>", unsafe_allow_html=True)
st.download_button(
    "Download CSV (valid students)",
    data=valid_work[show_cols].to_csv(index=False),
    file_name="students_with_groups.csv",
    mime="text/csv",
    use_container_width=True,
)
st.markdown("</div>", unsafe_allow_html=True)
