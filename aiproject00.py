import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# -------------------------
# Helpers
# -------------------------
def _reset_widget_keys(keys: list[str]) -> None:
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


def reset_app_state(keep_grade: bool = True) -> None:
    keys_to_reset = [
        "select_mode",
        "selected_scores_by_name",
        "selected_letters_input",
        "treat_map",
        "map_choice",
        "k_groups",
        "limit_group_size",
        "cap_pct",
    ]
    if not keep_grade:
        keys_to_reset.append("grade_main")
    _reset_widget_keys(keys_to_reset)
    st.session_state["current_file_sig"] = None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def excel_col_to_index(col: str) -> int:
    """
    Excel column letters -> 0-based index
    A -> 0, B -> 1, ..., Z -> 25, AA -> 26, AB -> 27, ...
    """
    col = col.strip().upper()
    if not col or not re.fullmatch(r"[A-Z]+", col):
        raise ValueError(f"Invalid column letters: '{col}'")
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def parse_excel_letters_input(text: str) -> list[int]:
    """
    Parses inputs like:
      "AA AB"
      "A:D, AA:AC"
      "b, c, z, aa"
    Returns unique 0-based indices in the order they appear.
    """
    if text is None:
        return []
    s = text.strip()
    if not s:
        return []

    tokens = re.split(r"[,\s]+", s)
    indices: list[int] = []
    seen = set()

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue

        # Range like A:D or AA:AC
        if ":" in tok:
            parts = tok.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid range token: '{tok}'")
            start = excel_col_to_index(parts[0])
            end = excel_col_to_index(parts[1])
            if start <= end:
                rng = range(start, end + 1)
            else:
                rng = range(start, end - 1, -1)
            for i in rng:
                if i not in seen:
                    indices.append(i)
                    seen.add(i)
        else:
            i = excel_col_to_index(tok)
            if i not in seen:
                indices.append(i)
                seen.add(i)

    return indices


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))


# -------------------------
# Core compute
# -------------------------
def compute_groups(
    df: pd.DataFrame,
    selected_grade: str,
    student_id_col: str,
    selected_score_cols: list[str],
    treat_map: bool,
    map_raw_col: str | None,
    k: int,
    cap_pct: int,
    weights_0to10: np.ndarray,
):
    work = df.copy()

    # If user wants one column treated as MAP percentile
    map_feature_key = None
    feature_display: list[str] = []
    feature_keys: list[str] = []

    if treat_map and map_raw_col:
        # Convert to percentile (within uploaded file)
        map_vals = pd.to_numeric(work[map_raw_col], errors="coerce")
        work["map_percentile"] = map_vals.rank(pct=True) * 100.0
        map_feature_key = "map_percentile"

    # Build model features in the same order as displayed weights
    for c in selected_score_cols:
        if treat_map and map_raw_col and c == map_raw_col:
            feature_display.append(f"{c} (MAP percentile)")
            feature_keys.append("map_percentile")
        else:
            feature_display.append(c)
            feature_keys.append(c)

    if len(feature_keys) < 1:
        raise ValueError("No usable score columns selected.")

    # Force numeric for non-map columns too (map_percentile already numeric)
    for c in feature_keys:
        if c != "map_percentile":
            work[c] = pd.to_numeric(work[c], errors="coerce")

    # Drop rows with missing values in ANY selected feature
    X = work[feature_keys].to_numpy(dtype=float)
    valid_mask = ~np.any(np.isnan(X), axis=1)

    excluded_cols = [student_id_col] + selected_score_cols
    excluded = work.loc[~valid_mask, excluded_cols].copy()
    valid_work = work.loc[valid_mask].copy()

    n_valid = len(valid_work)
    if n_valid == 0:
        raise ValueError("All students were excluded (missing selected scores).")
    if k > n_valid:
        raise ValueError(f"Groups ({k}) cannot be greater than valid students ({n_valid}).")

    # TRUE per-score weighting (0..10 each), normalized
    raw = np.array(weights_0to10, dtype=float)
    if raw.shape[0] != len(feature_keys):
        raise ValueError("Weights do not match the selected scores. Re-select columns and try again.")
    if raw.sum() <= 0:
        raise ValueError("All weights are 0. Set at least one score weight above 0.")

    raw_percent = raw * 10.0
    weights = (raw_percent / raw_percent.sum()).astype(float)  # sum=1

    weights_view = pd.DataFrame(
        {"score": feature_display, "final_weight_%": np.round(weights * 100.0, 2)}
    )

    # Standardize on valid rows only
    X_valid = valid_work[feature_keys].to_numpy(dtype=float)
    Z = StandardScaler().fit_transform(X_valid)

    # Weighted distance for KMeans (correct)
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

        # Harder-to-move students first (gap between best and 2nd best)
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
            raise ValueError("Assignment failed. Disable the cap or increase max %.")

    valid_work["_cluster_internal"] = assigned

    # Ranking: weighted standardized score proxy
    level_proxy = (Z * weights).sum(axis=1)
    valid_work["_level_proxy"] = level_proxy

    valid_work["Overall Rank"] = (
        valid_work["_level_proxy"].rank(ascending=False, method="dense").astype(int)
    )

    # Group 1 is highest mean proxy
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

    valid_work["Rank Within Group"] = (
        valid_work.groupby("Group")["_level_proxy"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    # Output columns
    return valid_work, excluded, weights_view, feature_display, map_feature_key


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Grouping Studio", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu {visibility:hidden;}
      footer {visibility:hidden;}
      header {visibility:hidden;}
      .stApp{ background: linear-gradient(180deg, #FAFAFA 0%, #F5F5F7 100%); }
      .block-container{ max-width: 1180px; padding-top: 1.1rem; padding-bottom: 2.2rem; }
      .card{
        border-radius:16px; padding:1.05rem; background:rgba(255,255,255,.88);
        border:1px solid rgba(0,0,0,.08); box-shadow:0 14px 42px rgba(0,0,0,.06);
        backdrop-filter: blur(8px); margin-bottom:.9rem;
      }
      .h{ margin:0 0 .7rem 0; font-weight:900; font-size:1.02rem; color:rgba(17,24,39,.92);}
      .muted{ color:rgba(17,24,39,.62); font-size:.92rem; }
      .tiny{ color:rgba(17,24,39,.52); font-size:.84rem; }
      .stDownloadButton button, .stButton button{ border-radius:999px !important; font-weight:900 !important; }
      code { font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='card'><div class='h'>Grouping Studio</div>"
    "<div class='muted'>Choose score columns (by name or Excel letters), set per-test weights, and get ranked groups.</div></div>",
    unsafe_allow_html=True,
)

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
    st.markdown("<div class='tiny'>Tip: Excel → Save As → CSV UTF-8.</div>", unsafe_allow_html=True)
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

df = read_csv_cached(uploaded.getvalue())
df_work = df.copy()

# Student ID column (optional)
cols_norm = {c: norm(c) for c in df_work.columns}
student_id_col = None
for c, n in cols_norm.items():
    if n in ["student_id", "student id", "id"]:
        student_id_col = c
        break
if student_id_col is None:
    df_work["student_id"] = [f"S{i+1:03d}" for i in range(len(df_work))]
    student_id_col = "student_id"

# -------------------------
# Column selection mode
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Select score columns</div>", unsafe_allow_html=True)

mode = st.radio(
    "Selection method",
    ["By column name", "By Excel letters (A, B, …, AA, AB)"],
    horizontal=True,
    key="select_mode",
)

selected_score_cols: list[str] = []

if mode == "By column name":
    selected_score_cols = st.multiselect(
        "Choose 1+ score columns",
        options=list(df_work.columns),
        default=[],
        key="selected_scores_by_name",
    )
else:
    st.markdown(
        "<div class='tiny'>Type letters separated by spaces/commas, and ranges like <code>A:D</code> or <code>AA:AC</code>.</div>",
        unsafe_allow_html=True,
    )
    letters_text = st.text_input(
        "Excel columns to include (example: AA AB or A:D, AA:AC)",
        value="",
        key="selected_letters_input",
    )
    try:
        idxs = parse_excel_letters_input(letters_text)
        if idxs:
            max_idx = len(df_work.columns) - 1
            for i in idxs:
                if i < 0 or i > max_idx:
                    raise ValueError(f"Column index out of range: {i+1} (file has {len(df_work.columns)} columns)")
            selected_score_cols = [df_work.columns[i] for i in idxs]
        else:
            selected_score_cols = []
    except Exception as e:
        st.error(str(e))
        selected_score_cols = []

# Show resolved columns
if selected_score_cols:
    st.markdown("<div class='tiny'><b>Selected:</b> " + ", ".join([f"<code>{c}</code>" for c in selected_score_cols]) + "</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

if len(selected_score_cols) < 1:
    st.info("Select at least 1 score column to continue.")
    st.stop()

# -------------------------
# Optional: treat one selected column as MAP (percentile)
# -------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>MAP percentile option (optional)</div>", unsafe_allow_html=True)

# auto-suggest a MAP-like column among selected
suggest_map = None
for c in selected_score_cols:
    n = norm(c)
    if "map" in n and "math" in n:
        suggest_map = c
        break

treat_map = st.checkbox("Treat one selected column as MAP (convert to percentile within this file)", value=(suggest_map is not None), key="treat_map")

map_raw_col = None
if treat_map:
    map_raw_col = st.selectbox(
        "Which selected column is the MAP score?",
        options=selected_score_cols,
        index=(selected_score_cols.index(suggest_map) if suggest_map in selected_score_cols else 0),
        key="map_choice",
    )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Weights (0..10 per selected score)
# -------------------------
feature_display_preview = []
for c in selected_score_cols:
    if treat_map and map_raw_col and c == map_raw_col:
        feature_display_preview.append(f"{c} (MAP percentile)")
    else:
        feature_display_preview.append(c)

num_features = len(feature_display_preview)
weights_key = f"weights_0to10_{abs(hash(tuple(feature_display_preview))) % 10**9}"

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Weights (0–10 per score)</div>", unsafe_allow_html=True)
st.markdown("<div class='tiny'>0 = off • 10 = strongest. We normalize weights automatically.</div>", unsafe_allow_html=True)

if num_features == 1:
    weights_0to10 = np.array([10], dtype=int)
    st.dataframe(pd.DataFrame({"score": feature_display_preview, "weight_0_to_10": [10]}), use_container_width=True)
else:
    default_df = pd.DataFrame({"score": feature_display_preview, "weight_0_to_10": [10] * num_features})
    w_df = st.data_editor(
        default_df,
        key=weights_key,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "score": st.column_config.TextColumn("Score", disabled=True),
            "weight_0_to_10": st.column_config.NumberColumn("Weight (0–10)", min_value=0, max_value=10, step=1),
        },
    )
    weights_0to10 = pd.to_numeric(w_df["weight_0_to_10"], errors="coerce").fillna(0).to_numpy(dtype=int)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Grouping settings
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
# Compute
# -------------------------
with st.spinner("Recalculating…"):
    try:
        valid_work, excluded, weights_view, feature_display_used, map_feature_key = compute_groups(
            df=df_work,
            selected_grade=selected_grade,
            student_id_col=student_id_col,
            selected_score_cols=selected_score_cols,
            treat_map=treat_map,
            map_raw_col=map_raw_col,
            k=k,
            cap_pct=cap_pct,
            weights_0to10=weights_0to10,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

# Weights used
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Weights used</div>", unsafe_allow_html=True)
st.dataframe(weights_view, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Results (ranked)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Results (ranked)</div>", unsafe_allow_html=True)

show_cols = ["Overall Rank", "Rank Within Group", student_id_col, "Group Name"]
for c in selected_score_cols:
    show_cols.append(c)
    if treat_map and map_raw_col and c == map_raw_col:
        show_cols.append("map_percentile")

out_table = (
    valid_work[show_cols]
    .sort_values(["Overall Rank", "Group Name", "Rank Within Group", student_id_col])
    .reset_index(drop=True)
)
st.dataframe(out_table, use_container_width=True, height=560)
st.markdown("</div>", unsafe_allow_html=True)

# Group sizes
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='h'>Group sizes</div>", unsafe_allow_html=True)
sizes = valid_work.groupby("Group Name").size().reset_index(name="students")
st.dataframe(sizes.sort_values("Group Name"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Excluded
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
    data=out_table.to_csv(index=False),
    file_name="students_with_groups_ranked.csv",
    mime="text/csv",
    use_container_width=True,
)
st.markdown("</div>", unsafe_allow_html=True)
