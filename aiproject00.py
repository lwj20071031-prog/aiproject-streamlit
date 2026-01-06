import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# -------------------------
# Style: uploader + remove button alignment
# - Hide uploader label and use one shared label above both
# - Force dropzone and Remove button to share the same min-height
# -------------------------
st.markdown(
    """
<style>
/* Shared "Upload" row visuals */
div[data-testid="stFileUploader"] {
  margin-top: 0px !important;
}

/* Dropzone fixed height */
div[data-testid="stFileUploader"] section {
  min-height: 120px !important;
}
div[data-testid="stFileUploaderDropzone"] {
  min-height: 120px !important;
  padding-top: 18px !important;
  padding-bottom: 18px !important;
  border-radius: 12px !important;
}

/* Make the Remove button container stretch and match height */
div[data-testid="stButton"] {
  height: 100% !important;
  display: flex !important;
}
div[data-testid="stButton"] > div {
  width: 100% !important;
  display: flex !important;
}
div[data-testid="stButton"] > div > button {
  width: 100% !important;
  min-height: 120px !important;
  border-radius: 12px !important;
}

/* Optional: avoid extra spacing above the button */
div[data-testid="column"] > div {
  padding-top: 0px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------
# Helpers
# -------------------------
def _reset_widget_keys(keys: list[str]) -> None:
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


def reset_app_state() -> None:
    _reset_widget_keys(
        [
            "id_letters_input",
            "selected_letters_input",
            "k_groups",
            "limit_group_size",
            "cap_pct",
        ]
    )
    st.session_state["current_file_sig"] = None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def index_to_excel_col(i: int) -> str:
    n = i + 1
    out = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def excel_col_to_index(col: str) -> int:
    col = col.strip().upper()
    if not col or not re.fullmatch(r"[A-Z]+", col):
        raise ValueError(f"Invalid column letters: '{col}'")
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def parse_excel_letters_input(text: str) -> list[int]:
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

        if ":" in tok:
            a, b = tok.split(":", 1)
            start = excel_col_to_index(a)
            end = excel_col_to_index(b)
            step = 1 if start <= end else -1
            for i in range(start, end + step, step):
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


def to_0_100(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.full_like(x, 50.0, dtype=float)
    return (x - mn) / (mx - mn) * 100.0


# -------------------------
# Core compute (BLANKS DO NOT COUNT)
# - Any student with ANY blank among selected score columns is excluded from everything.
# -------------------------
def compute_groups(
    df: pd.DataFrame,
    id_col: str,
    selected_score_cols: list[str],
    k: int,
    cap_pct: int,
    weights_0to10: np.ndarray,
):
    work = df.copy()
    feature_keys = list(selected_score_cols)

    # numeric conversion: blanks -> NaN
    for c in feature_keys:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Exclude any row with ANY missing selected score (no imputation)
    X = work[feature_keys].to_numpy(dtype=float)
    valid_mask = ~np.any(np.isnan(X), axis=1)

    excluded_cols = [id_col] + selected_score_cols
    excluded = work.loc[~valid_mask, excluded_cols].copy()
    valid_work = work.loc[valid_mask].copy()

    n_valid = len(valid_work)
    if n_valid == 0:
        raise ValueError("All students were excluded (missing selected scores).")
    if k > n_valid:
        raise ValueError(f"Number of groups ({k}) cannot exceed valid students ({n_valid}).")

    # weights (0..10 each)
    raw = np.array(weights_0to10, dtype=float)
    if raw.shape[0] != len(feature_keys):
        raise ValueError("Weights do not match the selected score columns.")
    if raw.sum() <= 0:
        raise ValueError("All weights are 0. Set at least one weight above 0.")

    weights = (raw / raw.sum()).astype(float)
    weights_view = pd.DataFrame(
        {"score": feature_keys, "final_weight_%": np.round(weights * 100.0, 2)}
    )

    # Standardize (valid-only)
    X_valid = valid_work[feature_keys].to_numpy(dtype=float)
    Z = StandardScaler().fit_transform(X_valid)

    # Weighted KMeans space
    Z_w = Z * np.sqrt(weights)

    # Fit KMeans
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Z_w)

    centroids = km.cluster_centers_
    dists = np.linalg.norm(Z_w[:, None, :] - centroids[None, :, :], axis=2)

    cap_adjust_note = None

    if cap_pct == 0:
        assigned = np.argmin(dists, axis=1)
    else:
        cap = int(np.ceil((cap_pct / 100.0) * n_valid))
        cap = max(cap, 1)

        # minimum cap needed to fit everyone
        min_cap = int(np.ceil(n_valid / k))
        if cap < min_cap:
            cap = min_cap
            cap_adjust_note = (
                f"Max % per group was too small to fit all valid students. "
                f"Auto-adjusted to at least {int(np.ceil(100 * cap / n_valid))}%."
            )

        # cap assignment
        sorted_d = np.sort(dists, axis=1)
        margin = sorted_d[:, 1] - sorted_d[:, 0]
        sorted_idx = np.argsort(-margin)

        remaining = np.array([cap] * k, dtype=int)
        assigned = np.full(n_valid, -1, dtype=int)

        for i in sorted_idx:
            for g in np.argsort(dists[i]):
                if remaining[g] > 0:
                    assigned[i] = g
                    remaining[g] -= 1
                    break

        if np.any(assigned == -1):
            raise ValueError("Assignment failed. Disable cap or increase max %.")

    valid_work["_cluster_internal"] = assigned

    # Overall Score (0..100)
    level_proxy = (Z * weights).sum(axis=1)
    valid_work["_level_proxy"] = level_proxy
    valid_work["Overall Score"] = np.round(to_0_100(level_proxy), 2)

    # Group 1 = best cluster
    order_best = (
        valid_work.groupby("_cluster_internal")["_level_proxy"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    cluster_to_groupnum = {cl: i + 1 for i, cl in enumerate(order_best)}
    valid_work["Group"] = valid_work["_cluster_internal"].map(cluster_to_groupnum).astype(int)
    valid_work["Group Name"] = valid_work["Group"].apply(lambda g: f"Group {g}")

    return valid_work, excluded, weights_view, cap_adjust_note


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="WAB Classroom Assignment Program", layout="wide")
st.title("WAB Classroom Assignment Program")

# Upload / remove file (aligned row)
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "current_file_sig" not in st.session_state:
    st.session_state["current_file_sig"] = None

st.subheader("Upload")  # shared label so both elements align perfectly

c1, c2 = st.columns([4, 1], vertical_alignment="top")
with c1:
    uploaded = st.file_uploader(
        label="",
        label_visibility="collapsed",
        type=["csv"],
        key=f"uploader_{st.session_state['uploader_key']}",
    )
with c2:
    if st.button("Remove file", use_container_width=True):
        st.session_state["uploader_key"] += 1
        reset_app_state()
        st.rerun()

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

file_sig = (uploaded.name, getattr(uploaded, "size", None))
if st.session_state["current_file_sig"] != file_sig:
    reset_app_state()
    st.session_state["current_file_sig"] = file_sig

df = read_csv_cached(uploaded.getvalue())
df_work = df.copy()

# Student name column (Excel letter)
st.subheader("Student name column (Excel letter)")
st.caption("Type ONE Excel letter for the student name column. Example: A  OR  AB")

id_letters = st.text_input(
    "Student name column",
    value="",
    key="id_letters_input",
)

id_col = None
if id_letters.strip():
    idxs = parse_excel_letters_input(id_letters)
    if len(idxs) != 1:
        st.error("Please type exactly ONE column letter for the student name column (example: A or AB).")
        st.stop()
    idx = idxs[0]
    if idx < 0 or idx >= len(df_work.columns):
        st.error(f"Column out of range: {index_to_excel_col(idx)} (file has {len(df_work.columns)} columns)")
        st.stop()
    id_col = df_work.columns[idx]
    st.dataframe(
        pd.DataFrame({"Excel": [index_to_excel_col(idx)], "Column title": [id_col]}),
        width="stretch",
        height=110,
    )
else:
    # fallback: try common names; if none, create student_id
    cols_norm = {c: norm(c) for c in df_work.columns}
    id_col = next((c for c, n in cols_norm.items() if n in ["student_id", "student id", "id", "name", "student name"]), None)
    if id_col is None:
        df_work["student_id"] = [f"S{i+1:03d}" for i in range(len(df_work))]
        id_col = "student_id"
        st.info("No student name column selected — created a 'student_id' column automatically.")

# Scores selection (Excel letters)
st.subheader("Select score columns (Excel letters)")
st.caption("Examples: A, B, C  OR  A B C")

letters_text = st.text_input(
    "Type Excel letters for the score columns to include",
    value="",
    key="selected_letters_input",
)

selected_score_cols: list[str] = []
try:
    idxs = parse_excel_letters_input(letters_text)
    if idxs:
        max_idx = len(df_work.columns) - 1
        for i in idxs:
            if i < 0 or i > max_idx:
                raise ValueError(
                    f"Column out of range: {index_to_excel_col(i)} (file has {len(df_work.columns)} columns)"
                )
        selected_score_cols = [df_work.columns[i] for i in idxs]

        # prevent ID col as score col
        if id_col in selected_score_cols:
            selected_score_cols = [c for c in selected_score_cols if c != id_col]
            st.warning("You included the student name column in the score list — it was removed automatically.")

        st.dataframe(
            pd.DataFrame(
                {"Excel": [index_to_excel_col(i) for i in idxs], "Column title": [df_work.columns[i] for i in idxs]}
            ),
            width="stretch",
            height=210,
        )
except Exception as e:
    st.error(str(e))

if not selected_score_cols:
    st.info("Type at least 1 Excel column letter to continue.")
    st.stop()

# Weights per selected score
st.subheader("Weights (0–10 per selected score)")
st.caption("0 = off • 10 = strongest (normalized automatically)")

weights_key = f"weights_0to10_{abs(hash(tuple(selected_score_cols))) % 10**9}"

if len(selected_score_cols) == 1:
    weights_0to10 = np.array([10], dtype=int)
    st.dataframe(
        pd.DataFrame({"score": selected_score_cols, "weight_0_to_10": [10]}),
        width="stretch",
        height=110,
    )
else:
    default_df = pd.DataFrame({"score": selected_score_cols, "weight_0_to_10": [10] * len(selected_score_cols)})
    w_df = st.data_editor(
        default_df,
        key=weights_key,
        width="stretch",
        num_rows="fixed",
        column_config={
            "score": st.column_config.TextColumn("Score", disabled=True),
            "weight_0_to_10": st.column_config.NumberColumn("Weight (0–10)", min_value=0, max_value=10, step=1),
        },
    )
    weights_0to10 = pd.to_numeric(w_df["weight_0_to_10"], errors="coerce").fillna(0).to_numpy(dtype=int)

# Grouping
st.subheader("Grouping settings")
k = st.slider("Number of groups (1–10)", 1, 10, 3, key="k_groups")
limit_group_size = st.checkbox("Limit max group size", value=False, key="limit_group_size")
cap_pct = st.slider("Max % per group", 1, 40, 20, key="cap_pct") if limit_group_size else 0

try:
    with st.spinner("Recalculating…"):
        valid_work, excluded, weights_view, cap_note = compute_groups(
            df=df_work,
            id_col=id_col,
            selected_score_cols=selected_score_cols,
            k=k,
            cap_pct=cap_pct,
            weights_0to10=weights_0to10,
        )
except Exception as e:
    st.error(str(e))
    st.stop()

if cap_note:
    st.warning(cap_note)

st.subheader("Weights used")
st.dataframe(weights_view, width="stretch")

st.subheader("Results")
show_cols = ["Overall Score", id_col, "Group Name"] + selected_score_cols
out_table = (
    valid_work[show_cols]
    .sort_values(["Overall Score", "Group Name", id_col], ascending=[False, True, True])
    .reset_index(drop=True)
)
st.dataframe(out_table, width="stretch", height=560)

st.subheader("Export")
st.download_button(
    "Download CSV (valid students)",
    data=out_table.to_csv(index=False),
    file_name="students_with_groups_overall_score.csv",
    mime="text/csv",
    use_container_width=True,
)
