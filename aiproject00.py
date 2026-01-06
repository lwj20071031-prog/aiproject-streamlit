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
        "selected_letters_input",
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


def index_to_excel_col(i: int) -> str:
    if i < 0:
        raise ValueError("Index must be >= 0")
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
    """
    Supports:
      "A, B, C"
      "A B C"
      "AA AB"
      "A:D" / "AA:AC"
      "A:D, AA:AC"
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

        if ":" in tok:
            parts = tok.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid range token: '{tok}'")
            start = excel_col_to_index(parts[0])
            end = excel_col_to_index(parts[1])
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
# Core compute
# -------------------------
def compute_groups(
    df: pd.DataFrame,
    selected_grade: str,
    student_id_col: str,
    selected_score_cols: list[str],
    k: int,
    cap_pct: int,
    weights_0to10: np.ndarray,
):
    work = df.copy()
    feature_keys = list(selected_score_cols)

    for c in feature_keys:
        work[c] = pd.to_numeric(work[c], errors="coerce")

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

    raw = np.array(weights_0to10, dtype=float)
    if raw.shape[0] != len(feature_keys):
        raise ValueError("Weights do not match selected scores. Re-select columns and try again.")
    if raw.sum() <= 0:
        raise ValueError("All weights are 0. Set at least one weight above 0.")

    weights = (raw / raw.sum()).astype(float)
    weights_view = pd.DataFrame(
        {"score": feature_keys, "final_weight_%": np.round(weights * 100.0, 2)}
    )

    X_valid = valid_work[feature_keys].to_numpy(dtype=float)
    Z = StandardScaler().fit_transform(X_valid)
    Z_w = Z * np.sqrt(weights)

    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Z_w)
    centroids = km.cluster_centers_
    dists = np.linalg.norm(Z_w[:, None, :] - centroids[None, :, :], axis=2)

    if cap_pct == 0:
        assigned = np.argmin(dists, axis=1)
    else:
        cap = int(np.ceil((cap_pct / 100.0) * n_valid))
        cap = max(cap, 1)
        if k * cap < n_valid:
            raise ValueError(
                f"Impossible: {k} groups × max {cap}/group = {k*cap}, but valid class size is {n_valid}."
            )

        sorted_d = np.sort(dists, axis=1)
        margin = sorted_d[:, 1] - sorted_d[:, 0]
        sorted_idx = np.argsort(-margin)  # easiest first

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

    level_proxy = (Z * weights).sum(axis=1)
    valid_work["_level_proxy"] = level_proxy
    valid_work["Overall Score"] = np.round(to_0_100(level_proxy), 2)

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

    return valid_work, excluded, weights_view


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="WAB Classroom Assignment Program", layout="wide")
st.title("WAB Classroom Assignment Program")

GRADE_CLASS_COUNT = {"JG1": 2, "SG1": 4, "Grade 2": 4, "Grade 3": 5, "Grade 4": 5, "Grade 5": 6}
selected_grade = st.selectbox(
    "Grade",
    list(GRADE_CLASS_COUNT.keys()),
    format_func=lambda g: f"{g} ({GRADE_CLASS_COUNT[g]} classes)",
    key="grade_main",
)

# Upload / remove file
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "current_file_sig" not in st.session_state:
    st.session_state["current_file_sig"] = None

c1, c2 = st.columns([3, 1])
with c1:
    uploaded = st.file_uploader(
        "Upload CSV (UTF-8 recommended)",
        type=["csv"],
        key=f"uploader_{st.session_state['uploader_key']}",
    )
with c2:
    if st.button("Remove file", use_container_width=True):
        st.session_state["uploader_key"] += 1
        reset_app_state(keep_grade=True)
        st.rerun()

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

file_sig = (uploaded.name, getattr(uploaded, "size", None))
if st.session_state["current_file_sig"] != file_sig:
    reset_app_state(keep_grade=True)
    st.session_state["current_file_sig"] = file_sig

df = read_csv_cached(uploaded.getvalue())
df_work = df.copy()

# Ensure student_id
cols_norm = {c: norm(c) for c in df_work.columns}
student_id_col = next((c for c, n in cols_norm.items() if n in ["student_id", "student id", "id"]), None)
if student_id_col is None:
    df_work["student_id"] = [f"S{i+1:03d}" for i in range(len(df_work))]
    student_id_col = "student_id"

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
        st.dataframe(
            pd.DataFrame(
                {"Excel": [index_to_excel_col(i) for i in idxs], "Column title": selected_score_cols}
            ),
            use_container_width=True,
            height=210,
        )
except Exception as e:
    st.error(str(e))

if not selected_score_cols:
    st.info("Type at least 1 Excel column letter to continue.")
    st.stop()

st.subheader("Weights (0–10 per selected score)")
st.caption("0 = off • 10 = strongest (normalized automatically)")

weights_key = f"weights_0to10_{abs(hash(tuple(selected_score_cols))) % 10**9}"

if len(selected_score_cols) == 1:
    weights_0to10 = np.array([10], dtype=int)
    st.dataframe(
        pd.DataFrame({"score": selected_score_cols, "weight_0_to_10": [10]}),
        use_container_width=True,
    )
else:
    default_df = pd.DataFrame({"score": selected_score_cols, "weight_0_to_10": [10] * len(selected_score_cols)})
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

st.subheader("Grouping settings")
k = st.slider("Number of groups (1–10)", 1, 10, 3, key="k_groups")
limit_group_size = st.checkbox("Limit max group size", value=False, key="limit_group_size")
cap_pct = st.slider("Max % per group", 1, 40, 20, key="cap_pct") if limit_group_size else 0

with st.spinner("Recalculating…"):
    valid_work, excluded, weights_view = compute_groups(
        df=df_work,
        selected_grade=selected_grade,
        student_id_col=student_id_col,
        selected_score_cols=selected_score_cols,
        k=k,
        cap_pct=cap_pct,
        weights_0to10=weights_0to10,
    )

st.subheader("Weights used")
st.dataframe(weights_view, use_container_width=True)

st.subheader("Results")
show_cols = ["Overall Score", student_id_col, "Group Name"] + selected_score_cols
out_table = (
    valid_work[show_cols]
    .sort_values(["Overall Score", "Group Name", student_id_col], ascending=[False, True, True])
    .reset_index(drop=True)
)
st.dataframe(out_table, use_container_width=True, height=560)

if len(excluded) > 0:
    st.subheader("Excluded (missing selected scores)")
    st.dataframe(excluded.reset_index(drop=True), use_container_width=True, height=240)

st.subheader("Export")
st.download_button(
    "Download CSV (valid students)",
    data=out_table.to_csv(index=False),
    file_name="students_with_groups_overall_score.csv",
    mime="text/csv",
    use_container_width=True,
)
