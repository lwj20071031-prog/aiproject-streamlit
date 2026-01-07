import re
import streamlit as st
import pandas as pd
import numpy as np


# -------------------------
# Excel letter helpers
# -------------------------
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


# -------------------------
# Robust CSV reader (AUTO header detection)
# Does NOT delete/drop any columns.
# -------------------------
def _header_score(row: list[str]) -> float:
    cells = [("" if x is None else str(x).strip()) for x in row]
    nonempty = [c for c in cells if c != "" and c.lower() != "nan"]
    if len(nonempty) == 0:
        return -1.0

    uniq = len(set([c.lower() for c in nonempty]))
    score = len(nonempty) + 0.5 * uniq

    numeric_like = 0
    for c in nonempty:
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", c):
            numeric_like += 1
    if numeric_like / max(len(nonempty), 1) > 0.6:
        score -= 2.0

    return score


@st.cache_data(show_spinner=False)
def read_csv_smart(file_bytes: bytes) -> pd.DataFrame:
    preview = None
    used_enc = None

    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            preview = pd.read_csv(
                pd.io.common.BytesIO(file_bytes),
                header=None,
                nrows=40,
                dtype=str,
                encoding=enc,
                engine="python",
            )
            used_enc = enc
            break
        except Exception:
            continue

    if preview is None:
        preview = pd.read_csv(
            pd.io.common.BytesIO(file_bytes),
            header=None,
            nrows=40,
            dtype=str,
            engine="python",
        )
        used_enc = "utf-8"

    best_i = 0
    best_score = -1e9
    for i in range(len(preview)):
        row = preview.iloc[i].tolist()
        sc = _header_score(row)
        if sc > best_score:
            best_score = sc
            best_i = i

    df = pd.read_csv(
        pd.io.common.BytesIO(file_bytes),
        header=best_i,
        encoding=used_enc,
        engine="python",
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df


# -------------------------
# Core math:
# 1) For each selected test: normalize that test to 0..100 (within this file) using min-max, ignoring blanks
# 2) For each student: compute weighted average across available tests (weights re-normalized per student)
# 3) Normalize the composite to 0..100 within this file (relative ranking score)
# -------------------------
def minmax_0_100_by_column(X: np.ndarray) -> np.ndarray:
    """
    Column-wise min-max scaling to 0..100 ignoring NaN.
    If a column has no variance (max == min), all non-missing become 50.
    """
    S = np.full_like(X, np.nan, dtype=float)
    d = X.shape[1]
    for j in range(d):
        col = X[:, j]
        m = ~np.isnan(col)
        if m.sum() == 0:
            continue
        mn = float(np.min(col[m]))
        mx = float(np.max(col[m]))
        if mx == mn:
            S[m, j] = 50.0
        else:
            S[m, j] = (col[m] - mn) / (mx - mn) * 100.0
    return S


def normalize_0_100_minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    mn = np.nanmin(v)
    mx = np.nanmax(v)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.full_like(v, 50.0, dtype=float)
    return (v - mn) / (mx - mn) * 100.0


def masked_weighted_dist2(S: np.ndarray, mask: np.ndarray, C: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted squared distance for KMeans, ignoring missing dims per student.
    We normalize by sum of weights over available dims.
    """
    n, d = S.shape
    k = C.shape[0]
    dist2 = np.full((n, k), np.inf, dtype=float)
    w = np.clip(w.astype(float), 0.0, None)

    for i in range(n):
        m = mask[i]
        wsum = float(np.sum(w[m]))
        if wsum <= 0:
            continue
        for g in range(k):
            diff = S[i, m] - C[g, m]
            dist2[i, g] = float(np.sum(w[m] * (diff ** 2)) / wsum)

    return dist2


def masked_kmeans(S: np.ndarray, mask: np.ndarray, weights: np.ndarray, k: int, max_iter: int = 50, seed: int = 0):
    """
    Simple KMeans variant that handles missing values by ignoring missing dimensions.
    Uses weighted distances.
    """
    rng = np.random.default_rng(seed)
    n, d = S.shape

    init_idx = rng.choice(n, size=k, replace=(n < k))
    C = np.zeros((k, d), dtype=float)
    for g, idx in enumerate(init_idx):
        m = mask[idx]
        C[g, m] = S[idx, m]

    labels = np.full(n, -1, dtype=int)

    for _ in range(max_iter):
        dist2 = masked_weighted_dist2(S, mask, C, weights)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # update centroids
        for g in range(k):
            idxs = np.where(labels == g)[0]
            if len(idxs) == 0:
                ridx = rng.integers(0, n)
                C[g, :] = 0.0
                m = mask[ridx]
                C[g, m] = S[ridx, m]
                continue

            for j in range(d):
                mj = mask[idxs, j]
                if np.any(mj):
                    C[g, j] = float(np.mean(S[idxs[mj], j]))
                else:
                    C[g, j] = 0.0

    return labels, C


def compute_groups(df, id_col, score_cols, k, cap_pct, weights_0to10):
    work = df.copy()

    # numeric conversion
    for c in score_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    X = work[score_cols].to_numpy(dtype=float)

    # 1) per-test normalize to 0..100 (handles “scores up to 200” etc.)
    S = minmax_0_100_by_column(X)  # NxD, NaN where missing

    raw = np.array(weights_0to10, dtype=float)
    if raw.shape[0] != len(score_cols):
        raise ValueError("Weights do not match selected score columns.")
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        raise ValueError("All weights are 0. Set at least one weight above 0.")

    weights = (raw / raw.sum()).astype(float)
    weights_view = pd.DataFrame({"score": score_cols, "final_weight_%": np.round(weights * 100.0, 2)})

    # valid student = has at least one available test with weight > 0
    mask = ~np.isnan(S)
    usable_w = (weights > 0)
    row_has_any = np.any(mask[:, usable_w], axis=1) if np.any(usable_w) else np.zeros(len(work), dtype=bool)

    excluded = work.loc[~row_has_any, [id_col] + score_cols].copy()
    valid_work = work.loc[row_has_any].copy()

    S_valid = S[row_has_any]
    mask_valid = mask[row_has_any]
    n_valid = len(valid_work)

    if n_valid == 0:
        raise ValueError("No students have any usable selected scores.")
    if k > n_valid:
        raise ValueError(f"Number of groups ({k}) cannot exceed valid students ({n_valid}).")

    # 2) composite per student (whole bundle), weights re-normalized per student over available tests
    composite = np.full(n_valid, np.nan, dtype=float)
    for i in range(n_valid):
        m = mask_valid[i] & (weights > 0)
        wsum = float(np.sum(weights[m]))
        if wsum <= 0:
            continue
        wrow = weights[m] / wsum
        composite[i] = float(np.sum(wrow * S_valid[i, m]))

    # 3) overall score = relative within file/class using min-max on composite
    overall_score = normalize_0_100_minmax(composite)
    valid_work["Overall Score"] = np.round(overall_score, 2)
    valid_work["_level_proxy"] = overall_score  # used for group ranking

    # clustering on the same normalized-per-test space (S_valid)
    labels, C = masked_kmeans(S_valid, mask_valid, weights, k=k, max_iter=50, seed=0)
    dist2 = masked_weighted_dist2(S_valid, mask_valid, C, weights)
    dists = np.sqrt(np.maximum(dist2, 0.0))

    cap_note = None
    if cap_pct == 0:
        assigned = labels
    else:
        cap = int(np.ceil((cap_pct / 100.0) * n_valid))
        cap = max(cap, 1)

        min_cap = int(np.ceil(n_valid / k))
        if cap < min_cap:
            cap = min_cap
            cap_note = (
                f"Max % per group was too small to fit all valid students. "
                f"Auto-adjusted to at least {int(np.ceil(100 * cap / n_valid))}%."
            )

        sorted_d = np.sort(dists, axis=1)
        margin = sorted_d[:, 1] - sorted_d[:, 0] if k >= 2 else np.full(n_valid, 0.0)
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

    # Group numbering: Group 1 = best mean Overall Score
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

    return valid_work, excluded, weights_view, cap_note


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="WAB Classroom Assignment Program", layout="wide")
st.title("WAB Classroom Assignment Program")

uploaded = st.file_uploader("Upload CSV (UTF-8 recommended)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
file_sig = (len(file_bytes), hash(file_bytes[:5000]))

# Reset widgets when file changes (so it doesn't keep old selections)
if st.session_state.get("_file_sig") != file_sig:
    st.session_state["_file_sig"] = file_sig
    for kk in list(st.session_state.keys()):
        if kk.startswith(("id_letters_input", "selected_letters_input", "k_groups", "limit_group_size", "cap_pct", "weights_")):
            st.session_state.pop(kk, None)

df = read_csv_smart(file_bytes)

# Student name column by Excel letter
st.subheader("Student name column (Excel letter)")
st.caption("Type ONE Excel letter for the student name column. Example: A or AB")
id_letters = st.text_input("Student name column", value="", key="id_letters_input")

if id_letters.strip():
    idxs = parse_excel_letters_input(id_letters)
    if len(idxs) != 1:
        st.error("Please type exactly ONE column letter for the student name column.")
        st.stop()
    idx = idxs[0]
    if idx < 0 or idx >= len(df.columns):
        st.error(f"Column out of range: {index_to_excel_col(idx)} (file has {len(df.columns)} columns)")
        st.stop()
    id_col = df.columns[idx]
    st.dataframe(
        pd.DataFrame({"Excel": [index_to_excel_col(idx)], "Column title": [id_col]}),
        width="stretch",
        height=110,
    )
else:
    # fallback: create an id column silently
    df = df.copy()
    df["student_id"] = [f"S{i+1:03d}" for i in range(len(df))]
    id_col = "student_id"

# Score columns selection by Excel letters
st.subheader("Select score columns (Excel letters)")
st.caption("Examples: A, B, C  OR  A B C")
letters_text = st.text_input("Type Excel letters for the score columns to include", value="", key="selected_letters_input")

selected_score_cols: list[str] = []
try:
    idxs = parse_excel_letters_input(letters_text)
    if idxs:
        max_idx = len(df.columns) - 1
        for i in idxs:
            if i < 0 or i > max_idx:
                raise ValueError(f"Column out of range: {index_to_excel_col(i)} (file has {len(df.columns)} columns)")
        selected_score_cols = [df.columns[i] for i in idxs]

        # If user accidentally included id column in scores, remove it (without deleting from df)
        if id_col in selected_score_cols:
            selected_score_cols = [c for c in selected_score_cols if c != id_col]

        st.dataframe(
            pd.DataFrame(
                {"Excel": [index_to_excel_col(i) for i in idxs], "Column title": [df.columns[i] for i in idxs]}
            ),
            width="stretch",
            height=210,
        )
except Exception as e:
    st.error(str(e))

if not selected_score_cols:
    st.info("Type at least 1 Excel column letter to continue.")
    st.stop()

# Weights per selected score (0-10)
st.subheader("Weights (0–10 per selected score)")
st.caption("0 = off • 10 = strongest (normalized automatically)")
weights_key = f"weights_{abs(hash(tuple(selected_score_cols))) % 10**9}"

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

# Grouping settings
st.subheader("Grouping settings")
k = st.slider("Number of groups (1–10)", 1, 10, 3, key="k_groups")
limit_group_size = st.checkbox("Limit max group size", value=False, key="limit_group_size")
cap_pct = st.slider("Max % per group", 1, 40, 20, key="cap_pct") if limit_group_size else 0

try:
    with st.spinner("Recalculating…"):
        valid_work, excluded, weights_view, cap_note = compute_groups(
            df=df,
            id_col=id_col,
            score_cols=selected_score_cols,
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

if len(excluded) > 0:
    with st.expander(f"Students excluded (no usable selected scores): {len(excluded)}"):
        st.dataframe(excluded.reset_index(drop=True), width="stretch", height=240)

st.subheader("Export")
st.download_button(
    "Download CSV (valid students)",
    data=out_table.to_csv(index=False),
    file_name="students_with_groups_overall_score.csv",
    mime="text/csv",
)
