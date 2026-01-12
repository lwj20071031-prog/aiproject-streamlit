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
    out: list[int] = []
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
                    out.append(i)
                    seen.add(i)
        else:
            i = excel_col_to_index(tok)
            if i not in seen:
                out.append(i)
                seen.add(i)

    return out


# -------------------------
# CSV reader (keeps first row as header)
# -------------------------
@st.cache_data(show_spinner=False)
def read_csv_basic(file_bytes: bytes) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            df = pd.read_csv(pd.io.common.BytesIO(file_bytes), encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not read CSV. Last error: {last_err}")


# -------------------------
# Per-test normalization to 0..100 (min-max), ignoring blanks.
# If a column has no variance, all non-missing become 50.
# -------------------------
def minmax_0_100_by_column(X: np.ndarray) -> np.ndarray:
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


def assign_groups_by_rank(sorted_index: pd.Index, k: int) -> pd.Series:
    n = len(sorted_index)
    k = max(1, min(int(k), n))

    base = n // k
    rem = n % k
    sizes = [(base + 1 if i < rem else base) for i in range(k)]

    groups = []
    for gi, sz in enumerate(sizes, start=1):
        groups.extend([gi] * sz)

    return pd.Series(groups, index=sorted_index, dtype=int)


# -------------------------
# Core computation
# - weights typed by user -> normalized so sum=100% (fraction sum=1)
# - per-test normalize to 0..100
# - overall_raw is WEIGHTED AVERAGE ignoring missing tests:
#   overall_raw = sum(w*s) / sum(w) over tests that exist for that student
# - final overall score rescaled across students: best=100, worst=0 (monotonic)
# -------------------------
def compute_results(df: pd.DataFrame, name_col: str, score_cols: list[str], weight_vals: np.ndarray, k: int):
    work = df.copy()

    # convert selected score columns to numeric (blanks -> NaN)
    for c in score_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    X = work[score_cols].to_numpy(dtype=float)

    # 1) per-test normalize to 0..100
    S = minmax_0_100_by_column(X)  # NxD, NaN where missing

    # 2) weights -> fractions that sum to 1
    u = np.array(weight_vals, dtype=float)
    if u.shape[0] != len(score_cols):
        raise ValueError("Weights do not match the selected score columns.")
    u = np.where(np.isfinite(u), u, 0.0)
    u = np.clip(u, 0.0, None)

    if float(u.sum()) <= 0:
        raise ValueError("All weights are 0. Set at least one weight above 0.")

    w = u / u.sum()  # sum=1 (100%)

    weights_view = pd.DataFrame(
        {"score": score_cols, "typed_weight": np.round(u, 6), "weight_%": np.round(w * 100.0, 2)}
    )

    # 3) weighted average per student, ignoring missing tests (NO penalty for blanks)
    present = ~np.isnan(S)                        # NxD
    denom = (present * w).sum(axis=1)             # N
    num = np.nansum(S * w, axis=1)                # N  (NaN ignored)
    overall_raw = np.where(denom > 0, num / denom, np.nan)  # N

    # 4) final rescale across students so best=100, worst=0
    overall_final = np.full_like(overall_raw, np.nan, dtype=float)
    mask = np.isfinite(overall_raw)
    if mask.any():
        vals = overall_raw[mask]
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmax == vmin:
            overall_final[mask] = 50.0
        else:
            overall_final[mask] = (vals - vmin) / (vmax - vmin) * 100.0

    work["Overall Score"] = np.round(overall_final, 2)

    # Sort: scored students first, best -> worst
    work["_scored"] = np.isfinite(work["Overall Score"]).astype(int)
    work_sorted = work.sort_values(["_scored", "Overall Score", name_col], ascending=[False, False, True]).copy()
    work_sorted.drop(columns=["_scored"], inplace=True)

    # Groups only for scored students
    scored_idx = work_sorted[np.isfinite(work_sorted["Overall Score"])].index
    if len(scored_idx) > 0:
        groups = assign_groups_by_rank(scored_idx, k)
        work_sorted["Group"] = np.nan
        work_sorted.loc[scored_idx, "Group"] = groups.astype(int)
        work_sorted["Group Name"] = np.where(
            np.isfinite(work_sorted["Group"]),
            work_sorted["Group"].astype("Int64").map(lambda g: f"Group {g}"),
            np.nan,
        )
    else:
        work_sorted["Group"] = np.nan
        work_sorted["Group Name"] = np.nan

    return work_sorted, weights_view


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="WAB Classroom Assignment Program", layout="wide")
st.title("WAB Classroom Assignment Program")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.stop()

file_bytes = uploaded.getvalue()
file_sig = (len(file_bytes), hash(file_bytes[:5000]))

# Reset state when file changes (prevents old selections showing up)
if st.session_state.get("_file_sig") != file_sig:
    st.session_state["_file_sig"] = file_sig
    for kk in list(st.session_state.keys()):
        if kk.startswith(("name_letters", "score_letters", "weights_editor", "k_groups")):
            st.session_state.pop(kk, None)

df = read_csv_basic(file_bytes)

# Column map (Excel letter -> title) for double-checking
colmap = pd.DataFrame(
    {"Excel": [index_to_excel_col(i) for i in range(len(df.columns))], "Column title": list(df.columns)}
)
st.dataframe(colmap, height=260, width="stretch")

# Name column (Excel letter)
st.subheader("Student name column")
name_letters = st.text_input("Type ONE Excel letter (example: E)", value="", key="name_letters")

if name_letters.strip():
    idxs = parse_excel_letters_input(name_letters)
    if len(idxs) != 1:
        st.error("Type exactly ONE Excel letter for the student name column.")
        st.stop()
    name_idx = idxs[0]
    if name_idx < 0 or name_idx >= len(df.columns):
        st.error(f"Out of range: {index_to_excel_col(name_idx)} (file has {len(df.columns)} columns)")
        st.stop()
    name_col = df.columns[name_idx]
else:
    # silent fallback
    df = df.copy()
    df["student_id"] = [f"S{i+1:03d}" for i in range(len(df))]
    name_col = "student_id"

# Score columns (Excel letters)
st.subheader("Score columns")
score_letters = st.text_input("Type Excel letters (example: A B C)", value="", key="score_letters")

try:
    score_idxs = parse_excel_letters_input(score_letters)
except Exception as e:
    st.error(str(e))
    st.stop()

if not score_idxs:
    st.stop()

bad = [i for i in score_idxs if i < 0 or i >= len(df.columns)]
if bad:
    st.error(f"Out of range: {', '.join(index_to_excel_col(i) for i in bad)}")
    st.stop()

# Build score columns (don’t allow name col)
score_cols = [df.columns[i] for i in score_idxs if df.columns[i] != name_col]
if not score_cols:
    st.error("No valid score columns selected (don’t include the name column).")
    st.stop()

# Show selected columns so users can verify
st.dataframe(
    pd.DataFrame({"Selected": [index_to_excel_col(i) for i in score_idxs], "Column title": [df.columns[i] for i in score_idxs]}),
    height=180,
    width="stretch",
)

# Weights (ANY non-negative numbers; normalized to sum 100%)
st.subheader("Weights (type any numbers)")
default_weights = pd.DataFrame({"score": score_cols, "weight": [0.0] * len(score_cols)})

w_df = st.data_editor(
    default_weights,
    key="weights_editor",
    width="stretch",
    num_rows="fixed",
    column_config={
        "score": st.column_config.TextColumn("Score", disabled=True),
        "weight": st.column_config.NumberColumn("Weight", min_value=0.0, step=0.1),
    },
)

weights = pd.to_numeric(w_df["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

# Group count
st.subheader("Grouping")
k = st.slider("Number of groups", 1, 10, 3, key="k_groups")

# Compute
try:
    results, weights_view = compute_results(
        df=df,
        name_col=name_col,
        score_cols=score_cols,
        weight_vals=weights,
        k=k,
    )
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Weights used (normalized to 100%)")
st.dataframe(weights_view, width="stretch")

st.subheader("Results")
show_cols = ["Overall Score", name_col, "Group Name"] + score_cols
st.dataframe(results[show_cols].reset_index(drop=True), height=560, width="stretch")

st.download_button(
    "Download CSV",
    data=results[show_cols].to_csv(index=False),
    file_name="students_grouped.csv",
    mime="text/csv",
    use_container_width=True,
)
