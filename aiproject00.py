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
# Smart CSV reader (auto header row)
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
# Normalization: per test -> 0..100 (min-max), ignore blanks
# If a test column has no variance, all non-missing become 50.
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


# -------------------------
# Compute Overall Score + Rank-based grouping
# Overall Score = sum( (w/10) * normalized_test_score )
# Missing scores contribute 0 (not renormalized).
# -------------------------
def compute_rank_groups(
    df: pd.DataFrame,
    id_col: str,
    score_cols: list[str],
    weights_0to10: np.ndarray,
    k: int,
    cap_pct: int,
):
    work = df.copy()

    # numeric conversion for selected score columns
    for c in score_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    X = work[score_cols].to_numpy(dtype=float)
    S = minmax_0_100_by_column(X)  # normalized per test, NaN where missing

    w010 = np.array(weights_0to10, dtype=float)
    if w010.shape[0] != len(score_cols):
        raise ValueError("Weights do not match selected score columns.")

    # convert 0..10 -> 0.0..1.0
    w = np.clip(w010, 0.0, 10.0) / 10.0

    if np.sum(w) <= 0:
        raise ValueError("All weights are 0. Set at least one test weight above 0.")

    # IMPORTANT: do NOT renormalize weights (this is what you wanted)
    # If total weight > 1, overall score could exceed 100, so we forbid it to match your meaning.
    total_weight = float(np.sum(w))
    if total_weight > 1.0 + 1e-9:
        raise ValueError(
            f"Total weight is {total_weight:.2f} (> 1.00). "
            f"Reduce weights so total is <= 1.00 (100%)."
        )

    # Missing scores contribute 0 (NOT renormalized)
    S0 = np.nan_to_num(S, nan=0.0)
    overall = S0 @ w  # 0..100 * fraction => 0..100*sum(w) <= 100
    work["Overall Score"] = np.round(overall, 2)

    # Excluded: students with no usable score values for any weighted test
    has_any_value = np.any(~np.isnan(S) & (w[None, :] > 0), axis=1)
    excluded = work.loc[~has_any_value, [id_col] + score_cols].copy()
    valid_work = work.loc[has_any_value].copy()

    if len(valid_work) == 0:
        raise ValueError("No students have any usable selected scores.")

    if k > len(valid_work):
        raise ValueError(f"Number of groups ({k}) cannot exceed valid students ({len(valid_work)}).")

    # Sort by Overall Score (best first)
    valid_work = valid_work.sort_values(["Overall Score", id_col], ascending=[False, True]).reset_index(drop=True)

    # Optional cap check (max % per group)
    if cap_pct and cap_pct > 0:
        max_size = int(np.ceil((cap_pct / 100.0) * len(valid_work)))
        max_size = max(max_size, 1)
        # If cap makes it impossible to place everyone, error clearly
        if max_size * k < len(valid_work):
            raise ValueError(
                f"Impossible: {k} groups × max {max_size}/group = {max_size*k}, "
                f"but you have {len(valid_work)} valid students. Increase max % or groups."
            )

    # Equal split by rank
    n = len(valid_work)
    base = n // k
    rem = n % k
    sizes = [(base + 1 if i < rem else base) for i in range(k)]  # top groups can get +1

    # If cap is enabled, ensure each size <= max_size
    if cap_pct and cap_pct > 0:
        max_size = int(np.ceil((cap_pct / 100.0) * n))
        max_size = max(max_size, 1)
        if any(s > max_size for s in sizes):
            raise ValueError(
                "Your max % per group is too small for an equal rank split. "
                "Increase max % or increase number of groups."
            )

    groups = []
    start = 0
    for gi, sz in enumerate(sizes, start=1):
        end = start + sz
        groups.extend([gi] * sz)
        start = end

    valid_work["Group"] = groups
    valid_work["Group Name"] = valid_work["Group"].apply(lambda g: f"Group {g}")

    # Also provide an overall number (rank) if you want it:
    valid_work["Overall Number"] = np.arange(1, len(valid_work) + 1)

    # Weights view (what % of the final decision each test controls)
    weights_view = pd.DataFrame(
        {
            "score": score_cols,
            "weight_0_to_10": w010.astype(int),
            "weight_fraction": np.round(w, 2),
            "weight_%": np.round(w * 100.0, 1),
        }
    )

    return valid_work, excluded, weights_view


# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="WAB Classroom Assignment Program", layout="wide")
st.title("WAB Classroom Assignment Program")

uploaded = st.file_uploader("Upload CSV (UTF-8 recommended)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
file_sig = (len(file_bytes), hash(file_bytes[:5000]))

# Reset widget state when file changes
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
    st.dataframe(pd.DataFrame({"Excel": [index_to_excel_col(idx)], "Column title": [id_col]}), width="stretch", height=110)
else:
    # fallback silently
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

# Weights per selected score (0-10 each)
st.subheader("Weights (0–10 per selected score)")
st.caption("0 = off • 10 = strongest • Total must be ≤ 10 (100%)")

weights_key = f"weights_{abs(hash(tuple(selected_score_cols))) % 10**9}"

if len(selected_score_cols) == 1:
    default_df = pd.DataFrame({"score": selected_score_cols, "weight_0_to_10": [10]})
else:
    default_df = pd.DataFrame({"score": selected_score_cols, "weight_0_to_10": [0] * len(selected_score_cols)})

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
total010 = int(np.sum(weights_0to10))
st.write(f"Total weight: **{total010}/10**")

# Grouping
st.subheader("Grouping settings")
k = st.slider("Number of groups (1–10)", 1, 10, 3, key="k_groups")
limit_group_size = st.checkbox("Limit max group size", value=False, key="limit_group_size")
cap_pct = st.slider("Max % per group", 1, 40, 20, key="cap_pct") if limit_group_size else 0

try:
    with st.spinner("Recalculating…"):
        valid_work, excluded, weights_view = compute_rank_groups(
            df=df,
            id_col=id_col,
            score_cols=selected_score_cols,
            weights_0to10=weights_0to10,
            k=k,
            cap_pct=cap_pct,
        )
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Weights used")
st.dataframe(weights_view, width="stretch")

st.subheader("Results")
show_cols = ["Overall Score", "Overall Number", id_col, "Group Name"] + selected_score_cols
out_table = valid_work[show_cols].reset_index(drop=True)
st.dataframe(out_table, width="stretch", height=560)

if len(excluded) > 0:
    with st.expander(f"Excluded students (no usable selected scores): {len(excluded)}"):
        st.dataframe(excluded.reset_index(drop=True), width="stretch", height=240)

st.subheader("Export")
st.download_button(
    "Download CSV (valid students)",
    data=out_table.to_csv(index=False),
    file_name="students_with_groups_overall_score.csv",
    mime="text/csv",
)
