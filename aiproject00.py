import re
import streamlit as st
import pandas as pd
import numpy as np


# =========================
# Excel-letter helpers
# =========================
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
    """
    Accepts: "A B C", "A, B, C", "A:D", "AA:AC"
    Returns 0-based indices in appearance order (deduped).
    """
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


# =========================
# Smart CSV reader (auto header row)
# =========================
def _header_score(row: list[str]) -> float:
    cells = [("" if x is None else str(x).strip()) for x in row]
    nonempty = [c for c in cells if c != "" and c.lower() != "nan"]
    if not nonempty:
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
        sc = _header_score(preview.iloc[i].tolist())
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


# =========================
# Grouping by ranking (Group 1 best)
# =========================
def assign_groups_by_rank(df_sorted: pd.DataFrame, k: int) -> pd.Series:
    n = len(df_sorted)
    k = max(1, min(int(k), n))
    base = n // k
    rem = n % k
    sizes = [(base + 1 if i < rem else base) for i in range(k)]

    groups = []
    for gi, sz in enumerate(sizes, start=1):
        groups.extend([gi] * sz)
    return pd.Series(groups, index=df_sorted.index, dtype=int)


# =========================
# Core compute
# - NO normalization across students
# - ONLY scale each test to 0..100 using per-test max (manual or auto)
# - weights typed -> converted to % so total=100%
# - blanks ignored per-student
# =========================
def compute_results(
    df: pd.DataFrame,
    name_col: str,
    score_cols: list[str],
    typed_weights: np.ndarray,     # any nonnegative numbers
    typed_max_scores: np.ndarray,  # optional; if <=0 or NaN => auto use column max
    k: int,
):
    work = df.copy()

    # numeric conversion (keep columns; blanks -> NaN)
    for c in score_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    X = work[score_cols].to_numpy(dtype=float)  # NxD (NaNs)

    # weights -> fractions sum=1
    u = np.array(typed_weights, dtype=float)
    u = np.where(np.isfinite(u), u, 0.0)
    u[u < 0] = 0.0
    if u.shape[0] != len(score_cols):
        raise ValueError("Weights do not match selected score columns.")
    if float(u.sum()) <= 0:
        raise ValueError("Set at least one weight above 0.")
    w = u / u.sum()  # sum=1

    # max scores (manual or auto)
    m_in = np.array(typed_max_scores, dtype=float)
    m_in = np.where(np.isfinite(m_in), m_in, np.nan)
    if m_in.shape[0] != len(score_cols):
        raise ValueError("Max scores do not match selected score columns.")

    M = np.zeros(len(score_cols), dtype=float)
    for j, c in enumerate(score_cols):
        manual = m_in[j]
        if np.isfinite(manual) and manual > 0:
            M[j] = float(manual)
        else:
            col = X[:, j]
            mm = np.nanmax(col)
            if np.isfinite(mm) and mm > 0:
                M[j] = float(mm)
            else:
                M[j] = np.nan  # unusable column (all blank/0)

    # scale to 0..100 by max score
    # scaled_ij = x_ij / M_j * 100
    S = np.full_like(X, np.nan, dtype=float)
    for j in range(len(score_cols)):
        if not np.isfinite(M[j]) or M[j] <= 0:
            continue
        S[:, j] = (X[:, j] / M[j]) * 100.0

    # clamp to [0,100] (in case someone scores > max or negative)
    S = np.clip(S, 0.0, 100.0)

    # weighted average ignoring blanks:
    present = ~np.isnan(S)              # NxD
    num = np.nansum(S * w, axis=1)      # N
    den = np.sum(present * w, axis=1)   # N (sum of weights for available tests)
    overall = np.where(den > 0, num / den, np.nan)

    work["Overall Score"] = np.round(overall, 2)

    # sort: scored first, best->worst
    work["_scored"] = np.isfinite(work["Overall Score"]).astype(int)
    work_sorted = work.sort_values(
        by=["_scored", "Overall Score", name_col],
        ascending=[False, False, True],
    ).drop(columns=["_scored"])

    # group only scored students
    scored_sorted = work_sorted[np.isfinite(work_sorted["Overall Score"])].copy()
    work_sorted["Group Name"] = np.nan
    if len(scored_sorted) > 0:
        scored_sorted["Group"] = assign_groups_by_rank(scored_sorted, k)
        scored_sorted["Group Name"] = scored_sorted["Group"].apply(lambda g: f"Group {g}")
        work_sorted.loc[scored_sorted.index, "Group Name"] = scored_sorted["Group Name"]

    # weights table (normalized)
    weights_view = pd.DataFrame(
        {
            "score": score_cols,
            "typed_weight": u,
            "weight_%": np.round(w * 100.0, 2),
            "max_score_used": np.round(M, 6),
        }
    )

    # optional debug scaled table (not shown by default)
    scaled_df = pd.DataFrame(S, columns=[f"{c} (scaled/100)" for c in score_cols])

    return work_sorted, weights_view, scaled_df


# =========================
# App UI
# =========================
st.set_page_config(page_title="WAB Classroom Assignment Program", layout="wide")
st.title("WAB Classroom Assignment Program")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.stop()

file_bytes = uploaded.getvalue()
file_sig = (len(file_bytes), hash(file_bytes[:5000]))

# reset UI state when file changes (prevents old selections/results)
if st.session_state.get("_file_sig") != file_sig:
    st.session_state["_file_sig"] = file_sig
    for kk in list(st.session_state.keys()):
        if kk.startswith(("name_letters", "score_letters", "k_groups", "editor_")):
            st.session_state.pop(kk, None)

df = read_csv_smart(file_bytes)

# show column map
colmap = pd.DataFrame(
    {"Excel": [index_to_excel_col(i) for i in range(len(df.columns))], "Column title": list(df.columns)}
)
st.dataframe(colmap, height=260, width="stretch")

# NAME column
st.subheader("Student name column")
name_letters = st.text_input("Type ONE Excel letter (example: E)", value="", key="name_letters")

if name_letters.strip():
    idxs = parse_excel_letters_input(name_letters)
    if len(idxs) != 1:
        st.error("Type exactly ONE Excel letter for the student name column.")
        st.stop()
    idx = idxs[0]
    if idx < 0 or idx >= len(df.columns):
        st.error(f"Out of range: {index_to_excel_col(idx)} (file has {len(df.columns)} columns)")
        st.stop()
    name_col = df.columns[idx]
else:
    # silent fallback
    df = df.copy()
    df["student_id"] = [f"S{i+1:03d}" for i in range(len(df))]
    name_col = "student_id"

# SCORE columns
st.subheader("Score columns")
st.caption("Examples: A, B, C OR A B C")
score_letters = st.text_input("Type Excel letters for score columns", value="", key="score_letters")

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

# don't allow selecting name column as score
score_cols = [df.columns[i] for i in score_idxs if df.columns[i] != name_col]
if not score_cols:
    st.error("No valid score columns selected (don’t include the name column).")
    st.stop()

picked = pd.DataFrame(
    {"Excel": [index_to_excel_col(i) for i in score_idxs], "Column title": [df.columns[i] for i in score_idxs]}
)
st.dataframe(picked, height=170, width="stretch")

# WEIGHTS + MAX SCORES editor (persist)
st.subheader("Weights & Max Score (for 0–100 scaling)")
st.caption("Type ANY weights; the app converts them into % that sum to 100.  Max Score: leave blank to auto-use the column max.")

sig = abs(hash(tuple(score_cols))) % 10**9
editor_key = f"editor_{sig}"

if editor_key in st.session_state:
    base = st.session_state[editor_key]
    if not isinstance(base, pd.DataFrame) or list(base.get("score", [])) != score_cols:
        base = pd.DataFrame({"score": score_cols, "weight": [0.0]*len(score_cols), "max_score": [np.nan]*len(score_cols)})
else:
    base = pd.DataFrame({"score": score_cols, "weight": [0.0]*len(score_cols), "max_score": [np.nan]*len(score_cols)})

edited = st.data_editor(
    base,
    key=editor_key,
    width="stretch",
    num_rows="fixed",
    column_config={
        "score": st.column_config.TextColumn("Score", disabled=True),
        "weight": st.column_config.NumberColumn("Weight", min_value=0.0, step=1.0),
        "max_score": st.column_config.NumberColumn("Max Score (optional)", min_value=0.0, step=1.0),
    },
)

typed_weights = pd.to_numeric(edited["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
typed_max = pd.to_numeric(edited["max_score"], errors="coerce").to_numpy(dtype=float)

# GROUPING
st.subheader("Grouping")
k = st.slider("Number of groups", 1, 10, 3, key="k_groups")

# COMPUTE
try:
    results, weights_view, scaled_df = compute_results(
        df=df,
        name_col=name_col,
        score_cols=score_cols,
        typed_weights=typed_weights,
        typed_max_scores=typed_max,
        k=k,
    )
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Weights used")
st.dataframe(weights_view, width="stretch")

st.subheader("Results")
show_cols = ["Overall Score", name_col, "Group Name"] + score_cols
st.dataframe(results[show_cols].reset_index(drop=True), height=560, width="stretch")

with st.expander("Show scaled-to-100 values (optional)"):
    tmp = pd.concat([results[[name_col, "Overall Score", "Group Name"]].reset_index(drop=True),
                     scaled_df.reset_index(drop=True)], axis=1)
    st.dataframe(tmp, height=420, width="stretch")

st.download_button(
    "Download CSV",
    data=results[show_cols].to_csv(index=False),
    file_name="students_grouped.csv",
    mime="text/csv",
    use_container_width=True,
)
