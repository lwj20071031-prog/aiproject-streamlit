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


# -------------------------
# Split into groups by rank (best -> Group 1)
# -------------------------
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


# -------------------------
# ✅ 핵심: "가중치 → fraction" 먼저 적용하고, 그 다음 Overall 정규화 (best=100, worst=0)
# - per-test 점수는 0..100으로 정규화
# - weights는 전체 합 100% 되도록 정규화 (2 vs 10이면 16.67% vs 83.33%)
# - 빈칸은 0점 처리 (가중치에서 빼지 않음, 페널티)
# - 마지막에 overall_raw를 학생들끼리 0..100으로 재정규화
# -------------------------
def compute_results(
    df: pd.DataFrame,
    id_col: str,
    score_cols: list[str],
    weights_0to10: np.ndarray,
    k: int,
):
    work = df.copy()

    for c in score_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    X = work[score_cols].to_numpy(dtype=float)

    # 1) 각 시험을 0..100으로 정규화
    S = minmax_0_100_by_column(X)  # NxD, NaN where missing
    S0 = np.nan_to_num(S, nan=0.0)  # ✅ 빈칸은 0점 처리

    # 2) weights 0..10 -> fraction(합 1.0)
    w010 = np.array(weights_0to10, dtype=float)
    if w010.shape[0] != len(score_cols):
        raise ValueError("Weights do not match selected score columns.")
    w010 = np.clip(w010, 0.0, 10.0)

    if float(w010.sum()) <= 0:
        raise ValueError("All weights are 0. Set at least one score weight above 0.")

    w = w010 / w010.sum()  # ✅ 합 1.0 (100%)

    # 3) 가중치 적용한 overall_raw 계산 (여기서 이미 weighting 반영 끝)
    overall_raw = S0 @ w  # 0..100 범위 (weights sum=1, tests 0..100)

    # 4) 마지막에 overall_raw를 학생들 사이에서 0..100으로 재정규화 (best=100, worst=0)
    #    (전체가 같은 값이면 50)
    vmin = float(np.min(overall_raw))
    vmax = float(np.max(overall_raw))
    if vmax == vmin:
        overall_final = np.full_like(overall_raw, 50.0, dtype=float)
    else:
        overall_final = (overall_raw - vmin) / (vmax - vmin) * 100.0

    work["Overall Score"] = np.round(overall_final, 2)

    # sort: best first
    work_sorted = work.sort_values(["Overall Score", id_col], ascending=[False, True]).copy()

    # groups by rank
    work_sorted["Group"] = assign_groups_by_rank(work_sorted, k)
    work_sorted["Group Name"] = work_sorted["Group"].apply(lambda g: f"Group {g}")

    # weights view
    weights_view = pd.DataFrame(
        {
            "score": score_cols,
            "weight_0_to_10": w010.astype(int),
            "weight_%": np.round(w * 100.0, 2),  # ✅ 2 vs 10이면 16.67 / 83.33
        }
    )

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

# reset UI state when file changes
if st.session_state.get("_file_sig") != file_sig:
    st.session_state["_file_sig"] = file_sig
    for kk in list(st.session_state.keys()):
        if kk.startswith(("id_letters_input", "score_letters_input", "k_groups", "weights_")):
            st.session_state.pop(kk, None)

df = read_csv_smart(file_bytes)

# Column map
colmap = pd.DataFrame(
    {"Excel": [index_to_excel_col(i) for i in range(len(df.columns))], "Column title": list(df.columns)}
)
st.dataframe(colmap, height=260, width="stretch")

# Name column
st.subheader("Student name column")
id_letters = st.text_input("Type ONE Excel letter (example: E)", value="", key="id_letters_input")

if id_letters.strip():
    idxs = parse_excel_letters_input(id_letters)
    if len(idxs) != 1:
        st.error("Type exactly ONE Excel letter for the student name column.")
        st.stop()
    idx = idxs[0]
    if idx < 0 or idx >= len(df.columns):
        st.error(f"Out of range: {index_to_excel_col(idx)} (file has {len(df.columns)} columns)")
        st.stop()
    id_col = df.columns[idx]
else:
    # silent fallback
    df = df.copy()
    df["student_id"] = [f"S{i+1:03d}" for i in range(len(df))]
    id_col = "student_id"

# Score columns
st.subheader("Score columns")
score_letters = st.text_input("Type Excel letters (example: A B C)", value="", key="score_letters_input")

try:
    idxs = parse_excel_letters_input(score_letters)
except Exception as e:
    st.error(str(e))
    st.stop()

if not idxs:
    st.stop()

bad = [i for i in idxs if i < 0 or i >= len(df.columns)]
if bad:
    st.error(f"Out of range: {', '.join(index_to_excel_col(i) for i in bad)}")
    st.stop()

score_cols = [df.columns[i] for i in idxs if df.columns[i] != id_col]
if not score_cols:
    st.error("No valid score columns selected (don’t include the name column).")
    st.stop()

# Weights
st.subheader("Weights (0–10 each)")
weights_key = f"weights_{abs(hash(tuple(score_cols))) % 10**9}"
default_df = pd.DataFrame({"score": score_cols, "weight_0_to_10": [0] * len(score_cols)})

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

# Group count
st.subheader("Grouping")
k = st.slider("Number of groups", 1, 10, 3, key="k_groups")

# Compute + display
try:
    results, weights_view = compute_results(
        df=df,
        id_col=id_col,
        score_cols=score_cols,
        weights_0to10=weights_0to10,
        k=k,
    )
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Weights used (normalized to 100%)")
st.dataframe(weights_view, width="stretch")

st.subheader("Results")
show_cols = ["Overall Score", id_col, "Group Name"] + score_cols
st.dataframe(results[show_cols].reset_index(drop=True), height=560, width="stretch")

st.download_button(
    "Download CSV",
    data=results[show_cols].to_csv(index=False),
    file_name="students_grouped.csv",
    mime="text/csv",
    use_container_width=True,
)
