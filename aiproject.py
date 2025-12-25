import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

APP_VERSION = "v10 (IG UI + grade/class always visible)"

# -------------------------
# Page + IG-like styling
# -------------------------
st.set_page_config(page_title="Grouping Studio", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(255, 0, 122, .06), transparent 40%),
                    radial-gradient(circle at 90% 20%, rgba(255, 196, 0, .08), transparent 45%),
                    radial-gradient(circle at 30% 90%, rgba(88, 81, 219, .08), transparent 45%),
                    #fafafa;
      }

      .block-container {padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1200px;}

      .ig-topbar{
        border-radius: 22px;
        padding: 1.1rem 1.2rem;
        background: linear-gradient(135deg,
          rgba(255, 0, 122, 0.95) 0%,
          rgba(255, 196, 0, 0.95) 35%,
          rgba(88, 81, 219, 0.95) 70%,
          rgba(0, 149, 246, 0.95) 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(0,0,0,.16);
        margin-bottom: 0.85rem;
      }
      .ig-title{font-weight: 900; letter-spacing: .2px; font-size: 1.35rem; margin: 0;}
      .ig-sub{opacity: .95; margin: .35rem 0 0; font-size: .92rem; line-height: 1.3;}
      .ig-badge{
        display:inline-block;
        padding:.18rem .55rem;
        border-radius:999px;
        background: rgba(255,255,255,.18);
        border: 1px solid rgba(255,255,255,.25);
        font-size: .78rem;
        font-weight: 700;
        margin-left: .35rem;
      }

      .ig-card{
        background: rgba(255,255,255,.75);
        border: 1px solid rgba(0,0,0,.06);
        border-radius: 20px;
        padding: 1rem 1rem;
        box-shadow: 0 14px 40px rgba(15,23,42,.06);
        backdrop-filter: blur(7px);
        margin-bottom: .85rem;
      }
      .ig-h{
        font-weight: 900;
        font-size: 1.02rem;
        margin: 0 0 .65rem 0;
        color: rgba(17,24,39,.92);
      }
      .ig-muted{color: rgba(17,24,39,.65); font-size: .9rem;}
      .ig-tiny{color: rgba(17,24,39,.55); font-size: .84rem;}

      .chips{display:flex; gap:.55rem; flex-wrap:wrap; margin:.15rem 0 .65rem;}
      .chip{
        display:flex; align-items:center; gap:.45rem;
        padding:.38rem .62rem;
        border-radius: 999px;
        background: white;
        border: 1px solid rgba(0,0,0,.08);
        box-shadow: 0 8px 20px rgba(0,0,0,.05);
        font-weight: 750;
        font-size: .85rem;
        color: rgba(17,24,39,.85);
      }
      .dot{
        width: 10px; height: 10px; border-radius: 50%;
        background: linear-gradient(135deg, #ff007a, #ffc400, #5851db, #0095f6);
      }

      .stDownloadButton button, .stButton button {
        border-radius: 999px !important;
        padding: .6rem 1.0rem !important;
        font-weight: 800 !important;
      }

      .stSelectbox div[data-baseweb="select"] > div,
      .stMultiSelect div[data-baseweb="select"] > div,
      .stFileUploader div {
        border-radius: 14px !important;
      }

      /* Make sidebar optional; user can collapse it without losing controls */
      section[data-testid="stSidebar"] {
        background: rgba(255,255,255,.82);
        border-right: 1px solid rgba(0,0,0,.06);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="ig-topbar">
      <div class="ig-title">Grouping Studio<span class="ig-badge">{APP_VERSION}</span></div>
      <div class="ig-sub">
        Choose scores → set group rules → export a clean roster.
        <b>Group 1 = highest</b>. MAP is converted to percentile (0–100).
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Grade/Class (ALWAYS VISIBLE on main page)
# -------------------------
GRADE_CLASS_COUNT = {
    "JG1": 2,
    "SG1": 4,
    "Grade 2": 4,
    "Grade 3": 5,
    "Grade 4": 5,
    "Grade 5": 6,
}

st.markdown("<div class='ig-card'>", unsafe_allow_html=True)
st.markdown("<div class='ig-h'>🏫 Class Selection</div>", unsafe_allow_html=True)

g1, g2, g3 = st.columns([1.2, 1.2, 2.0])

with g1:
    selected_grade = st.selectbox(
        "Grade",
        list(GRADE_CLASS_COUNT.keys()),
        format_func=lambda g: f"{g} ({GRADE_CLASS_COUNT[g]} classes)",
        key="grade_main",
    )
with g2:
    selected_class = st.selectbox(
        "Class",
        [f"Class {i}" for i in range(1, GRADE_CLASS_COUNT[selected_grade] + 1)],
        key="class_main",
    )
with g3:
    st.markdown(
        f"<div class='ig-muted'>Available classes in <b>{selected_grade}</b>: "
        f"<b>{GRADE_CLASS_COUNT[selected_grade]}</b></div>"
        f"<div class='ig-tiny'>Tip: the left sidebar can be collapsed — these controls stay here.</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Upload
# -------------------------
st.markdown("<div class='ig-card'>", unsafe_allow_html=True)
st.markdown("<div class='ig-h'>📤 Upload</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV (UTF-8 recommended)", type=["csv"])
st.markdown("<div class='ig-tiny'>Tip: Excel → Save As → CSV UTF-8.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
work = df.copy()

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
    st.error("No MAP column found and no 'math unit test N' columns found (1..10). Check your CSV headers.")
    st.stop()

n_students = len(work)

# -------------------------
# Snapshot chips (not debug)
# -------------------------
chips = [
    f"👥 {n_students} students",
    f"🗺️ MAP: {'Yes' if map_col else 'No'}",
    f"🧪 Unit tests: {len(unit_test_cols)}",
    f"🏫 {selected_grade} • {selected_class}",
]
st.markdown("<div class='ig-card'>", unsafe_allow_html=True)
st.markdown("<div class='ig-h'>✨ Quick Snapshot</div>", unsafe_allow_html=True)
st.markdown("<div class='chips'>", unsafe_allow_html=True)
for label in chips:
    st.markdown(f"<div class='chip'><span class='dot'></span>{label}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Score selection (DEFAULT EMPTY)
# Works with ONE test alone
# If exactly one unit test selected => ignore MAP
# -------------------------
options = []
if map_col is not None:
    options.append(map_col)
options += unit_test_cols

st.markdown("<div class='ig-card'>", unsafe_allow_html=True)
st.markdown("<div class='ig-h'>🧩 Choose scores</div>", unsafe_allow_html=True)
selected_score_cols = st.multiselect(
    "Start empty — select 1 or more score columns",
    options=options,
    default=[],
    help="You can select only 1 score column. If you select only 1 unit test, MAP is ignored automatically."
)
st.markdown("<div class='ig-tiny'>Rule: selecting exactly one <b>unit test</b> runs grouping using only that test.</div>", unsafe_allow_html=True)
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
# Controls (k, cap, MAP weight) defaults: cap=0, map weight=0
# -------------------------
st.markdown("<div class='ig-card'>", unsafe_allow_html=True)
st.markdown("<div class='ig-h'>🎚️ Group rules</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    k = st.slider("Groups", 2, 10, 3)
with c2:
    cap_pct = st.slider("Max group size (% of class)", 0, 40, 0, help="0% = no limit")
with c3:
    if use_map and len(selected_unit_tests) > 0:
        map_weight_pct = st.slider("MAP weight (%)", 0, 100, 0)
    else:
        map_weight_pct = 0
        st.caption("MAP weight appears only when MAP + unit tests are selected.")

if k > n_students:
    st.error(f"Groups ({k}) cannot be greater than number of students ({n_students}).")
    st.stop()

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Build model features
# MAP -> percentile 0..100
# Unit tests raw
# Works with ONE feature
# -------------------------
model_features = []
if use_map:
    map_vals = pd.to_numeric(work[map_col], errors="coerce")
    work["map_percentile"] = map_vals.rank(pct=True) * 100.0
    model_features.append("map_percentile")

for c in selected_unit_tests:
    work[c] = pd.to_numeric(work[c], errors="coerce")
model_features += selected_unit_tests

if len(model_features) < 1:
    st.error("No usable numeric features found.")
    st.stop()

# -------------------------
# Weights
# -------------------------
if use_map and len(selected_unit_tests) > 0:
    map_w = map_weight_pct / 100.0
    remaining = 1.0 - map_w
    unit_each = remaining / len(selected_unit_tests)
    weights = np.array([map_w] + [unit_each] * len(selected_unit_tests), dtype=float)
else:
    weights = np.ones(len(model_features), dtype=float)

weights = weights / weights.sum()

# -------------------------
# Preprocess + KMeans
# -------------------------
X_raw = work[model_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
X_imputed = SimpleImputer(strategy="mean").fit_transform(X_raw)
Z = StandardScaler().fit_transform(X_imputed)
Z_w = Z * np.sqrt(weights)

km = KMeans(n_clusters=k, random_state=0, n_init=10)
km.fit(Z_w)
centroids = km.cluster_centers_

dists = np.linalg.norm(Z_w[:, None, :] - centroids[None, :, :], axis=2)

# Assignment with optional cap
if cap_pct == 0:
    assigned = np.argmin(dists, axis=1)
else:
    cap = int(np.ceil((cap_pct / 100.0) * n_students))
    cap = max(cap, 1)
    if k * cap < n_students:
        st.error(
            f"Impossible: {k} groups × max {cap}/group = {k*cap}, but class size is {n_students}. "
            f"Increase cap % or increase groups."
        )
        st.stop()

    sorted_idx = np.argsort(np.sort(dists, axis=1)[:, 1] - np.sort(dists, axis=1)[:, 0])
    remaining = np.array([cap] * k, dtype=int)
    assigned = np.full(n_students, -1, dtype=int)

    for i in sorted_idx:
        for g in np.argsort(dists[i]):
            if remaining[g] > 0:
                assigned[i] = g
                remaining[g] -= 1
                break

    if np.any(assigned == -1):
        st.error("Assignment failed unexpectedly. Try increasing cap %.")
        st.stop()

work["_cluster_internal"] = assigned  # never shown

# Rank clusters: Group 1 best
level_proxy = (Z * weights).sum(axis=1)
work["_level_proxy"] = level_proxy

order_best = (
    work.groupby("_cluster_internal")["_level_proxy"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)
cluster_to_groupnum = {cl: i + 1 for i, cl in enumerate(order_best)}
work["Group"] = work["_cluster_internal"].map(cluster_to_groupnum).astype(int)

group_prefix = f"{selected_grade} {selected_class}"
work["Group Name"] = work["Group"].apply(lambda g: f"{group_prefix} • Group {g}")

# Influence % (MAP vs Unit tests)
row_centroid = centroids[assigned]
diff = (Z_w - row_centroid)
feat_contrib = diff ** 2
total = feat_contrib.sum(axis=1, keepdims=True)
total[total == 0] = 1.0
feat_contrib_pct = feat_contrib / total * 100.0

if use_map:
    map_infl = feat_contrib_pct[:, 0]
    work["MAP_influence_%"] = np.round(map_infl, 2)
    work["UnitTests_influence_%"] = np.round(100.0 - map_infl, 2)
else:
    work["MAP_influence_%"] = 0.0
    work["UnitTests_influence_%"] = 100.0

# -------------------------
# Results
# -------------------------
st.markdown("<div class='ig-card'>", unsafe_allow_html=True)
st.markdown("<div class='ig-h'>🧾 Results</div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns([1, 1, 1, 1])
with m1:
    st.metric("Groups", k)
with m2:
    st.metric("Cap", f"{cap_pct}%")
with m3:
    st.metric("Selected scores", len(selected_score_cols))
with m4:
    st.metric("Label", f"{selected_grade}")

show_cols = [student_id_col, "Group Name"]

if use_map:
    show_cols += [map_col, "map_percentile"]

show_cols += selected_unit_tests
show_cols += ["MAP_influence_%", "UnitTests_influence_%"]

out_table = work[show_cols].sort_values(["Group Name", student_id_col]).reset_index(drop=True)
st.dataframe(out_table, use_container_width=True, height=560)

st.markdown("<div class='ig-tiny'>Sidebar can be hidden — grade/class controls stay at the top.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Group sizes
st.markdown("<div class='ig-card'>", unsafe_allow_html=True)
st.markdown("<div class='ig-h'>📌 Group sizes</div>", unsafe_allow_html=True)
sizes = work.groupby("Group Name").size().reset_index(name="students")
st.dataframe(sizes.sort_values("Group Name"), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Export
st.markdown("<div class='ig-card'>", unsafe_allow_html=True)
st.markdown("<div class='ig-h'>⬇️ Export</div>", unsafe_allow_html=True)
download_df = work[show_cols].copy()
st.download_button(
    "Download CSV",
    data=download_df.to_csv(index=False),
    file_name="students_with_groups.csv",
    mime="text/csv",
    use_container_width=True
)
st.markdown("</div>", unsafe_allow_html=True)
