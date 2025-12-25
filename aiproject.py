import re
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

APP_VERSION = "v8 (polished UI)"

# -------------------------
# Page + styling
# -------------------------
st.set_page_config(page_title="Math Grouping Studio", layout="wide")

st.markdown(
    """
    <style>
      /* Hide Streamlit chrome (optional but professional) */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* App background spacing */
      .block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}

      /* Hero */
      .hero {
        padding: 1.25rem 1.25rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(15,23,42,1) 0%, rgba(30,58,138,1) 55%, rgba(2,132,199,1) 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
        margin-bottom: 1rem;
      }
      .hero h1 {margin: 0; font-size: 1.55rem; font-weight: 750; letter-spacing: 0.2px;}
      .hero p {margin: .35rem 0 0; opacity: .9; font-size: .95rem; line-height: 1.35;}

      /* Section headers */
      .section-title {
        font-weight: 750;
        font-size: 1.1rem;
        margin: 0.2rem 0 0.65rem;
      }

      /* Card */
      .card {
        border: 1px solid rgba(148,163,184,.35);
        background: rgba(255,255,255,.65);
        backdrop-filter: blur(6px);
        border-radius: 16px;
        padding: 0.95rem 0.95rem;
        box-shadow: 0 8px 24px rgba(15,23,42,.06);
      }
      .muted {color: rgba(15,23,42,.72); font-size: .92rem;}
      .tiny {color: rgba(15,23,42,.60); font-size: .85rem;}
      .pill {
        display: inline-block;
        padding: .22rem .55rem;
        border-radius: 999px;
        background: rgba(2,132,199,.12);
        border: 1px solid rgba(2,132,199,.25);
        color: rgba(2,132,199,1);
        font-size: .82rem;
        font-weight: 650;
      }

      /* Divider */
      hr {border: none; border-top: 1px solid rgba(148,163,184,.35); margin: 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Math Grouping Studio</h1>
      <p>Upload a CSV, select score columns, and generate classroom groups labeled <b>Group 1 (highest)</b> → Group N.
      MAP is converted to percentile (0–100). Unit tests stay raw.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar: Grade/Class (dynamic counts)
# -------------------------
GRADE_CLASS_COUNT = {
    "JG1": 2,
    "SG1": 4,
    "Grade 2": 4,
    "Grade 3": 5,
    "Grade 4": 5,
    "Grade 5": 6,
}

with st.sidebar:
    st.markdown("### Class Context")
    selected_grade = st.selectbox("Grade", list(GRADE_CLASS_COUNT.keys()))
    selected_class = st.selectbox(
        "Class",
        [f"Class {i}" for i in range(1, GRADE_CLASS_COUNT[selected_grade] + 1)]
    )
    st.markdown(f"<span class='pill'>{selected_grade} • {selected_class}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### About")
    st.caption(f"App version: {APP_VERSION}")

# -------------------------
# Step 1: Upload
# -------------------------
st.markdown("<div class='section-title'>1) Upload</div>", unsafe_allow_html=True)
upload_box = st.container()
with upload_box:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV (UTF-8 recommended)", type=["csv"])
    st.markdown("<div class='tiny'>Tip: Save as <b>CSV UTF-8</b> from Excel to avoid encoding issues.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
work = df.copy()

# -------------------------
# Detect columns
# -------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

cols_norm = {c: norm(c) for c in work.columns}

# student id
student_id_col = None
for c, n in cols_norm.items():
    if n in ["student_id", "student id", "id"]:
        student_id_col = c
        break

if student_id_col is None:
    work["student_id"] = [f"S{i+1:03d}" for i in range(len(work))]
    student_id_col = "student_id"

# MAP column
map_candidates = [c for c, n in cols_norm.items() if ("map" in n and "math" in n)]
map_col = map_candidates[0] if map_candidates else None

# Unit tests 1..10
unit_pairs = []
for c, n in cols_norm.items():
    m = re.search(r"math\s*unit\s*test\s*(\d+)", n)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 10:
            unit_pairs.append((num, c))
unit_test_cols = [c for _, c in sorted(unit_pairs)]

if map_col is None and len(unit_test_cols) == 0:
    st.error("Could not find MAP or any 'math unit test N' columns (1..10). Check your CSV headers.")
    st.stop()

# -------------------------
# Status strip
# -------------------------
n_students = len(work)
left, mid, right = st.columns([1, 1, 1])
with left:
    st.metric("Students", n_students)
with mid:
    st.metric("MAP column", "Found" if map_col else "Not found")
with right:
    st.metric("Unit tests detected", len(unit_test_cols))

st.markdown("<hr/>", unsafe_allow_html=True)

# -------------------------
# Step 2: Select score columns (DEFAULT: NONE)
# -------------------------
st.markdown("<div class='section-title'>2) Choose score columns</div>", unsafe_allow_html=True)

options = []
if map_col is not None:
    options.append(map_col)
options += unit_test_cols

st.markdown("<div class='card'>", unsafe_allow_html=True)
selected_score_cols = st.multiselect(
    "Select the score columns to use (start empty; choose at least 2)",
    options=options,
    default=[],
    help="Do not select student_id. Choose MAP and/or unit tests."
)
st.markdown("</div>", unsafe_allow_html=True)

if len(selected_score_cols) < 2:
    st.info("Select at least **2** score columns to continue.")
    st.stop()

use_map = (map_col is not None and map_col in selected_score_cols)
selected_unit_tests = [c for c in selected_score_cols if c != map_col]

# -------------------------
# Step 3: Controls (groups + cap + weights)
# -------------------------
st.markdown("<div class='section-title'>3) Grouping controls</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    k = st.slider("Number of groups", min_value=2, max_value=10, value=3)
with c2:
    cap_pct = st.slider(
        "Max size per group (% of class)",
        min_value=0, max_value=40, value=0,
        help="0% = no limit. If cap is too small, grouping becomes impossible."
    )
with c3:
    # MAP weight only makes sense if MAP + at least one unit test selected
    if use_map and len(selected_unit_tests) > 0:
        map_weight_pct = st.slider("MAP weight (%)", 0, 100, 0)
    else:
        map_weight_pct = 0  # not used in that scenario
        st.caption("MAP weight available when MAP + unit tests are selected.")

# Build model features
model_features = []
if use_map:
    map_vals = pd.to_numeric(work[map_col], errors="coerce")
    work["map_percentile"] = map_vals.rank(pct=True) * 100.0
    model_features.append("map_percentile")

for c in selected_unit_tests:
    work[c] = pd.to_numeric(work[c], errors="coerce")
model_features += selected_unit_tests

if len(model_features) < 2:
    st.error("Not enough numeric features after processing. (Select MAP and/or unit tests.)")
    st.stop()

# Weights
if use_map and len(selected_unit_tests) > 0:
    map_w = map_weight_pct / 100.0
    remaining = 1.0 - map_w
    unit_each = remaining / len(selected_unit_tests)
    weights = np.array([map_w] + [unit_each] * len(selected_unit_tests), dtype=float)
else:
    weights = np.ones(len(model_features), dtype=float)

weights = weights / weights.sum()

with st.expander("Show weights used"):
    st.dataframe(
        pd.DataFrame({"feature_used": model_features, "weight_%": (weights * 100).round(2)}),
        use_container_width=True
    )

# -------------------------
# Compute groups
# -------------------------
# Preprocess: impute + standardize + weight
X_raw = work[model_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
X_imputed = SimpleImputer(strategy="mean").fit_transform(X_raw)
Z = StandardScaler().fit_transform(X_imputed)
Z_w = Z * np.sqrt(weights)

# KMeans centroids
km = KMeans(n_clusters=k, random_state=0, n_init=10)
km.fit(Z_w)
centroids = km.cluster_centers_

# Distances
dists = np.linalg.norm(Z_w[:, None, :] - centroids[None, :, :], axis=2)

# Optional capacity assignment
if cap_pct == 0:
    assigned = np.argmin(dists, axis=1)
else:
    cap = int(np.ceil((cap_pct / 100.0) * n_students))
    cap = max(cap, 1)
    if k * cap < n_students:
        st.error(
            f"Impossible settings: {k} groups × max {cap} students/group = {k*cap}, "
            f"but class size is {n_students}. Increase the % cap or increase number of groups."
        )
        st.stop()

    # hard-first ordering
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
        st.error("Assignment failed unexpectedly. Try increasing the cap %.")
        st.stop()

work["_cluster_internal"] = assigned  # internal only

# Rank clusters: Group 1 = best
level_proxy = (Z * weights).sum(axis=1)
work["_level_proxy"] = level_proxy

cluster_order_best_to_worst = (
    work.groupby("_cluster_internal")["_level_proxy"]
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)
cluster_to_groupnum = {cl: i + 1 for i, cl in enumerate(cluster_order_best_to_worst)}
work["Group"] = work["_cluster_internal"].map(cluster_to_groupnum).astype(int)

group_prefix = f"{selected_grade} {selected_class}"
work["Group Name"] = work["Group"].apply(lambda g: f"{group_prefix} Group {g}")

# Influence %
row_centroid = centroids[assigned]
diff = (Z_w - row_centroid)
feat_contrib = diff ** 2
total = feat_contrib.sum(axis=1, keepdims=True)
total[total == 0] = 1.0
feat_contrib_pct = feat_contrib / total * 100.0

contrib_df = pd.DataFrame(feat_contrib_pct, columns=[f"{c} influence %" for c in model_features])

if use_map:
    map_infl = contrib_df.get("map_percentile influence %", pd.Series(np.zeros(n_students))).to_numpy()
    work["MAP_influence_%"] = np.round(map_infl, 2)
    work["UnitTests_influence_%"] = np.round(100.0 - map_infl, 2)
else:
    work["MAP_influence_%"] = 0.0
    work["UnitTests_influence_%"] = 100.0

# -------------------------
# Step 4: Results (clean)
# -------------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>4) Results</div>", unsafe_allow_html=True)

# Clean output columns (no cluster, no z1/z2, no assignment suggestions)
show_cols = [student_id_col, "Group Name"]

if use_map:
    show_cols += [map_col, "map_percentile"]

show_cols += selected_unit_tests
show_cols += ["MAP_influence_%", "UnitTests_influence_%"]

# Compact summary row
s1, s2, s3, s4 = st.columns([1, 1, 1, 1])
with s1:
    st.metric("Groups", k)
with s2:
    st.metric("Cap", f"{cap_pct}%")
with s3:
    st.metric("Selected tests", len(selected_score_cols))
with s4:
    st.metric("Group label", group_prefix)

st.markdown("<div class='card'>", unsafe_allow_html=True)
out_table = work[show_cols].sort_values(["Group Name", student_id_col]).reset_index(drop=True)
st.dataframe(out_table, use_container_width=True, height=520)
st.markdown("</div>", unsafe_allow_html=True)

# Group sizes
with st.expander("Group sizes"):
    sizes = work.groupby("Group Name").size().reset_index(name="students")
    st.dataframe(sizes.sort_values("Group Name"), use_container_width=True)

with st.expander("Advanced: per-feature influence %"):
    st.dataframe(pd.concat([out_table, contrib_df.round(2)], axis=1), use_container_width=True)

# Download
st.markdown("<div class='section-title'>5) Export</div>", unsafe_allow_html=True)
download_df = pd.concat([work[show_cols], contrib_df.round(2)], axis=1)

st.download_button(
    "Download CSV",
    data=download_df.to_csv(index=False),
    file_name="students_with_groups.csv",
    mime="text/csv",
    use_container_width=True
)
