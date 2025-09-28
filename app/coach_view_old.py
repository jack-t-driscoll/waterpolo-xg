# -*- coding: utf-8 -*-
"""
Coach View — Ultra-Quiet UI (Schema-aware, variant-compatible)
- No debug output anywhere.
- Small schema pill in sidebar only.
- Prefers Schema v1; auto-maps variants silently.
- Signed angle bins (−90..+90), classic layout, unique keys, robust math.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Paths ----------
THIS_FILE = Path(__file__).resolve()
APP_DIR = THIS_FILE.parent
REPO_ROOT = APP_DIR.parent
DEFAULT_APP_CSV = APP_DIR / "shots.csv"
LABELS_DIR = REPO_ROOT / "data" / "labels"

# ---------- Config ----------
GOAL_DEFAULT = (0.0, 0.0)
ANGLE_BIN_EDGES = np.arange(-90, 95, 5)
SCHEMA_V1_COLS = [
    "match_id","video_file","t_start","t_end","team","shooter","outcome",
    "x","y","goal_x","goal_y","angle_deg_signed","distance_m",
    "shot_type","pressure","man_up","goalie_x","goalie_y","quarter","clock","notes","schema_version"
]
CANDIDATES = {
    "team":    ["team","our_team_name","team_name","our_team"],
    "shooter": ["shooter","player_number","shooter_number","drawn_by_player_number"],
    "outcome": ["outcome","shot_result","shot_result_raw"],
    "x":       ["x","shooter_x"],
    "y":       ["y","shooter_y"],
    "goal_x":  ["goal_x"],
    "goal_y":  ["goal_y"],
    "man_up":  ["man_up","man_state"],
}

# ---------- Page ----------
st.set_page_config(page_title="Coach View", layout="wide")
st.title("Coach View")

# ---------- Utilities ----------
def detect_encoding_and_sep(path: Path):
    for enc in ("utf-8","utf-8-sig"):
        try:
            path.read_text(encoding=enc, errors="strict")[:1024]
            return enc, None  # let pandas sniff delimiter with engine="python"
        except Exception:
            continue
    return "utf-8", None

def find_csv_candidates():
    cands = []
    if DEFAULT_APP_CSV.exists(): cands.append(DEFAULT_APP_CSV)
    if LABELS_DIR.exists(): cands += sorted(LABELS_DIR.glob("*.csv"), key=lambda p: p.name.lower())
    seen=set(); uniq=[]
    for p in cands:
        r=p.resolve()
        if r not in seen:
            seen.add(r); uniq.append(p)
    return uniq

def first_present(df: pd.DataFrame, names: list[str]):
    for n in names:
        if n in df.columns:
            return n
    return None

def series_from(df: pd.DataFrame, cand_names: list[str], default=np.nan) -> pd.Series:
    name = first_present(df, cand_names)
    return df[name] if name else pd.Series(default, index=df.index)

def normalize_outcome(s: pd.Series) -> pd.Series:
    base = s.astype(str).str.strip().str.lower()
    def map_one(val: str) -> str:
        t = val.replace("_"," ").replace("-"," ").replace("/"," ").replace("|"," ").replace(","," ")
        parts = t.split()
        for p in parts:
            if p in {"goal"}: return "goal"
            if p in {"save","saved"}: return "saved"
            if p in {"post","bar","crossbar"}: return "post"
            if p in {"block","blocked","deflect"}: return "blocked"
            if p in {"miss","wide","out"}: return "miss"
        if "goal" in val: return "goal"
        if "save" in val: return "saved"
        if "post" in val or "bar" in val: return "post"
        if "block" in val: return "blocked"
        if "miss" in val or "wide" in val or "out" in val: return "miss"
        return val
    return base.map(map_one)

def normalize_man_up(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    def to_int(v: str) -> int:
        if v in {"1","true","yes","y","man-up","manup","powerplay"}: return 1
        if v in {"0","false","no","n",""}: return 0
        try: return 1 if float(v) != 0 else 0
        except: return 0
    return x.map(to_int).astype(int)

def ensure_angle_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    x_ser  = pd.to_numeric(series_from(df, CANDIDATES["x"]), errors="coerce")
    y_ser  = pd.to_numeric(series_from(df, CANDIDATES["y"]), errors="coerce")
    gx_ser = pd.to_numeric(series_from(df, CANDIDATES["goal_x"], GOAL_DEFAULT[0]), errors="coerce").fillna(GOAL_DEFAULT[0])
    gy_ser = pd.to_numeric(series_from(df, CANDIDATES["goal_y"], GOAL_DEFAULT[1]), errors="coerce").fillna(GOAL_DEFAULT[1])

    if "angle_deg_signed" not in df.columns or pd.to_numeric(df["angle_deg_signed"], errors="coerce").isna().any():
        df["angle_deg_signed"] = np.degrees(np.arctan2(gy_ser - y_ser, gx_ser - x_ser))
    if "distance_m" not in df.columns or pd.to_numeric(df["distance_m"], errors="coerce").isna().any():
        df["distance_m"] = np.sqrt((x_ser - gx_ser)**2 + (y_ser - gy_ser)**2)

    df["angle_deg_signed"] = pd.to_numeric(df["angle_deg_signed"], errors="coerce").clip(-90, 90)
    df["x"] = x_ser; df["y"] = y_ser; df["goal_x"] = gx_ser; df["goal_y"] = gy_ser
    return df

def schema_pill(text, color):
    st.sidebar.markdown(
        f"""<div style="display:inline-block;padding:2px 8px;border-radius:999px;
        background:{color};color:white;font-size:12px;">{text}</div>""",
        unsafe_allow_html=True,
    )

# ---------- Sidebar: Data ----------
with st.sidebar:
    st.header("Data source")
    candidates = find_csv_candidates()
    if not candidates:
        st.error(
            "No CSV found. Put a file at:\n"
            f"- {DEFAULT_APP_CSV}\n"
            f"- {LABELS_DIR}/shots.csv\n"
            "or any CSV in app/ or data/labels/."
        )
        st.stop()

    options = [str(p.relative_to(REPO_ROOT)) if REPO_ROOT in p.parents else str(p) for p in candidates]
    chosen = st.selectbox("Select CSV", options, index=0, key="csv_select")
    chosen_path = candidates[options.index(chosen)]
    st.caption(f"Selected: `{chosen_path}`")

# ---------- Load CSV ----------
try:
    mtime = chosen_path.stat().st_mtime
except Exception as e:
    st.error(f"Cannot stat file: {e}")
    st.stop()

@st.cache_data(show_spinner=False)
def load_csv(path_str: str, mtime_key: float) -> pd.DataFrame:
    p = Path(path_str)
    enc, sep = detect_encoding_and_sep(p)
    return pd.read_csv(p, encoding=enc, sep=sep, engine="python")

try:
    raw = load_csv(str(chosen_path), mtime)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

if raw.empty:
    st.warning("CSV loaded but has 0 rows.")
    st.stop()

# ---------- Schema pill (quiet) ----------
is_schema_v1 = all(col in raw.columns for col in SCHEMA_V1_COLS)
if is_schema_v1:
    schema_pill("Schema v1", "#16a34a")  # green
else:
    schema_pill("Legacy schema (auto-mapped)", "#f59e0b")  # amber

# ---------- Canonicalize (v1 & variants) ----------
df = raw.copy()
if "team" not in df.columns:
    df["team"] = series_from(df, CANDIDATES["team"]).astype(str)
if "shooter" not in df.columns:
    df["shooter"] = series_from(df, CANDIDATES["shooter"]).astype(str)
if "outcome" not in df.columns or not df["outcome"].notna().all():
    df["outcome"] = normalize_outcome(series_from(df, CANDIDATES["outcome"]))
if "man_up" not in df.columns:
    df["man_up"] = normalize_man_up(series_from(df, CANDIDATES["man_up"], 0))
else:
    df["man_up"] = normalize_man_up(df["man_up"])
df = ensure_angle_distance(df)

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
total_shots = int(len(df))
goals = int((df["outcome"] == "goal").sum())
conv = (goals / total_shots * 100) if total_shots else 0.0

with c1: st.metric("Shots", f"{total_shots}")
with c2: st.metric("Goals", f"{goals}")
with c3: st.metric("Conversion", f"{conv:.1f}%")
with c4: st.metric("Man-up rate", f"{df['man_up'].mean() * 100:.1f}%")

# ---------- Tabs ----------
tab_overview, tab_shooters, tab_angle, tab_downloads = st.tabs(
    ["Overview", "Shooters", "By angle", "Downloads"]
)

with tab_overview:
    st.subheader("Filtered shots")
    preferred = ["team","shooter","outcome","x","y","distance_m","angle_deg_signed","man_up","match_id","video_file","quarter","clock","notes"]
    cols = [c for c in preferred if c in df.columns] or list(df.columns)
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

with tab_shooters:
    st.subheader("Shooter performance")
    d = df.copy()
    d["shooter_str"] = d["team"].astype(str) + " • #" + d["shooter"].astype(str)
    grp = d.groupby("shooter_str", dropna=False).agg(
        shots=("outcome", "size"),
        goals=("outcome", lambda s: (s == "goal").astype(int).sum()),
    ).reset_index()
    grp[["shots","goals"]] = grp[["shots","goals"]].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    grp["conv_pct"] = np.where(grp["shots"]>0, (grp["goals"]/grp["shots"]*100).round(1), 0.0)
    grp = grp.sort_values(["shots","goals"], ascending=[False,False])
    st.dataframe(grp.rename(columns={"shooter_str":"Shooter (Team • #)"}), use_container_width=True, hide_index=True)

with tab_angle:
    st.subheader("Shooting outcomes by signed angle (−90°…+90°)")
    d = df.copy()
    def label_angle_bin(val: float) -> str:
        if pd.isna(val): return "Unknown"
        idx = np.digitize([val], ANGLE_BIN_EDGES, right=True)[0] - 1
        idx = max(0, min(idx, len(ANGLE_BIN_EDGES) - 2))
        lo = ANGLE_BIN_EDGES[idx]; hi = ANGLE_BIN_EDGES[idx + 1]
        return f"{int(lo)}° to {int(hi)}°"
    d["angle_bin"] = d["angle_deg_signed"].apply(label_angle_bin)
    agg = d.groupby("angle_bin", dropna=False).agg(
        shots=("outcome","size"),
        goals=("outcome", lambda s: (s == "goal").astype(int).sum()),
    ).reset_index()
    agg[["shots","goals"]] = agg[["shots","goals"]].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    agg["conv_pct"] = np.where(agg["shots"]>0, (agg["goals"]/agg["shots"]*100).round(1), 0.0)

    ordered = [f"{int(lo)}° to {int(hi)}°" for lo,hi in zip(ANGLE_BIN_EDGES[:-1], ANGLE_BIN_EDGES[1:])]
    agg["angle_bin"] = pd.Categorical(agg["angle_bin"], categories=ordered, ordered=True)
    agg = agg.sort_values("angle_bin")

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("angle_bin:N", title="Signed angle bin (°)"),
            y=alt.Y("conv_pct:Q", title="Conversion (%)"),
            tooltip=["angle_bin","shots","goals","conv_pct"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

    show_counts = st.checkbox("Show bin counts table", value=False, key="angle_counts_checkbox")
    if show_counts:
        st.dataframe(agg.rename(columns={"angle_bin":"Angle bin (°)"}), use_container_width=True, hide_index=True)

with tab_downloads:
    st.subheader("Export")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="shots_filtered.csv", mime="text/csv", key="dl_btn")
