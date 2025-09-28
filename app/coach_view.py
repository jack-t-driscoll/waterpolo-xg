# -*- coding: utf-8 -*-
"""
Coach View — Classic, no sidebar, rich cuts, schema-flex, ultra-quiet
- Classic layout: Title → KPIs → Tabs (cuts)
- Cuts: Team, Team & Number, Angle, Distance, Shot Type, Attack Type, Pressure, Handedness, Period, plus Shooters and Overview.
- No sidebar, no filter expander.
- Schema-flex for legacy/new exports.
- No pre-reading of CSV (no 1024-char peeks).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------- Paths --------------------
THIS_FILE = Path(__file__).resolve()
APP_DIR = THIS_FILE.parent
DEFAULT_APP_CSV = APP_DIR / "shots.csv"

# -------------------- Config -------------------
GOAL_DEFAULT = (0.0, 0.0)                # default goal center if absent
ANGLE_BIN_EDGES = np.arange(-90, 95, 5)  # 5-degree bins
DIST_BINS = [0, 2, 4, 6, 8, 10, 12, 999]  # meters
DIST_LABELS = ["0-2m", "2-4m", "4-6m", "6-8m", "8-10m", "10-12m", "12m+"]

# Column name candidates per concept (schema-flex)
CANDIDATES = {
    "team":       ["team", "our_team_name", "team_name", "our_team"],
    "shooter":    ["shooter", "player_number", "shooter_number", "drawn_by_player_number"],
    "outcome":    ["outcome", "shot_result", "shot_result_raw"],
    "x":          ["x", "shooter_x"],
    "y":          ["y", "shooter_y"],
    "goal_x":     ["goal_x"],
    "goal_y":     ["goal_y"],
    "man_up":     ["man_up", "man_state"],
    "quarter":    ["quarter", "period"],
    "shot_type":  ["shot_type", "event_type"],
    "attack_type":["attack_type"],
    "pressure":   ["pressure", "defender_count"],
    "handed":     ["shooter_handedness", "handedness", "hand"],
}

# -------------------- Page ---------------------
st.set_page_config(page_title="Coach View", layout="wide")
st.title("Coach View")

# CSS guard: hide any raw <pre> blocks (in case a bare string sneaks in from anywhere)
st.markdown("""
<style>
div[data-testid="stMarkdown"] pre { display: none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- CSV Loader (no previews) ----------------
def read_csv_safely(path: Path) -> pd.DataFrame:
    # Try a couple of encodings; let pandas sniff the delimiter (engine="python")
    errs = []
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            errs.append(f"{enc}: {e}")
    raise RuntimeError("Unable to read CSV with common encodings:\n" + "\n".join(errs))

@st.cache_data(show_spinner=False)
def load_csv_nopeek(path_str: str, mtime_key: float) -> pd.DataFrame:
    return read_csv_safely(Path(path_str))

# -------------------- Helpers ------------------
def first_present(df: pd.DataFrame, names: list[str]):
    for n in names:
        if n in df.columns:
            return n
    return None

def series_from(df: pd.DataFrame, names: list[str], default=np.nan) -> pd.Series:
    col = first_present(df, names)
    return df[col] if col else pd.Series(default, index=df.index)

def normalize_outcome(s: pd.Series) -> pd.Series:
    base = s.astype(str).str.strip().str.lower()
    def map_one(v: str) -> str:
        t = v.replace("_"," ").replace("-"," ").replace("/"," ").replace("|"," ").replace(","," ")
        parts = t.split()
        for p in parts:
            if p in {"goal"}: return "goal"
            if p in {"save","saved"}: return "saved"
            if p in {"post","bar","crossbar"}: return "post"
            if p in {"block","blocked","deflect"}: return "blocked"
            if p in {"miss","wide","out"}: return "miss"
        if "goal" in v: return "goal"
        if "save" in v: return "saved"
        if "post" in v or "bar" in v: return "post"
        if "block" in v: return "blocked"
        if "miss" in v or "wide" in v or "out" in v: return "miss"
        return v
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
    if "angle_deg_signed" not in df.columns or pd.to_numeric(df.get("angle_deg_signed"), errors="coerce").isna().any():
        df["angle_deg_signed"] = np.degrees(np.arctan2(gy_ser - y_ser, gx_ser - x_ser))
    if "distance_m" not in df.columns or pd.to_numeric(df.get("distance_m"), errors="coerce").isna().any():
        df["distance_m"] = np.sqrt((x_ser - gx_ser)**2 + (y_ser - gy_ser)**2)
    df["angle_deg_signed"] = pd.to_numeric(df["angle_deg_signed"], errors="coerce").clip(-90, 90)
    df["x"] = x_ser; df["y"] = y_ser; df["goal_x"] = gx_ser; df["goal_y"] = gy_ser
    return df

def label_angle_bin(val: float) -> str:
    if pd.isna(val): return "Unknown"
    idx = np.digitize([val], ANGLE_BIN_EDGES, right=True)[0] - 1
    idx = max(0, min(idx, len(ANGLE_BIN_EDGES) - 2))
    lo = ANGLE_BIN_EDGES[idx]; hi = ANGLE_BIN_EDGES[idx + 1]
    return f"{int(lo)}° to {int(hi)}°"

def label_dist_bin(val: float) -> str:
    if pd.isna(val): return "Unknown"
    for i in range(len(DIST_BINS)-1):
        if DIST_BINS[i] <= val < DIST_BINS[i+1]:
            return DIST_LABELS[i]
    return DIST_LABELS[-1]

def cut_table(dfin: pd.DataFrame, by: str, order=None, rename=None):
    d = dfin.copy()
    if by not in d.columns:
        return pd.DataFrame(columns=[rename or by, "shots", "goals", "conv_pct"])
    agg = d.groupby(by, dropna=False).agg(
        shots=("outcome", "size"),
        goals=("outcome", lambda s: (s == "goal").astype(int).sum()),
    ).reset_index()
    agg[["shots","goals"]] = agg[["shots","goals"]].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    agg["conv_pct"] = np.where(agg["shots"]>0, (agg["goals"]/agg["shots"]*100).round(1), 0.0)
    if order is not None:
        agg[by] = pd.Categorical(agg[by], categories=order, ordered=True)
        agg = agg.sort_values(by)
    else:
        agg = agg.sort_values(["shots","goals"], ascending=[False, False])
    if rename:
        agg = agg.rename(columns={by: rename})
    return agg

# -------------------- Load Data ----------------
if not DEFAULT_APP_CSV.exists():
    st.error(f"Missing data file: `{DEFAULT_APP_CSV}`. Please place your shots CSV there.")
    st.stop()

mtime = DEFAULT_APP_CSV.stat().st_mtime
try:
    raw = load_csv_nopeek(str(DEFAULT_APP_CSV), mtime)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

if raw.empty:
    st.warning("CSV loaded but has 0 rows.")
    st.stop()

# Canonicalize schema
df = raw.copy()
if "team" not in df.columns:        df["team"]        = series_from(df, CANDIDATES["team"]).astype(str)
if "shooter" not in df.columns:     df["shooter"]     = series_from(df, CANDIDATES["shooter"]).astype(str)
if "outcome" not in df.columns or not df["outcome"].notna().all():
    df["outcome"] = normalize_outcome(series_from(df, CANDIDATES["outcome"]))
if "man_up" not in df.columns:      df["man_up"]      = normalize_man_up(series_from(df, CANDIDATES["man_up"], 0))
else:                                df["man_up"]      = normalize_man_up(df["man_up"])
if "quarter" not in df.columns:     df["quarter"]     = series_from(df, CANDIDATES["quarter"]).astype(str)
if "shot_type" not in df.columns:   df["shot_type"]   = series_from(df, CANDIDATES["shot_type"]).astype(str)
if "attack_type" not in df.columns: df["attack_type"] = series_from(df, CANDIDATES["attack_type"]).astype(str)
if "pressure" not in df.columns:    df["pressure"]    = series_from(df, CANDIDATES["pressure"]).astype(str)
if "handed" not in df.columns:      df["handed"]      = series_from(df, CANDIDATES["handed"]).astype(str)

df = ensure_angle_distance(df)

# -------------------- KPIs ---------------------
c1, c2, c3, c4 = st.columns(4)
total_shots = int(len(df))
goals = int((df["outcome"] == "goal").sum()) if "outcome" in df.columns else 0
conv = (goals / total_shots * 100) if total_shots else 0.0
with c1: st.metric("Shots", f"{total_shots}")
with c2: st.metric("Goals", f"{goals}")
with c3: st.metric("Conversion", f"{conv:.1f}%")
with c4:
    man_rate = (df["man_up"].mean() * 100) if "man_up" in df.columns and total_shots else 0.0
    st.metric("Man-up rate", f"{man_rate:.1f}%")

# -------------------- Tabs ---------------------
tab_overview, tab_shooters, tab_team, tab_teamnum, tab_angle, tab_distance, tab_shot_type, tab_attack_type, tab_pressure, tab_handed, tab_period, tab_downloads = st.tabs(
    ["Overview", "Shooters", "By team", "By team & number", "By angle", "By distance", "By shot type", "By attack type", "By pressure", "By handedness", "By period", "Downloads"]
)

# Overview
with tab_overview:
    st.subheader("Filtered shots")
    preferred = ["team","shooter","outcome","x","y","distance_m","angle_deg_signed","man_up","shot_type","attack_type","pressure","handed","quarter","notes"]
    cols = [c for c in preferred if c in df.columns] or list(df.columns)
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

# Shooters
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

# By team
with tab_team:
    st.subheader("Shooting outcomes by team")
    agg = cut_table(df, "team", rename="Team")
    st.dataframe(agg, use_container_width=True, hide_index=True)

# By team & number
with tab_teamnum:
    st.subheader("Shooting outcomes by team & number")
    d = df.copy()
    if {"team","shooter"}.issubset(d.columns):
        d["team_num"] = d["team"].astype(str) + " • #" + d["shooter"].astype(str)
        agg = cut_table(d, "team_num", rename="Team • #")
        st.dataframe(agg, use_container_width=True, hide_index=True)
    else:
        st.info("Missing team/shooter columns for this cut.")

# By angle
with tab_angle:
    st.subheader("Shooting outcomes by signed angle (−90°…+90°)")
    d = df.copy()
    d["angle_bin"] = d["angle_deg_signed"].apply(label_angle_bin)
    ordered = [f"{int(lo)}° to {int(hi)}°" for lo,hi in zip(ANGLE_BIN_EDGES[:-1], ANGLE_BIN_EDGES[1:])]
    agg = cut_table(d, "angle_bin", order=ordered, rename="Angle bin (°)")
    chart = (
        alt.Chart(agg).mark_bar().encode(
            x=alt.X("Angle bin (°):N", title="Signed angle bin (°)"),
            y=alt.Y("conv_pct:Q", title="Conversion (%)"),
            tooltip=["Angle bin (°)","shots","goals","conv_pct"],
        ).properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    if st.checkbox("Show bin counts table (angle)", value=False, key="counts_angle"):
        st.dataframe(agg, use_container_width=True, hide_index=True)

# By distance
with tab_distance:
    st.subheader("Shooting outcomes by distance")
    d = df.copy()
    d["dist_bin"] = pd.cut(pd.to_numeric(d["distance_m"], errors="coerce"), bins=DIST_BINS, labels=DIST_LABELS, right=False)
    agg = cut_table(d, "dist_bin", order=DIST_LABELS, rename="Distance band")
    chart = (
        alt.Chart(agg).mark_bar().encode(
            x=alt.X("Distance band:N", title="Distance band"),
            y=alt.Y("conv_pct:Q", title="Conversion (%)"),
            tooltip=["Distance band","shots","goals","conv_pct"],
        ).properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    if st.checkbox("Show bin counts table (distance)", value=False, key="counts_distance"):
        st.dataframe(agg, use_container_width=True, hide_index=True)

# By shot type
with tab_shot_type:
    st.subheader("Shooting outcomes by shot type")
    st.dataframe(cut_table(df, "shot_type", rename="Shot type"), use_container_width=True, hide_index=True)

# By attack type
with tab_attack_type:
    st.subheader("Shooting outcomes by attack type")
    st.dataframe(cut_table(df, "attack_type", rename="Attack type"), use_container_width=True, hide_index=True)

# By pressure
with tab_pressure:
    st.subheader("Shooting outcomes by pressure")
    st.dataframe(cut_table(df, "pressure", rename="Pressure"), use_container_width=True, hide_index=True)

# By handedness
with tab_handed:
    st.subheader("Shooting outcomes by handedness")
    st.dataframe(cut_table(df, "handed", rename="Handedness"), use_container_width=True, hide_index=True)

# By period/quarter
with tab_period:
    st.subheader("Shooting outcomes by period/quarter")
    st.dataframe(cut_table(df, "quarter", rename="Period"), use_container_width=True, hide_index=True)

# Downloads
with tab_downloads:
    st.subheader("Export")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="shots_filtered.csv", mime="text/csv", key="dl_btn")
