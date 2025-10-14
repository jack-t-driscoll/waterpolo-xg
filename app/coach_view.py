# -*- coding: utf-8 -*-
"""
Coach View — Two-track layout (Shots | Non-shot events), schema-flex, shots-first
- Parent tabs: "Shots", "Non-shot events"
- Shots: Overview, Shooters, By team, By angle, By distance, By shot type, By attack type, By pressure,
         By handedness, By period, Downloads, Heatmap (if available)
- Non-shot: Possession summary, Turnovers (summary), Turnovers by team & player, Turnovers by context,
           Ejections drawn, 5m drawn
- Geometry: prefers meters (x_m, y_m); else converts normalized (shooter_x, shooter_y)→meters (20x15)
- Angle: atan2(x_m, y_m) * 180/pi (signed; left<0, right>0)
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
NORMALIZED_APP_CSV = APP_DIR / "shots_norm.csv"  # optional (hybrid)

# -------------------- Config -------------------
POOL_WIDTH_M = 20.0
FRONTCOURT_LENGTH_M = 15.0
ANGLE_BIN_EDGES = np.arange(-90, 95, 5)  # 5-degree bins
DIST_BINS = [0, 2, 4, 6, 8, 10, 12, 999]  # meters
DIST_LABELS = ["0-2m", "2-4m", "4-6m", "6-8m", "8-10m", "10-12m", "12m+"]

# Column name candidates per concept (schema-flex)
CANDIDATES = {
    "team":       ["team", "our_team_name", "team_name", "our_team"],
    "player":     ["player_number", "shooter", "shooter_number"],
    "outcome":    ["outcome", "shot_result", "shot_result_raw"],
    "x_norm":     ["x", "shooter_x"],   # 0..1
    "y_norm":     ["y", "shooter_y"],   # 0..1
    "x_m":        ["x_m"],              # -10..+10
    "y_m":        ["y_m"],              # 0..15
    "man_state":  ["man_state"],
    "period":     ["period", "quarter"],
    "shot_type":  ["shot_type"],
    "attack_type":["attack_type"],
    "pressure":   ["pressure", "defender_count"],
    "handed":     ["handed", "shooter_handedness"],
    "event_type": ["event_type"],
    "turnover_type": ["turnover_type"],
    "turnover_player_number": ["turnover_player_number"],
    "drawn_by_player_number": ["drawn_by_player_number"],
    "possession_id": ["possession_id"],
}

# -------------------- Page ---------------------
st.set_page_config(page_title="Coach View", layout="wide")
st.title("Coach View")

# CSS guard
st.markdown("""
<style>
div[data-testid="stMarkdown"] pre { display: none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- CSV Loader ----------------
def read_csv_safely(path: Path) -> pd.DataFrame:
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

def ensure_geometry_meters(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    x_m_col = first_present(d, CANDIDATES["x_m"])
    y_m_col = first_present(d, CANDIDATES["y_m"])
    if x_m_col and y_m_col:
        x_m = pd.to_numeric(d[x_m_col], errors="coerce")
        y_m = pd.to_numeric(d[y_m_col], errors="coerce")
    else:
        x_norm = pd.to_numeric(series_from(d, CANDIDATES["x_norm"]), errors="coerce")
        y_norm = pd.to_numeric(series_from(d, CANDIDATES["y_norm"]), errors="coerce")
        x_m = (x_norm - 0.5) * POOL_WIDTH_M
        y_m = y_norm * FRONTCOURT_LENGTH_M
    d["x_m"] = x_m
    d["y_m"] = y_m
    d["distance_m"] = np.hypot(d["x_m"], d["y_m"])
    d["angle_deg_signed"] = np.degrees(np.arctan2(d["x_m"], d["y_m"]))  # atan2(x, y)
    return d

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

def build_heatmap_df(dfin: pd.DataFrame) -> pd.DataFrame:
    d = dfin.copy()
    d["angle_deg_signed"] = pd.to_numeric(d["angle_deg_signed"], errors="coerce")
    d["distance_m"] = pd.to_numeric(d["distance_m"], errors="coerce")
    angle_labels = [f"{int(lo)}° to {int(hi)}°" for lo, hi in zip(ANGLE_BIN_EDGES[:-1], ANGLE_BIN_EDGES[1:])]
    d["angle_bin"] = pd.cut(d["angle_deg_signed"], ANGLE_BIN_EDGES, right=False, include_lowest=True, labels=angle_labels)
    d["dist_bin"] = pd.cut(d["distance_m"], DIST_BINS, right=False, labels=DIST_LABELS)
    grp = d.groupby(["dist_bin","angle_bin"], dropna=False).agg(
        shots=("outcome","size"),
        goals=("outcome", lambda s: (s == "goal").astype(int).sum())
    ).reset_index()
    grp["conv_pct"] = np.where(grp["shots"] > 0, (grp["goals"]/grp["shots"]*100).round(1), 0.0)
    grp["dist_bin"] = pd.Categorical(grp["dist_bin"], categories=DIST_LABELS, ordered=True)
    grp["angle_bin"] = pd.Categorical(grp["angle_bin"], categories=angle_labels, ordered=True)
    return grp

def _to_int_series(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    return s2.dropna().astype(int).reindex(s.index)

def _ensure_team(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a single, 1-D 'team' column for grouping."""
    d = df.copy()
    if "team" in d.columns and not isinstance(d["team"], pd.DataFrame):
        return d
    tcol = first_present(d, CANDIDATES["team"])
    d["team"] = d[tcol].astype(str) if tcol else "Unknown"
    return d

def _unique_possessions(df: pd.DataFrame) -> int:
    pid_col = first_present(df, CANDIDATES["possession_id"])
    if not pid_col or pid_col not in df.columns:
        return 0
    return int(df[pid_col].dropna().astype(str).nunique())

def _unique_possessions_by_team(df: pd.DataFrame, team_col: str = "team") -> pd.Series:
    """Return a Series indexed by team with unique possession counts; safe if team missing."""
    d = _ensure_team(df)
    pid_col = first_present(d, CANDIDATES["possession_id"])
    if not pid_col or pid_col not in d.columns:
        return d.groupby("team").size().mul(0)
    tmp = d[["team", pid_col]].dropna().copy()
    tmp[pid_col] = tmp[pid_col].astype(str)
    return tmp.drop_duplicates(["team", pid_col]).groupby("team")[pid_col].nunique()

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

# Base df (full) + universal team column
df_full = _ensure_team(raw.copy())

# ---- Event slices ----
evt_col = first_present(df_full, CANDIDATES["event_type"]) or "event_type"
evt = df_full[evt_col].astype(str).str.lower() if evt_col in df_full.columns else pd.Series("", index=df_full.index)

df_shots = df_full[evt.eq("shot")].copy()
df_nonshot = df_full[~evt.eq("shot")].copy()

# ---- Shots canonicals ----
out_col = first_present(df_shots, CANDIDATES["outcome"])
df_shots["outcome"] = normalize_outcome(df_shots[out_col]) if out_col else pd.Series(np.nan, index=df_shots.index)
df_shots = ensure_geometry_meters(df_shots)

player_col_shots = first_present(df_shots, CANDIDATES["player"])
period_col_shots = first_present(df_shots, CANDIDATES["period"])
shot_type_col    = first_present(df_shots, CANDIDATES["shot_type"])
attack_col       = first_present(df_shots, CANDIDATES["attack_type"])
pressure_col     = first_present(df_shots, CANDIDATES["pressure"])

if "handed" not in df_shots.columns:
    df_shots["handed"] = series_from(df_shots, CANDIDATES["handed"]).astype(str).str.strip()

if player_col_shots:
    df_shots[player_col_shots] = _to_int_series(df_shots[player_col_shots]).astype("Int64")
if pressure_col == "defender_count":
    df_shots[pressure_col] = _to_int_series(df_shots[pressure_col]).astype("Int64")
if period_col_shots:
    df_shots[period_col_shots] = _to_int_series(df_shots[period_col_shots]).astype("Int64")

# ---- Normalized dataset for Heatmap (shots-only) ----
df_norm = None
if NORMALIZED_APP_CSV.exists():
    try:
        mtime_norm = NORMALIZED_APP_CSV.stat().st_mtime
        temp = load_csv_nopeek(str(NORMALIZED_APP_CSV), mtime_norm)
        if not temp.empty:
            tevt_col = first_present(temp, CANDIDATES["event_type"]) or "event_type"
            tevt = temp[tevt_col].astype(str).str.lower() if tevt_col in temp.columns else pd.Series("", index=temp.index)
            df_norm = temp[tevt.eq("shot")].copy()
            tout_col = first_present(df_norm, CANDIDATES["outcome"])
            df_norm["outcome"] = normalize_outcome(df_norm[tout_col]) if tout_col else pd.Series(np.nan, index=df_norm.index)
            df_norm = ensure_geometry_meters(df_norm)
            df_norm = _ensure_team(df_norm)
    except Exception:
        df_norm = None

# -------------------- KPIs (shots-only) ---------------------
c1, c2, c3, c4 = st.columns(4)
total_shots = int(len(df_shots))
goals = int((df_shots["outcome"] == "goal").sum()) if "outcome" in df_shots.columns else 0
conv = (goals / total_shots * 100) if total_shots else 0.0
with c1: st.metric("Shots", f"{total_shots}")
with c2: st.metric("Goals", f"{goals}")
with c3: st.metric("Conversion", f"{conv:.1f}%")
with c4:
    if "man_state" in df_shots.columns:
        man_rate = (df_shots["man_state"].astype(str).str.upper().isin({"6V5","6V4"}).mean() * 100) if total_shots else 0.0
        st.metric("Man-up rate", f"{man_rate:.1f}%")
    else:
        st.metric("Man-up rate", "—")

# -------------------- Parent tabs ---------------------
parent_shots, parent_nonshot = st.tabs(["Shots", "Non-shot events"])

# ===== Shots parent =====
with parent_shots:
    # Subtabs
    base_tabs = ["Overview", "Shooters", "By team", "By angle", "By distance",
                 "By shot type", "By attack type", "By pressure", "By handedness", "By period", "Downloads"]
    tabs = base_tabs + (["Heatmap"] if df_norm is not None else [])
    (tab_overview, tab_shooters, tab_team, tab_angle, tab_distance,
     tab_shot_type, tab_attack_type, tab_pressure, tab_handed, tab_period, tab_downloads, *maybe_heatmap) = st.tabs(tabs)

    # Overview (full, unfiltered)
    with tab_overview:
        st.subheader("All events (raw)")
        st.dataframe(df_full, use_container_width=True, hide_index=True)

    # Shooters
    with tab_shooters:
        st.subheader("Shooter performance (shots only)")
        if player_col_shots:
            d = df_shots.copy()
            d["shooter_str"] = d["team"].astype(str) + " • #" + d[player_col_shots].astype(str)
            grp = d.groupby("shooter_str", dropna=False).agg(
                shots=("outcome", "size"),
                goals=("outcome", lambda s: (s == "goal").astype(int).sum()),
            ).reset_index()
            grp[["shots","goals"]] = grp[["shots","goals"]].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
            grp["conv_pct"] = np.where(grp["shots"]>0, (grp["goals"]/grp["shots"]*100).round(1), 0.0)
            grp = grp.sort_values(["shots","goals"], ascending=[False,False])
            st.dataframe(grp.rename(columns={"shooter_str":"Shooter (Team • #)"}), use_container_width=True, hide_index=True)
        else:
            st.info("Missing player number column for this cut.")

    # By team
    with tab_team:
        st.subheader("Shooting outcomes by team (shots only)")
        st.dataframe(cut_table(df_shots, "team", rename="Team"), use_container_width=True, hide_index=True)

    # By angle
    with tab_angle:
        st.subheader("Shooting outcomes by angle (° bins) — shots only")
        d = df_shots.copy()
        d["angle_deg_signed"] = pd.to_numeric(d["angle_deg_signed"], errors="coerce")
        angle_labels = [f"{int(lo)}° to {int(hi)}°" for lo, hi in zip(ANGLE_BIN_EDGES[:-1], ANGLE_BIN_EDGES[1:])]
        d["angle"] = pd.cut(d["angle_deg_signed"], ANGLE_BIN_EDGES, right=False, include_lowest=True, labels=angle_labels)
        st.dataframe(cut_table(d, "angle", order=angle_labels, rename="Angle bin (°)"),
                     use_container_width=True, hide_index=True)

    # By distance
    with tab_distance:
        st.subheader("Shooting outcomes by distance — shots only")
        d = df_shots.copy()
        d["distance_m"] = pd.to_numeric(d["distance_m"], errors="coerce")
        d["dist_bin"] = pd.cut(d["distance_m"], DIST_BINS, right=False, labels=DIST_LABELS)
        st.dataframe(cut_table(d, "dist_bin", order=DIST_LABELS, rename="Distance band"),
                     use_container_width=True, hide_index=True)

    # By shot type
    with tab_shot_type:
        st.subheader("Shooting outcomes by shot type — shots only")
        if shot_type_col:
            st.dataframe(cut_table(df_shots.rename(columns={shot_type_col:"shot_type"}), "shot_type", rename="Shot type"),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Missing shot_type column.")

    # By attack type
    with tab_attack_type:
        st.subheader("Shooting outcomes by attack type — shots only")
        if attack_col:
            st.dataframe(cut_table(df_shots.rename(columns={attack_col:"attack_type"}), "attack_type", rename="Attack type"),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Missing attack_type column.")

    # By pressure
    with tab_pressure:
        st.subheader("Shooting outcomes by pressure — shots only")
        d = df_shots.copy()
        if pressure_col == "defender_count":
            d[pressure_col] = pd.to_numeric(d[pressure_col], errors="coerce").astype("Int64")
        st.dataframe(cut_table(d, pressure_col if pressure_col else "pressure", rename="Pressure"),
                     use_container_width=True, hide_index=True)

    # By handedness
    with tab_handed:
        st.subheader("Shooting outcomes by handedness — shots only")
        d = df_shots.copy()
        if "handed" not in d.columns:
            d["handed"] = series_from(d, CANDIDATES["handed"]).astype(str).str.strip()
        st.dataframe(cut_table(d, "handed", rename="Handedness"),
                     use_container_width=True, hide_index=True)

    # By period
    with tab_period:
        st.subheader("Shooting outcomes by period — shots only")
        d = df_shots.copy()
        if period_col_shots:
            d[period_col_shots] = pd.to_numeric(d[period_col_shots], errors="coerce").astype("Int64")
            st.dataframe(cut_table(d.rename(columns={period_col_shots:"period"}), "period", rename="Period"),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Missing period column.")

    # Downloads (with VLC Jump List)
    with tab_downloads:
        st.subheader("Export (raw all-events CSV)")
        csv_bytes = df_full.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="shots_all_events.csv", mime="text/csv", key="dl_all_csv")

        st.markdown("---")
        st.subheader("Jump list (VLC) — shots only")

        # Optional videos mapping
        vids_path = APP_DIR.parent / "data" / "videos.csv"
        vids = None
        if vids_path.exists():
            try:
                vids = pd.read_csv(vids_path, dtype=str)
            except Exception:
                vids = None

        def _mmss_to_seconds(mmss: str) -> float:
            try:
                s = str(mmss).strip()
                if not s or s.lower() == "nan":
                    return np.nan
                parts = s.split(":")
                if len(parts) == 2:
                    m, s = int(parts[0]), float(parts[1])
                    return m * 60 + s
                elif len(parts) == 3:
                    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                    return h * 3600 + m * 60 + s
                else:
                    return float(s)  # if it's just seconds
            except Exception:
                return np.nan

        # Build jump list from SHOTS only
        j = df_shots.copy()

        # Resolve video path: try join to videos.csv on (video_file == source_video_id), else use video_file as-is
        video_file_col = "video_file" if "video_file" in j.columns else None
        if vids is not None and video_file_col and {"source_video_id","source_video_path"}.issubset(vids.columns):
            j = j.merge(
                vids[["source_video_id", "source_video_path"]],
                left_on=video_file_col, right_on="source_video_id", how="left"
            )
            j["video_path"] = j["source_video_path"].fillna(j[video_file_col])
        else:
            if video_file_col:
                j["video_path"] = j[video_file_col].astype(str)
            else:
                j["video_path"] = ""

        # Parse start time
        ts_col = "video_timestamp_mmss" if "video_timestamp_mmss" in j.columns else None
        j["start_time_s"] = j[ts_col].map(_mmss_to_seconds) if ts_col else np.nan

        # Friendly label
        label_parts = []
        if "team" in j.columns: label_parts.append(j["team"].astype(str))
        if player_col_shots:    label_parts.append(("#" + j[player_col_shots].astype(str)))
        if period_col_shots:    label_parts.append(("Q" + j[period_col_shots].astype(str)))
        if label_parts:
            j["label"] = label_parts[0]
            for p in label_parts[1:]:
                j["label"] = j["label"] + " • " + p
        else:
            j["label"] = ""

        # VLC command per row
        j["vlc_cmd"] = j.apply(
            lambda r: f'vlc --qt-start-minimized --play-and-exit --start-time="{r["start_time_s"]}" "{r["video_path"]}"',
            axis=1
        )

        jump_cols = ["video_path", "start_time_s", "label", "vlc_cmd"]
        jump_df = j[jump_cols]
        jl_bytes = jump_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download VLC jump list (CSV)", data=jl_bytes, file_name="jump_list_vlc.csv", mime="text/csv", key="dl_jump_csv")

    # Heatmap
    if df_norm is not None and len(maybe_heatmap) == 1:
        (tab_heatmaps,) = maybe_heatmap
        with tab_heatmaps:
            st.subheader("Heatmap (distance × angle) — shots only")
            try:
                shots = df_norm.copy()
                heat = build_heatmap_df(shots)
                angle_labels = [f"{int(lo)}° to {int(hi)}°" for lo, hi in zip(ANGLE_BIN_EDGES[:-1], ANGLE_BIN_EDGES[1:])]
                heat["angle_bin"] = pd.Categorical(heat["angle_bin"], categories=angle_labels, ordered=True)
                heat["dist_bin"] = pd.Categorical(heat["dist_bin"], categories=DIST_LABELS, ordered=True)
                base = alt.Chart(heat).encode(
                    x=alt.X("angle_bin:O", title="Angle (° bins)", sort=angle_labels),
                    y=alt.Y("dist_bin:O", title="Distance bands (m)", sort=DIST_LABELS),
                ).properties(height=360)
                conv_map = base.mark_rect().encode(
                    color=alt.Color("conv_pct:Q", title="Conversion (%)", scale=alt.Scale(scheme="greens")),
                    tooltip=["dist_bin","angle_bin","shots","goals","conv_pct"]
                )
                counts = base.mark_text(baseline="middle").encode(
                    text=alt.Text("shots:Q", format="d"),
                    color=alt.value("#111")
                )
                st.altair_chart(conv_map + counts, use_container_width=True)
            except Exception as e:
                st.warning(f"Heatmap could not be rendered: {e}")

# ===== Non-shot parent =====
with parent_nonshot:
    df_ns = df_nonshot.copy()

    # Prepare specialized slices
    pid_col = first_present(df_ns, CANDIDATES["possession_id"])
    period_col_ns = first_present(df_ns, CANDIDATES["period"])
    to_type_col = first_present(df_ns, CANDIDATES["turnover_type"])
    to_player_col = first_present(df_ns, CANDIDATES["turnover_player_number"])
    drawn_player_col = first_present(df_ns, CANDIDATES["drawn_by_player_number"])

    df_turn = df_ns[evt.eq("turnover")].copy()
    df_draw = df_ns[evt.eq("ejection_drawn")].copy()
    df_5m   = df_ns[evt.eq("5m_drawn")].copy()

    (tab_possum, tab_tosum, tab_toplayer, tab_tocontext, tab_draw, tab_5m) = st.tabs(
        ["Possession summary", "Turnovers (summary)", "Turnovers by team & player", "Turnovers by context", "Ejections drawn", "5m drawn"]
    )

    # Possession summary
    with tab_possum:
        st.subheader("Possession summary")
        total_pos = _unique_possessions(df_full) if pid_col else 0
        shot_pos  = _unique_possessions(df_shots) if pid_col else 0
        turn_pos  = _unique_possessions(df_turn) if pid_col else 0
        draw_pos  = _unique_possessions(df_draw) if pid_col else 0
        five_pos  = _unique_possessions(df_5m) if pid_col else 0

        summary_rows = []
        if pid_col:
            summary_rows += [
                {"metric":"Total possessions", "value": total_pos},
                {"metric":"Shot rate (per 100 poss.)", "value": round(100 * (shot_pos/total_pos), 1) if total_pos else "—"},
                {"metric":"Turnover rate (per 100 poss.)", "value": round(100 * (turn_pos/total_pos), 1) if total_pos else "—"},
                {"metric":"Ejection-drawn rate (per 100 poss.)", "value": round(100 * (draw_pos/total_pos), 1) if total_pos else "—"},
                {"metric":"5m-drawn rate (per 100 poss.)", "value": round(100 * (five_pos/total_pos), 1) if total_pos else "—"},
            ]
        else:
            summary_rows.append({"metric":"Total possessions", "value":"(possession_id missing)"})
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if pid_col:
            st.caption("Rates by team (per 100 team possessions)")
            team_pos = _unique_possessions_by_team(df_full)  # safe; ensures 'team'
            rows = []
            for team, denom in team_pos.items():
                if denom == 0: continue
                sp = _unique_possessions(df_shots[df_shots["team"]==team])
                tp = _unique_possessions(df_turn[df_turn["team"]==team])
                dp = _unique_possessions(df_draw[df_draw["team"]==team])
                fp = _unique_possessions(df_5m[df_5m["team"]==team])
                rows.append({
                    "Team": team,
                    "Shot rate /100": round(100*sp/denom,1),
                    "Turnover rate /100": round(100*tp/denom,1),
                    "Ejection-drawn rate /100": round(100*dp/denom,1),
                    "5m-drawn rate /100": round(100*fp/denom,1),
                    "Team possessions": int(denom),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("Team"), use_container_width=True, hide_index=True)

    # Turnovers (summary)
    with tab_tosum:
        st.subheader("Turnovers (summary)")
        d = df_turn.copy()
        if to_type_col:
            grp = d.groupby([to_type_col], dropna=False).size().rename("count").reset_index()
            grp = grp.rename(columns={to_type_col: "Turnover type"})
        else:
            grp = pd.DataFrame({"Turnover type":["(unknown)"], "count":[len(d)]})
        total_pos = _unique_possessions(df_full) if pid_col else 0
        grp["rate_per_100_poss"] = grp["count"].apply(lambda c: round(100*c/total_pos,1) if total_pos else np.nan)
        st.dataframe(grp.sort_values("count", ascending=False), use_container_width=True, hide_index=True)

        if period_col_ns:
            d[period_col_ns] = _to_int_series(d[period_col_ns]).astype("Int64")
            grp2 = d.groupby(period_col_ns, dropna=False).size().rename("count").reset_index()
            grp2 = grp2.rename(columns={period_col_ns:"Period"})
            st.caption("By period")
            st.dataframe(grp2.sort_values("Period"), use_container_width=True, hide_index=True)

    # Turnovers by team & player
    with tab_toplayer:
        st.subheader("Turnovers by team & player")
        d = df_turn.copy()
        if to_player_col:
            nums = _to_int_series(d[to_player_col]).astype("Int64").astype(str)
            d["Team • #"] = d["team"].astype(str) + " • #" + nums
            grp = d.groupby(["Team • #", "team"], dropna=False).size().rename("count").reset_index()
            team_tot = grp.groupby("team")["count"].transform("sum")
            grp["team_share_pct"] = (grp["count"] / team_tot * 100).round(1).fillna(0)
            team_pos = _unique_possessions_by_team(df_full)
            grp["rate_per_100_team_poss"] = grp.apply(
                lambda r: round(100 * r["count"] / team_pos.get(r["team"], 0), 1) if team_pos.get(r["team"], 0) else np.nan,
                axis=1
            )
            st.dataframe(grp.drop(columns=["team"]).sort_values(["count","team_share_pct"], ascending=[False, False]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No turnover_player_number column found.")

    # Turnovers by context
    with tab_tocontext:
        st.subheader("Turnovers by context")
        d = df_turn.copy()
        atk = first_present(d, CANDIDATES["attack_type"])
        if atk:
            grp = d.groupby(atk, dropna=False).size().rename("count").reset_index()
            grp = grp.rename(columns={atk:"Attack type"})
            st.dataframe(grp.sort_values("count", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No attack_type column present.")
        mstate = first_present(d, CANDIDATES["man_state"])
        if mstate:
            grp = d.groupby(mstate, dropna=False).size().rename("count").reset_index()
            grp = grp.rename(columns={mstate:"Man state"})
            st.dataframe(grp.sort_values("count", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No man_state column present.")

    # Ejections drawn
    with tab_draw:
        st.subheader("Ejections drawn")
        d = df_draw.copy()
        grp_team = d.groupby("team", dropna=False).size().rename("count").reset_index().rename(columns={"team":"Team"})
        team_pos = _unique_possessions_by_team(df_full)
        grp_team["rate_per_100_team_poss"] = grp_team.apply(
            lambda r: round(100 * r["count"] / team_pos.get(r["Team"], 0), 1) if team_pos.get(r["Team"], 0) else np.nan,
            axis=1
        )
        st.dataframe(grp_team.sort_values("count", ascending=False), use_container_width=True, hide_index=True)

        if drawn_player_col:
            nums = _to_int_series(d[drawn_player_col]).astype("Int64").astype(str)
            d["Team • #"] = d["team"].astype(str) + " • #" + nums
            grp_p = d.groupby(["Team • #", "team"], dropna=False).size().rename("count").reset_index()
            grp_p = grp_p.rename(columns={"team":"Team"})
            grp_p["rate_per_100_team_poss"] = grp_p.apply(
                lambda r: round(100 * r["count"] / team_pos.get(r["Team"], 0), 1) if team_pos.get(r["Team"], 0) else np.nan,
                axis=1
            )
            st.dataframe(grp_p.drop(columns=["Team"]).sort_values("count", ascending=False),
                         use_container_width=True, hide_index=True)

        if period_col_ns:
            d[period_col_ns] = _to_int_series(d[period_col_ns]).astype("Int64")
            grp2 = d.groupby(period_col_ns, dropna=False).size().rename("count").reset_index()
            grp2 = grp2.rename(columns={period_col_ns:"Period"})
            st.caption("By period")
            st.dataframe(grp2.sort_values("Period"), use_container_width=True, hide_index=True)

    # 5m drawn
    with tab_5m:
        st.subheader("5m drawn")
        d = df_5m.copy()
        grp_team = d.groupby("team", dropna=False).size().rename("count").reset_index().rename(columns={"team":"Team"})
        team_pos = _unique_possessions_by_team(df_full)
        grp_team["rate_per_100_team_poss"] = grp_team.apply(
            lambda r: round(100 * r["count"] / team_pos.get(r["Team"], 0), 1) if team_pos.get(r["Team"], 0) else np.nan,
            axis=1
        )
        st.dataframe(grp_team.sort_values("count", ascending=False), use_container_width=True, hide_index=True)

        if drawn_player_col:
            nums = _to_int_series(d[drawn_player_col]).astype("Int64").astype(str)
            d["Team • #"] = d["team"].astype(str) + " • #" + nums
            grp_p = d.groupby(["Team • #", "team"], dropna=False).size().rename("count").reset_index()
            grp_p = grp_p.rename(columns={"team":"Team"})
            grp_p["rate_per_100_team_poss"] = grp_p.apply(
                lambda r: round(100 * r["count"] / team_pos.get(r["Team"], 0), 1) if team_pos.get(r["Team"], 0) else np.nan,
                axis=1
            )
            st.dataframe(grp_p.drop(columns=["Team"]).sort_values("count", ascending=False),
                         use_container_width=True, hide_index=True)

        if period_col_ns:
            d[period_col_ns] = _to_int_series(d[period_col_ns]).astype("Int64")
            grp2 = d.groupby(period_col_ns, dropna=False).size().rename("count").reset_index()
            grp2 = grp2.rename(columns={period_col_ns:"Period"})
            st.caption("By period")
            st.dataframe(grp2.sort_values("Period"), use_container_width=True, hide_index=True)
