# app/shot_logger.py
# Shot & Possession Logger — pre-predictive, stable UI
# Writes normalized coordinates:
#   shooter_x_norm = (x_m + 10) / 20  (0=left post line, 0.5=center, 1=right post line)
#   shooter_y_norm = y_m / 15         (0=goal line, 1=15m)
# Displays meters live; stores normalized in CSV to match prior rows.

import math
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

# ---------- Paths ----------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "shots.csv"

# ---------- Geometry ----------
# UI in meters: x ∈ [-10, +10], y ∈ [0, 15]
# Stored normalized: x_n ∈ [0,1], y_n ∈ [0,1]
FRONTCOURT_LENGTH_M = 15.0
POOL_WIDTH_M = 20.0  # -10..+10

# ---------- Schema (must match exporter) ----------
COLUMNS = [
    "possession_id", "game_id",
    "our_team_name", "opponent_team_name",
    "our_team_level", "opponent_team_level",
    "period", "time_remaining",
    "man_state", "event_type",
    "turnover_type", "turnover_player_number",
    "drawn_by_player_number",
    "shooter_x", "shooter_y",              # NOTE: stored as normalized fractions (0..1)
    "defender_count",
    "goalie_present", "goalie_distance_m", "goalie_lateral",
    "possession_passes",
    "attack_type", "shot_type", "shot_result",
    "shooter_handedness", "player_number",
    "video_file", "video_timestamp_mmss",
]

# ---------- Helpers ----------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path, dtype=str)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[COLUMNS]
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

def ensure_headers(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)

def next_possession_id(df: pd.DataFrame, game_id: str) -> str:
    if df.empty:
        return f"{game_id}_P0001"
    m = df[df["game_id"] == str(game_id)]
    if m.empty:
        return f"{game_id}_P0001"
    seq = []
    for sid in m["possession_id"].dropna().astype(str):
        if "_P" in sid:
            tail = sid.split("_P")[-1]
            if tail.isdigit():
                seq.append(int(tail))
    nxt = (max(seq) + 1) if seq else 1
    return f"{game_id}_P{nxt:04d}"

def compute_distance_angle(x_m: float, y_m: float) -> Tuple[float, float]:
    dx, dy = x_m, y_m
    dist = math.hypot(dx, dy)
    ang = 90.0 if dy == 0 else math.degrees(math.atan2(abs(dx), dy))
    signed = -ang if dx < 0 else ang
    return round(dist, 3), round(signed, 2)

def to_str(x):
    return "" if x is None else str(x)

# ---------- UI ----------
st.set_page_config(page_title="Water Polo Shot Logger", layout="wide")
st.title("🏐 Water Polo — Shot & Possession Logger (normalized save)")

# Sidebar: session
with st.sidebar:
    st.markdown("### Session / Game")
    game_id = st.text_input("game_id", value="1")
    our_team_name = st.text_input("our_team_name", value="")
    opponent_team_name = st.text_input("opponent_team_name", value="")
    our_team_level = st.selectbox(
        "our_team_level",
        ["ILHS","IL Club","CA Club","ODP Zone","ODP Academy","College","Other",""],
        index=1
    )
    opponent_team_level = st.selectbox(
        "opponent_team_level",
        ["ILHS","IL Club","CA Club","ODP Zone","ODP Academy","College","Other",""],
        index=1
    )
    period = st.selectbox("period (optional)", ["", "1","2","3","4","OT1","OT2"], index=0)
    time_remaining = st.text_input("time_remaining (MM:SS, optional)", value="")

# Top row: event + man state + passes
c1, c2, c3 = st.columns([1.2, 1.2, 1])
with c1:
    event_type = st.radio(
        "event_type",
        ["shot", "turnover", "ejection_drawn", "5m_drawn"],
        horizontal=True
    )
with c2:
    man_state = st.selectbox(
        "man_state (offense perspective)",
        ["6v6","6v5","6v4","7v6","6v7","5v5","Other"],
        index=0
    )
with c3:
    possession_passes = st.number_input("possession_passes", min_value=0, max_value=20, value=0, step=1)

# Video row
v1, v2 = st.columns([1,1])
with v1:
    video_file = st.text_input("video_file (e.g., game1.mp4 / IMG_0001.MOV)", value="")
with v2:
    video_timestamp_mmss = st.text_input("video_timestamp_mmss (MM:SS)", value="")

# Attack type (includes perimeter)
attack_type = st.selectbox(
    "attack_type",
    ["set offense","perimeter","counterattack","man-up","broken play","drive","fast break","other"],
    index=1
)

st.markdown("---")

# Conditional sections
turnover_type = ""
turnover_player_number = ""
drawn_by_player_number = ""
# Shot fields (UI in meters)
x_m = None
y_m = None
defender_count = None
goalie_present = ""
goalie_distance_m = ""
goalie_lateral = ""
shot_type = ""
shot_result = ""
shooter_handedness = ""
player_number = ""

if event_type == "turnover":
    t1, t2 = st.columns([1,1])
    with t1:
        turnover_type = st.selectbox(
            "turnover_type",
            ["steal","bad pass","ball under","shot clock violation","offensive foul","throw to corner (deliberate)","other"],
            index=1
        )
    with t2:
        turnover_player_number = st.text_input("turnover_player_number (optional; int)", value="")
elif event_type in ["ejection_drawn", "5m_drawn"]:
    drawn_by_player_number = st.text_input("drawn_by_player_number (optional; int)", value="")
elif event_type == "shot":
    st.markdown("#### Shot details")
    a, b = st.columns([1,1])
    with a:
        x_m = st.selectbox(
            "x (meters from goal center; -10..+10; left negative, right positive)",
            options=[i for i in range(-10, 11)],
            index=10  # 0m by default
        )
        y_m = st.selectbox(
            "y (meters from goal line; 0..15)",
            options=[i for i in range(0, 16)],
            index=7
        )
        defender_count = st.selectbox("defender_count (within 1m)", options=[0,1,2,3,4,5], index=1)
        shooter_handedness = st.selectbox("shooter_handedness", ["right","left","unknown"], index=0)
        player_number = st.text_input("shooter player_number (int)", value="")
    with b:
        goalie_present = st.selectbox("goalie_present?", ["true","false",""], index=0)
        goalie_distance_m = st.selectbox("goalie_distance_m (m)", ["", "0","0.5","1","1.5","2"], index=1 if goalie_present=="true" else 0)
        goalie_lateral = st.selectbox("goalie_lateral", ["left","center","right",""], index=1 if goalie_present=="true" else 3)
        shot_type = st.selectbox("shot_type", ["standard","lob","backhand","sweep","wet","deflection","5m","other"], index=0)
        shot_result = st.selectbox(
            "shot_result",
            ["goal","blocked (GK)","field block","post","crossbar","miss wide","miss high","other"],
            index=0
        )

    # Live geometry (meters)
    dist_m, ang_deg = compute_distance_angle(float(x_m), float(y_m))
    st.caption(f"Inferred geometry — distance: **{dist_m:.2f} m**, angle: **{ang_deg:.1f}°** (signed; right + / left -)")

st.markdown("---")

# Save section
df = safe_read_csv(DATA_PATH)
suggested_id = next_possession_id(df, game_id.strip() if game_id else "1")
possession_id = st.text_input("possession_id", value=suggested_id)

def build_row():
    # Convert meters → normalized (on save)
    if event_type == "shot":
        x_norm = round((float(x_m) + 10.0) / 20.0, 3)  # -10→0.000, 0→0.500, +10→1.000
        y_norm = round(float(y_m) / 15.0, 3)          # 0→0.000, 15→1.000
        x_out = f"{x_norm:.3f}"
        y_out = f"{y_norm:.3f}"
    else:
        x_out = ""
        y_out = ""

    row = {
        "possession_id": to_str(possession_id),
        "game_id": to_str(game_id),

        "our_team_name": to_str(our_team_name),
        "opponent_team_name": to_str(opponent_team_name),
        "our_team_level": to_str(our_team_level),
        "opponent_team_level": to_str(opponent_team_level),

        "period": to_str(period),
        "time_remaining": to_str(time_remaining),

        "man_state": to_str(man_state),
        "event_type": to_str(event_type),

        "turnover_type": to_str(turnover_type if event_type == "turnover" else ""),
        "turnover_player_number": to_str(turnover_player_number if event_type == "turnover" else ""),

        "drawn_by_player_number": to_str(drawn_by_player_number if event_type in ["ejection_drawn","5m_drawn"] else ""),

        # STORED AS NORMALIZED FRACTIONS:
        "shooter_x": x_out,
        "shooter_y": y_out,

        "defender_count": to_str(defender_count if event_type == "shot" else ""),

        "goalie_present": to_str(goalie_present if event_type == "shot" else ""),
        "goalie_distance_m": to_str(goalie_distance_m if (event_type == "shot" and goalie_present == "true") else ""),
        "goalie_lateral": to_str(goalie_lateral if (event_type == "shot" and goalie_present == "true") else ""),

        "possession_passes": to_str(possession_passes),

        "attack_type": to_str(attack_type),
        "shot_type": to_str(shot_type if event_type == "shot" else ""),
        "shot_result": to_str(shot_result if event_type == "shot" else ""),

        "shooter_handedness": to_str(shooter_handedness if event_type == "shot" else ""),
        "player_number": to_str(player_number if event_type == "shot" else ""),

        "video_file": to_str(video_file),
        "video_timestamp_mmss": to_str(video_timestamp_mmss),
    }
    for c in COLUMNS:
        row.setdefault(c, "")
    return row

cL, cR = st.columns([1,1])
with cL:
    if st.button("💾 Save row", type="primary"):
        try:
            ensure_headers(DATA_PATH)
            row = build_row()
            existing = safe_read_csv(DATA_PATH)
            out = pd.concat([existing, pd.DataFrame([row], columns=COLUMNS)], ignore_index=True)
            out.to_csv(DATA_PATH, index=False)
            st.success(f"Saved → {DATA_PATH.name} (possession_id={row['possession_id']})")
            st.rerun()
        except Exception as e:
            st.error(f"Save failed: {e}")

with cR:
    if st.button("🔎 Show last 10"):
        st.dataframe(safe_read_csv(DATA_PATH).tail(10), use_container_width=True)

st.caption("Notes: x/y stored as normalized fractions (x: 0=goalie-left post line, 0.5=center, 1=goalie-right; y: 0=goal line, 1=15m). Period/time are optional. Turnovers/ejections/5m keep shot fields blank.")
