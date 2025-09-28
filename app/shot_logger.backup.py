# app/shot_logger.py
# Streamlit Shot Logger — resilient version (no model dependency)
# - Writes to app/shots.csv, creating it with headers if missing
# - Matches exporter EXPECTED_COLS
# - Shows inferred distance/angle live for shots
# - Optional fields won't block saving

import math
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

# -------- Paths --------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "shots.csv"

# -------- Geometry constants --------
FRONTCOURT_LENGTH_M = 15.0  # y: 0 at goal line to 15m
POOL_WIDTH_M = 20.0         # x normalized across -10..+10 → 20m full width

# -------- Schema (must align with tools/export_all_views.py) --------
EXPECTED_COLS = [
    "possession_id", "game_id",
    "our_team_name", "opponent_team_name",
    "our_team_level", "opponent_team_level",
    "period", "time_remaining",
    "man_state", "event_type",
    "turnover_type", "turnover_player_number",
    "drawn_by_player_number",
    "shooter_x", "shooter_y",
    "defender_count",
    "goalie_present", "goalie_distance_m", "goalie_lateral",
    "possession_passes",
    "attack_type", "shot_type", "shot_result",
    "shooter_handedness", "player_number",
    "video_file", "video_timestamp_mmss",
]

# -------- Utilities --------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=EXPECTED_COLS)
    try:
        df = pd.read_csv(path, dtype=str)
        # add any missing cols
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[EXPECTED_COLS]
    except Exception:
        # If file exists but is unreadable/empty, reset it
        return pd.DataFrame(columns=EXPECTED_COLS)

def ensure_headers(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=EXPECTED_COLS).to_csv(path, index=False)

def next_possession_id(df: pd.DataFrame, game_id: str) -> str:
    # format: "<game_id>_P####"
    if df.empty:
        return f"{game_id}_P0001"
    m = df[df["game_id"] == str(game_id)].copy()
    if m.empty:
        return f"{game_id}_P0001"
    # extract numeric tail
    seq = []
    for sid in m["possession_id"].astype(str):
        if "_P" in sid:
            tail = sid.split("_P")[-1]
            if tail.isdigit():
                seq.append(int(tail))
    nxt = (max(seq) + 1) if seq else 1
    return f"{game_id}_P{nxt:04d}"

def compute_distance_angle(x: float, y: float) -> Tuple[float, float]:
    """
    x in [-10..+10] (0 is goal center; <0 goalie-left, >0 goalie-right)
    y in [0..15] (0 at goal line, increasing outward)
    Returns (distance_m, signed_angle_deg)
    """
    if x is None or y is None:
        return (float("nan"), float("nan"))
    dx = x  # already meters left/right from center
    dy = y  # meters from goal line
    distance = math.hypot(dx, dy)
    base_angle_rad = math.atan2(abs(dx), dy) if dy != 0 else (math.pi / 2)
    base_angle_deg = math.degrees(base_angle_rad)
    signed_deg = -base_angle_deg if dx < 0 else base_angle_deg
    return (round(distance, 3), round(signed_deg, 2))

# -------- UI --------
st.set_page_config(page_title="Water Polo Shot Logger", layout="wide")
st.title("🏐 Water Polo — Shot & Possession Logger (stable)")

with st.sidebar:
    st.markdown("### Session/Game")
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
    period = st.selectbox("period (optional)", ["", "1", "2", "3", "4", "OT1", "OT2"])
    time_remaining = st.text_input("time_remaining (MM:SS, optional)", value="")

st.markdown("---")
st.subheader("Event")

event_type = st.radio(
    "event_type",
    ["shot", "turnover", "ejection_drawn", "5m_drawn"],
    horizontal=True
)

man_state = st.selectbox(
    "man_state (offense perspective)",
    ["6v6","6v5","6v4","7v6","6v7","5v5","Other"],
    index=0
)

attack_type = st.selectbox(
    "attack_type",
    ["set offense","counterattack","man-up","broken play","drive","fast break","other"],
    index=0
)

possession_passes = st.number_input("possession_passes (count)", min_value=0, max_value=20, value=0, step=1)

st.markdown("---")
st.subheader("Video")
video_file = st.text_input("video_file (e.g., game1.mp4 or IMG_0001.MOV)", value="")
video_timestamp_mmss = st.text_input("video_timestamp_mmss (MM:SS)", value="")

# -------- Conditional sections --------
turnover_type = ""
turnover_player_number = ""
drawn_by_player_number = ""

if event_type == "turnover":
    st.subheader("Turnover details")
    turnover_type = st.selectbox(
        "turnover_type",
        ["steal","bad pass","ball under","shot clock violation","offensive foul","throw to corner (deliberate)","other"],
        index=1
    )
    turnover_player_number = st.text_input("player # most responsible (optional; int)", value="")
elif event_type in ["ejection_drawn", "5m_drawn"]:
    st.subheader("Foul drawn details")
    drawn_by_player_number = st.text_input("drawn_by_player_number (optional; int)", value="")

# Shot section
shooter_x = None
shooter_y = None
defender_count = None
goalie_present = None
goalie_distance_m = None
goalie_lateral = ""
shot_type = ""
shot_result = ""
shooter_handedness = ""
player_number = ""

if event_type == "shot":
    st.subheader("Shot details")
    cols = st.columns(2)
    with cols[0]:
        # x from -10..+10 in 1m steps
        shooter_x = st.selectbox(
            "x (meters from goal center; -10..+10; left negative, right positive)",
            options=[i for i in range(-10, 11)],
            index=10  # default 0
        )
        # y from 0..15 in 1m steps
        shooter_y = st.selectbox(
            "y (meters from goal line; 0..15)",
            options=[i for i in range(0, 16)],
            index=7
        )
        defender_count = st.selectbox(
            "defender_count (within 1m)",
            options=[0,1,2,3,4,5],
            index=1
        )
        shooter_handedness = st.selectbox("shooter_handedness", ["right","left","unknown"], index=0)
        player_number = st.text_input("shooter player_number (int)", value="")
    with cols[1]:
        goalie_present = st.selectbox("goalie_present?", ["true","false",""], index=0)
        goalie_distance_m = st.selectbox("goalie_distance_m (m)", [ "", "0", "0.5", "1", "1.5", "2" ], index=1 if goalie_present=="true" else 0)
        goalie_lateral = st.selectbox("goalie_lateral", ["left","center","right",""], index=1 if goalie_present=="true" else 3)
        shot_type = st.selectbox(
            "shot_type",
            ["standard","lob","backhand","sweep","wet","deflection","5m","other"],
            index=0
        )
        shot_result = st.selectbox(
            "shot_result",
            ["goal","blocked (GK)","field block","post","crossbar","miss wide","miss high","other"],
            index=0
        )

    # Live geometry readout
    d_m, a_deg = compute_distance_angle(float(shooter_x), float(shooter_y))
    st.caption(f"Inferred geometry — distance: **{d_m:.2f} m**, angle: **{a_deg:.1f}°** (signed: right + / left -)")

# -------- Save logic --------
st.markdown("---")
st.subheader("Save")

df = safe_read_csv(DATA_PATH)

# Suggest next possession_id for the game
suggested_poss_id = next_possession_id(df, game_id.strip() if game_id else "1")
possession_id = st.text_input("possession_id (auto-suggested)", value=suggested_poss_id)

# Build row dict (strings only; keep blanks for optional)
def to_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x)):
        return ""
    return str(x)

def build_row_dict():
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

        "turnover_type": to_str(turnover_type if event_type=="turnover" else ""),
        "turnover_player_number": to_str(turnover_player_number if event_type=="turnover" else ""),

        "drawn_by_player_number": to_str(drawn_by_player_number if event_type in ["ejection_drawn","5m_drawn"] else ""),

        "shooter_x": to_str(shooter_x if event_type=="shot" else ""),
        "shooter_y": to_str(shooter_y if event_type=="shot" else ""),

        "defender_count": to_str(defender_count if event_type=="shot" else ""),

        "goalie_present": to_str(goalie_present if event_type=="shot" else ""),
        "goalie_distance_m": to_str(goalie_distance_m if (event_type=="shot" and goalie_present=="true") else ""),
        "goalie_lateral": to_str(goalie_lateral if (event_type=="shot" and goalie_present=="true") else ""),

        "possession_passes": to_str(possession_passes),

        "attack_type": to_str(attack_type),
        "shot_type": to_str(shot_type if event_type=="shot" else ""),
        "shot_result": to_str(shot_result if event_type=="shot" else ""),

        "shooter_handedness": to_str(shooter_handedness if event_type=="shot" else ""),
        "player_number": to_str(player_number if event_type=="shot" else ""),

        "video_file": to_str(video_file),
        "video_timestamp_mmss": to_str(video_timestamp_mmss),
    }
    # Guarantee all expected keys exist (future-proof)
    for c in EXPECTED_COLS:
        if c not in row:
            row[c] = ""
    return row

col_left, col_right = st.columns([1,1])
with col_left:
    if st.button("💾 Save row", type="primary"):
        try:
            ensure_headers(DATA_PATH)
            row = build_row_dict()
            # Append safely
            existing = safe_read_csv(DATA_PATH)
            new_row_df = pd.DataFrame([row], columns=EXPECTED_COLS)
            updated = pd.concat([existing, new_row_df], ignore_index=True)
            updated.to_csv(DATA_PATH, index=False)
            st.success(f"Row saved to {DATA_PATH.name} (poss_id={row['possession_id']})")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to save row: {e}")

with col_right:
    if st.button("🔎 Show last 10 saved"):
        existing = safe_read_csv(DATA_PATH)
        st.dataframe(existing.tail(10))

st.markdown("---")
st.caption("Tips: period and time_remaining are optional. For turnovers/ejections/5m, shot fields are intentionally blank. x/y are dropdowns in meters with y=0 at goal line and x=0 at goal center (left negative, right positive).")
