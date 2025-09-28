# -*- coding: utf-8 -*-
import math
import os
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Water Polo Shot Logger", page_icon=":swimmer:", layout="wide")

# ---- Configurable pool geometry (meters)
POOL_LENGTH = 30.0  # distance from goal line to opposite goal line
POOL_WIDTH = 20.0   # sideline to sideline
GOAL_CENTER_X = POOL_WIDTH / 2.0
GOAL_LINE_Y = 0.0   # measure y from attacking goal line outward

DATA_PATH = "data/shots.csv"

st.title("🏊 Water Polo Shot Logger (v1)")
st.caption("Phase 1 tool to log shots consistently. Sliders today; CV-click later.")

# --- Ensure data dir exists
os.makedirs("data", exist_ok=True)

# --- Load existing data (if any)
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame()

with st.sidebar:
    st.header("Game & Context")
    game_id = st.number_input("game_id", min_value=1, step=1, value=1)
    period = st.number_input("period (1–4; OT=5/6)", min_value=1, step=1, value=1)
    time_remaining = st.text_input("time_remaining (mm:ss)", value="")
    attack_type = st.selectbox("attack_type", ["set", "counter", "man-up", "other"])
    shot_type = st.selectbox("shot_type", ["ordinary", "skip", "lob", "backhand", "other"])
    possession_passes = st.number_input("possession_passes", min_value=0, step=1, value=0)
    defender_count = st.number_input("defender_count (within ~2m)", min_value=0, max_value=4, step=1, value=1)
    goalie_position = st.selectbox("goalie_position", ["center", "left", "right", "unknown"])
    shot_result = st.selectbox("shot_result", ["goal (1)", "no goal (0)"])
    attacking_far_goal = st.toggle("Attacking far goal (flip half)", value=False)

st.subheader("Shooter Location")
st.write("Use sliders to place the shooter (meters). 0 ≤ x ≤ width, 0 ≤ y ≤ length (from attacking goal line).")

col1, col2 = st.columns(2)
with col1:
    shooter_x = st.slider("shooter_x (across width, m)", 0.0, POOL_WIDTH, value=GOAL_CENTER_X, step=0.1)
with col2:
    shooter_y = st.slider("shooter_y (from goal line, m)", 0.0, POOL_LENGTH, value=6.0, step=0.1)

# Mirror coords if attacking the far goal
if attacking_far_goal:
    shooter_y_eff = POOL_LENGTH - shooter_y
    shooter_x_eff = POOL_WIDTH - shooter_x
else:
    shooter_y_eff = shooter_y
    shooter_x_eff = shooter_x

# Geometry: distance & angle
dx = shooter_x_eff - GOAL_CENTER_X
dy = shooter_y_eff - GOAL_LINE_Y
distance_to_goal = math.hypot(dx, dy)
angle_to_goal = math.degrees(abs(math.atan2(dx, dy)))

st.markdown("### Derived Geometry")
st.write(f"**distance_to_goal:** {distance_to_goal:.2f} m  |  **angle_to_goal:** {angle_to_goal:.1f}°")

result_map = {"goal (1)": 1, "no goal (0)": 0}
shot_result_int = result_map[shot_result]

row = {
    "logged_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "game_id": int(game_id),
    "period": int(period),
    "time_remaining": time_remaining,
    "shooter_x": round(shooter_x_eff, 2),
    "shooter_y": round(shooter_y_eff, 2),
    "distance_to_goal": round(distance_to_goal, 2),
    "angle_to_goal": round(angle_to_goal, 1),
    "defender_count": int(defender_count),
    "goalie_position": goalie_position,
    "possession_passes": int(possession_passes),
    "attack_type": attack_type,
    "shot_type": shot_type,
    "shot_result": int(shot_result_int),
}

st.markdown("### Current Shot (pending save)")
st.json(row)

c1, c2 = st.columns([1,1])
with c1:
    if st.button("➕ Add shot to dataset", use_container_width=True):
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        st.success(f"Saved to {DATA_PATH} ({len(df)} rows)")
with c2:
    if st.button("🗑️ Undo last row", use_container_width=True, disabled=len(df)==0):
        if len(df) > 0:
            df = df.iloc[:-1].copy()
            df.to_csv(DATA_PATH, index=False)
            st.warning("Removed last row")

st.markdown("### Dataset Preview")
if len(df):
    st.dataframe(df.tail(25), use_container_width=True, height=300)
else:
    st.info("No rows yet. Log your first shot with the button above.")
