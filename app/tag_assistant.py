# app/tag_assistant.py
# -*- coding: utf-8 -*-
"""
Tagging Assistant (merged)
- Preserves original behavior:
  * Loads app/shots.csv
  * Loads app/reports/frames/frames_manifest.csv
  * Shots-only workflow
  * Three keyframes (t-2s, t0, t+2s) preview if available
  * Nudge timecode +/- and save corrections
  * Writes:
      - app/reports/corrections/timestamp_fixes.csv
      - app/reports/corrections/corrections.csv  (possession_id, field, value)
- Adds:
  * Optional filters (team, game_id, free-text)
  * More common corrections (defender_count, goalie_present, shot_type, man_state, turnover_type)
  * Safer handling when columns are missing
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Paths ----------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
FRAMES_DIR = APP_DIR / "reports" / "frames"
CORR_DIR = APP_DIR / "reports" / "corrections"
CORR_DIR.mkdir(parents=True, exist_ok=True)

SHOTS_CSV = APP_DIR / "shots.csv"
MANIFEST_CSV = FRAMES_DIR / "frames_manifest.csv"

# -------------- Page -------------------
st.set_page_config(page_title="Tagging Assistant", layout="wide")
st.title("Tagging Assistant")

# -------------- Time helpers -----------
def seconds_to_mmss(t: float) -> str:
    if t is None or np.isnan(t):
        return ""
    if t < 0:
        t = 0
    m = int(t // 60)
    s = t - m * 60
    # Keep one decimal (mm:ss.s) for precise nudges
    return f"{m:02d}:{s:04.1f}"

def mmss_to_seconds(s: str) -> float:
    try:
        s = str(s).strip()
        parts = s.split(":")
        if len(parts) == 2:
            m, sec = int(parts[0]), float(parts[1])
            return m * 60 + sec
        elif len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + sec
        else:
            return float(s)
    except Exception:
        return math.nan

# -------------- Load -------------------
if not SHOTS_CSV.exists():
    st.error(f"Missing {SHOTS_CSV}")
    st.stop()

if not MANIFEST_CSV.exists():
    st.error(f"Missing {MANIFEST_CSV}. Re-run keyframe extraction first.")
    st.stop()

shots_all = pd.read_csv(SHOTS_CSV, dtype=str)
man = pd.read_csv(MANIFEST_CSV, dtype=str)

# Manifest sanity
needed = {"possession_id", "video_file"}
if not needed.issubset(set(man.columns)):
    st.error(
        "frames_manifest.csv is missing possession_id and/or video_file. "
        "Update tools/extract_keyframes.py per instructions and re-run."
    )
    st.stop()

# Shots only
evt = shots_all.get("event_type", "").astype(str).str.lower() if "event_type" in shots_all.columns else ""
shots = shots_all[evt.eq("shot")] if isinstance(evt, pd.Series) else shots_all.copy()

# Current shot times
if "video_timestamp_mmss" in shots.columns:
    shots["t_s"] = shots["video_timestamp_mmss"].map(mmss_to_seconds)
else:
    shots["t_s"] = np.nan

# Columns we might use
team_col = "our_team_name" if "our_team_name" in shots.columns else ("team" if "team" in shots.columns else None)
player_col = "player_number" if "player_number" in shots.columns else None
outcome_col = "shot_result" if "shot_result" in shots.columns else ("outcome" if "outcome" in shots.columns else None)
game_col = "game_id" if "game_id" in shots.columns else None
period_col = "period" if "period" in shots.columns else None

# ----------------- Filters -------------
with st.expander("Filters", expanded=True):
    c1, c2, c3 = st.columns([1, 1, 2])
    team_choice = "(all)"
    game_choice = "(all)"
    txt = ""

    if team_col:
        teams = ["(all)"] + sorted([t for t in shots[team_col].dropna().unique().tolist() if str(t).strip() != ""])
        team_choice = c1.selectbox("Team", teams, index=0)
    if game_col:
        games = ["(all)"] + sorted([g for g in shots[game_col].dropna().unique().tolist() if str(g).strip() != ""])
        game_choice = c2.selectbox("Game", games, index=0)
    txt = c3.text_input("Find (possession_id / player / time / notes)", "")

df_list = shots.copy()
if team_col and team_choice != "(all)":
    df_list = df_list[df_list[team_col] == team_choice]
if game_col and game_choice != "(all)":
    df_list = df_list[df_list[game_col] == game_choice]
if txt.strip():
    key = txt.strip().lower()
    def _match_row(r):
        return (
            key in str(r.get("possession_id","")).lower()
            or key in str(r.get("player_number","")).lower()
            or key in str(r.get("video_timestamp_mmss","")).lower()
            or key in str(r.get("notes","")).lower()
        )
    df_list = df_list[df_list.apply(_match_row, axis=1)]

# ----------------- Selector ------------
left, right = st.columns([1, 3])

with left:
    st.subheader("Find shot")
    if df_list.empty:
        st.info("No shots found after filters.")
        st.stop()

    # Build a display label
    def _label_row(r):
        pid = r.get("possession_id", "")
        num = str(r.get(player_col, "")) if player_col else ""
        per = str(r.get(period_col, "")) if period_col else ""
        tmmss = r.get("video_timestamp_mmss", "")
        base = f"{pid}"
        parts = []
        if num: parts.append(f"#{num}")
        if per: parts.append(f"Q{per}")
        if tmmss: parts.append(str(tmmss))
        if parts:
            base += " • " + " • ".join(parts)
        return base

    options = list(range(len(df_list)))
    sel_idx = st.selectbox("Shot", options=options, format_func=lambda i: _label_row(df_list.iloc[i]), index=0)

    # Chosen shot row
    shot = df_list.iloc[sel_idx: sel_idx+1].copy().iloc[0]
    poss_id = str(shot.get("possession_id", ""))

with right:
    st.subheader("Review frames & adjust")

    # Find keyframes for this possession
    ms = man[man["possession_id"] == poss_id].copy()

    def img(tag: str):
        row = ms[ms["context"] == tag]
        if row.empty:
            return None
        p = Path(row.iloc[0].get("frame_path", ""))
        return p if (p and p.exists()) else None

    img_tm = img("tm")
    img_t0 = img("t0")
    img_tp = img("tp")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("t−2s")
        if img_tm: st.image(str(img_tm))
        else: st.write("—")
    with c2:
        st.caption("t0")
        if img_t0: st.image(str(img_t0))
        else: st.write("—")
    with c3:
        st.caption("t+2s")
        if img_tp: st.image(str(img_tp))
        else: st.write("—")

    st.markdown("---")

    # Current values
    cur_ts = str(shot.get("video_timestamp_mmss", ""))
    cur_t = shot.get("t_s", np.nan)
    if isinstance(cur_t, (int, float)) and not np.isnan(cur_t):
        st.write(f"**Current timecode:** {cur_ts}  (≈ {cur_t:.2f}s)")
    else:
        st.write(f"**Current timecode:** {cur_ts}")

    # Nudge controls
    nudge = st.radio(
        "Adjust time by",
        options=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        index=3, horizontal=True,
        format_func=lambda x: f"{x:+.1f}s"
    )
    base_t = (float(cur_t) if isinstance(cur_t, (int, float)) and not np.isnan(cur_t) else 0.0)
    proposed_t = max(0.0, base_t + float(nudge))
    proposed_tc = seconds_to_mmss(proposed_t)
    st.write(f"**Proposed timecode:** {proposed_tc}  (Δ {float(nudge):+0.1f}s)")

    # Optional metadata fixes
    st.markdown("**Optional metadata fixes**")
    new_vals = {}

    if player_col:
        cur_num = str(shot.get(player_col, "")).strip()
        new_vals[player_col] = st.text_input("Player #", value=cur_num, key="player_number_in")

    if outcome_col:
        cur_out = str(shot.get(outcome_col, "")).strip()
        new_vals[outcome_col] = st.text_input("Outcome", value=cur_out, key="outcome_in")

    # Common fields often corrected
    if "defender_count" in shots.columns:
        cur_defs = str(shot.get("defender_count", "")).strip()
        new_vals["defender_count"] = st.text_input("Defender count (int)", value=cur_defs, key="defenders_in")
    if "goalie_present" in shots.columns:
        cur_gk = str(shot.get("goalie_present", "")).strip()
        new_vals["goalie_present"] = st.text_input('Goalie present ("true"/"false")', value=cur_gk, key="gk_in")
    if "shot_type" in shots.columns:
        cur_st = str(shot.get("shot_type", "")).strip()
        new_vals["shot_type"] = st.text_input("Shot type", value=cur_st, key="shot_type_in")
    if "man_state" in shots.columns:
        cur_man = str(shot.get("man_state", "")).strip()
        new_vals["man_state"] = st.text_input('Man state (e.g., "6v5")', value=cur_man, key="man_in")
    if "turnover_type" in shots_all.columns:
        # present in whole dataset, not necessarily in shots-only subset, but allow correction anyway
        cur_to = str(shots_all[shots_all.get("possession_id","")==poss_id].get("turnover_type", np.nan).iloc[0]) \
                 if "turnover_type" in shots_all.columns and (shots_all.get("possession_id","")==poss_id).any() else ""
        new_vals["turnover_type"] = st.text_input("Turnover type (if needed)", value=cur_to, key="to_in")

    notes = st.text_input("Notes (for timestamp_fixes.csv)", value="", key="notes_in")

    # Save corrections
    if st.button("Save corrections", type="primary"):
        # 1) timestamp_fixes.csv (for audit/review)
        ts_path = CORR_DIR / "timestamp_fixes.csv"
        if ts_path.exists():
            ts_df = pd.read_csv(ts_path, dtype=str)
        else:
            ts_df = pd.DataFrame(columns=["possession_id", "new_video_timestamp_mmss", "delta_s", "notes"])
        ts_row = {
            "possession_id": poss_id,
            "new_video_timestamp_mmss": proposed_tc,
            "delta_s": f"{float(nudge):+0.1f}",
            "notes": notes,
        }
        ts_df = pd.concat([ts_df, pd.DataFrame([ts_row])], ignore_index=True)
        ts_df.to_csv(ts_path, index=False)

        # 2) corrections.csv (for normalize_and_export.py --corrections ...)
        corr_path = CORR_DIR / "corrections.csv"
        if corr_path.exists():
            corr_df = pd.read_csv(corr_path, dtype=str)
        else:
            corr_df = pd.DataFrame(columns=["possession_id", "field", "value"])

        corr_rows = [{"possession_id": poss_id, "field": "video_timestamp_mmss", "value": proposed_tc}]

        # Only include changed (non-empty and different) fields
        for field, new_val in new_vals.items():
            if new_val is None:
                continue
            new_val_str = str(new_val).strip()
            if new_val_str == "":
                continue
            old_val = str(shot.get(field, "")).strip()
            if new_val_str != old_val:
                corr_rows.append({"possession_id": poss_id, "field": field, "value": new_val_str})

        if corr_rows:
            corr_df = pd.concat([corr_df, pd.DataFrame(corr_rows)], ignore_index=True)
            corr_df.to_csv(corr_path, index=False)
            st.success(
                f"Saved {len(corr_rows)} correction(s). "
                f"Re-run normalize/export with --corrections for modeling. "
                f"Re-run keyframes later if you shift times broadly."
            )
        else:
            st.info("No changes to save.")

    st.caption(
        "Outputs: app/reports/corrections/timestamp_fixes.csv and app/reports/corrections/corrections.csv"
    )
