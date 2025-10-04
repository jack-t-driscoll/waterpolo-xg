# app/tag_assistant.py
from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
FRAMES_DIR = APP_DIR / "reports" / "frames"
CORR_DIR = APP_DIR / "reports" / "corrections"
CORR_DIR.mkdir(parents=True, exist_ok=True)

SHOTS_CSV = APP_DIR / "shots.csv"
MANIFEST_CSV = FRAMES_DIR / "frames_manifest.csv"

st.set_page_config(page_title="Tagging Assistant", layout="wide")
st.title("Tagging Assistant")

def seconds_to_mmss(t: float) -> str:
    if t is None or np.isnan(t):
        return ""
    if t < 0:
        t = 0
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:05.2f}".replace(".", ":") if False else f"{m:02d}:{s:04.1f}"

def mmss_to_seconds(s: str) -> float:
    try:
        s = str(s).strip()
        parts = s.split(":")
        if len(parts) == 2:
            m, sec = int(parts[0]), float(parts[1])
            return m*60 + sec
        elif len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            return h*3600 + m*60 + sec
        else:
            return float(s)
    except Exception:
        return math.nan

# Load data
if not SHOTS_CSV.exists():
    st.error(f"Missing {SHOTS_CSV}")
    st.stop()

if not MANIFEST_CSV.exists():
    st.error(f"Missing {MANIFEST_CSV}. Re-run keyframe extraction first.")
    st.stop()

shots = pd.read_csv(SHOTS_CSV, dtype=str)
man = pd.read_csv(MANIFEST_CSV, dtype=str)

# Require IDs in manifest (added by the tiny tweak)
needed = {"possession_id","video_file"}
if not needed.issubset(set(man.columns)):
    st.error("frames_manifest.csv is missing possession_id and/or video_file. "
             "Update tools/extract_keyframes.py per instructions and re-run.")
    st.stop()

# Shots only
evt = shots.get("event_type", "").astype(str).str.lower() if "event_type" in shots.columns else ""
shots = shots[evt.eq("shot")] if isinstance(evt, pd.Series) else shots

# Current shot times
if "video_timestamp_mmss" in shots.columns:
    shots["t_s"] = shots["video_timestamp_mmss"].map(mmss_to_seconds)
else:
    shots["t_s"] = np.nan

# Select a team and shot to tag
team_col = "our_team_name" if "our_team_name" in shots.columns else ("team" if "team" in shots.columns else None)
player_col = "player_number" if "player_number" in shots.columns else None
outcome_col = "shot_result" if "shot_result" in shots.columns else ("outcome" if "outcome" in shots.columns else None)

left, right = st.columns([1,3])

with left:
    st.subheader("Find shot")
    if team_col:
        teams = ["(all)"] + sorted(shots[team_col].dropna().unique().tolist())
        team_choice = st.selectbox("Team", teams, index=0)
    else:
        team_choice = "(all)"

    df_list = shots.copy()
    if team_col and team_choice != "(all)":
        df_list = df_list[df_list[team_col] == team_choice]

    # Build a display name
    disp = []
    for _, r in df_list.iterrows():
        pid = r.get("possession_id", "")
        num = str(r.get(player_col, "")) if player_col else ""
        per = str(r.get("period", "")) if "period" in r else ""
        tmmss = r.get("video_timestamp_mmss", "")
        label = f"{pid} • #{num} • Q{per} • {tmmss}"
        disp.append(label)
    if not disp:
        st.info("No shots found after filters.")
        st.stop()

    sel_idx = st.selectbox("Shot", options=list(range(len(disp))), format_func=lambda i: disp[i])
    shot = df_list.iloc[sel_idx: sel_idx+1].copy()
    # Fetch the same row from main shots to ensure all columns
    poss_id = str(shot.iloc[0].get("possession_id",""))
    shot = shots[shots["possession_id"] == poss_id].iloc[0]

with right:
    st.subheader("Review frames & adjust")

    # Find frames for this shot
    ms = man[man["possession_id"] == poss_id].copy()
    # Expect up to tm/t0/tp
    def img(tag):
        row = ms[ms["context"] == tag]
        if row.empty: return None
        p = Path(row.iloc[0]["frame_path"])
        return p if p.exists() else None

    img_tm = img("tm")
    img_t0 = img("t0")
    img_tp = img("tp")

    c1,c2,c3 = st.columns(3)
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
    cur_ts = shot.get("video_timestamp_mmss", "")
    cur_t = shot.get("t_s", np.nan)
    st.write(f"**Current timecode:** {cur_ts}  (≈ {cur_t:.2f}s)" if isinstance(cur_t, (int,float)) and not np.isnan(cur_t) else f"**Current timecode:** {cur_ts}")

    # Nudge controls
    nudge = st.radio("Adjust time by", options=[-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0], index=3, horizontal=True, format_func=lambda x: f"{x:+.1f}s")
    proposed_t = (cur_t if isinstance(cur_t,(int,float)) and not np.isnan(cur_t) else 0.0) + nudge
    proposed_t = max(0.0, proposed_t)
    proposed_tc = seconds_to_mmss(proposed_t)
    st.write(f"**Proposed timecode:** {proposed_tc}  (Δ {nudge:+.1f}s)")

    # Optional metadata fixes
    st.markdown("**Optional metadata fixes**")
    new_num = None
    new_out = None
    if player_col:
        cur_num = str(shot.get(player_col,""))
        new_num = st.text_input("Player #", value=cur_num)
    if outcome_col:
        cur_out = str(shot.get(outcome_col,""))
        new_out = st.text_input("Outcome", value=cur_out)

    notes = st.text_input("Notes", value="")

    # Save corrections
    if st.button("Save corrections", type="primary"):
        # timestamp_fixes.csv (for audit)
        ts_path = CORR_DIR / "timestamp_fixes.csv"
        ts_df = pd.read_csv(ts_path, dtype=str) if ts_path.exists() else pd.DataFrame(columns=["possession_id","new_video_timestamp_mmss","delta_s","notes"])
        new_row = {
            "possession_id": poss_id,
            "new_video_timestamp_mmss": proposed_tc,
            "delta_s": f"{nudge:+.1f}",
            "notes": notes,
        }
        ts_df = pd.concat([ts_df, pd.DataFrame([new_row])], ignore_index=True)
        ts_df.to_csv(ts_path, index=False)

        # corrections.csv (for normalize_and_export.py --corrections)
        corr_path = CORR_DIR / "corrections.csv"
        corr_df = pd.read_csv(corr_path, dtype=str) if corr_path.exists() else pd.DataFrame(columns=["possession_id","field","value"])

        corr_rows = [{"possession_id": poss_id, "field": "video_timestamp_mmss", "value": proposed_tc}]
        if player_col and new_num is not None and new_num.strip() != str(shot.get(player_col,"")).strip():
            corr_rows.append({"possession_id": poss_id, "field": player_col, "value": new_num.strip()})
        if outcome_col and new_out is not None and new_out.strip() != str(shot.get(outcome_col,"")).strip():
            corr_rows.append({"possession_id": poss_id, "field": outcome_col, "value": new_out.strip()})

        corr_df = pd.concat([corr_df, pd.DataFrame(corr_rows)], ignore_index=True)
        corr_df.to_csv(corr_path, index=False)

        st.success("Saved. Re-run keyframes later if you shift times broadly. For modeling, pass --corrections to normalize_and_export.py.")

    st.caption("Outputs: app/reports/corrections/timestamp_fixes.csv and app/reports/corrections/corrections.csv")
