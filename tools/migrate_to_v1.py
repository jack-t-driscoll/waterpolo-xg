# -*- coding: utf-8 -*-
"""
migrate_to_v1.py
Convert known shots CSV variants into the canonical Schema v1.

Usage:
    python tools/migrate_to_v1.py input.csv output.csv

Schema v1 columns:
match_id,video_file,t_start,t_end,team,shooter,outcome,
x,y,goal_x,goal_y,angle_deg_signed,distance_m,
shot_type,pressure,man_up,goalie_x,goalie_y,quarter,clock,notes,schema_version
"""

import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

SCHEMA_V1 = [
    "match_id","video_file","t_start","t_end","team","shooter","outcome",
    "x","y","goal_x","goal_y","angle_deg_signed","distance_m",
    "shot_type","pressure","man_up","goalie_x","goalie_y","quarter","clock","notes","schema_version"
]

GOAL_DEFAULT = (0.0, 0.0)

CANDIDATES = {
    "team": ["team","our_team_name","team_name","our_team"],
    "shooter": ["shooter","player_number","shooter_number","drawn_by_player_number"],
    "outcome": ["outcome","shot_result","shot_result_raw"],
    "x": ["x","shooter_x"],
    "y": ["y","shooter_y"],
    "goal_x": ["goal_x"],
    "goal_y": ["goal_y"],
    "man_up": ["man_up","man_state"],
    "video_file": ["video_file","video","file","source"],
    "match_id": ["match_id","game_id","match","game"],
    "quarter": ["quarter","period"],
    "clock": ["clock","time_remaining","game_clock"],
    "shot_type": ["shot_type","event_type"],
    "pressure": ["pressure","defender_count"],
    "t_start": ["t_start","timestamp_start_s","video_timestamp_mmss","video_timestamp_s"],
    "t_end": ["t_end","timestamp_end_s"],
    "goalie_x": ["goalie_x"], "goalie_y": ["goalie_y"],
    "notes": ["notes","note","comment"],
}

def first_present(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def col(df, names, default=np.nan, dtype=None):
    name = first_present(df, names)
    s = df[name] if name else pd.Series(default, index=df.index)
    if dtype == "num":
        return pd.to_numeric(s, errors="coerce")
    if dtype == "str":
        return s.astype(str)
    return s

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

def main(inp, outp):
    p = Path(inp)
    if not p.exists():
        raise SystemExit(f"Input not found: {p}")

    # Encoding and delimiter: let pandas sniff delimiter; try utf-8 then utf-8-sig
    for enc in ("utf-8","utf-8-sig"):
        try:
            df = pd.read_csv(p, encoding=enc, sep=None, engine="python")
            break
        except Exception:
            df = None
    if df is None:
        raise SystemExit("Failed to read CSV in utf-8/utf-8-sig.")

    out = pd.DataFrame(index=df.index)

    out["match_id"]  = col(df, CANDIDATES["match_id"], "", "str")
    out["video_file"]= col(df, CANDIDATES["video_file"], "", "str")
    out["t_start"]   = col(df, CANDIDATES["t_start"], np.nan, "num")
    out["t_end"]     = col(df, CANDIDATES["t_end"], np.nan, "num")
    out["team"]      = col(df, CANDIDATES["team"], "", "str")
    out["shooter"]   = col(df, CANDIDATES["shooter"], "", "str")
    out["outcome"]   = normalize_outcome(col(df, CANDIDATES["outcome"], "", None))

    x = col(df, CANDIDATES["x"], np.nan, "num")
    y = col(df, CANDIDATES["y"], np.nan, "num")
    gx = col(df, CANDIDATES["goal_x"], GOAL_DEFAULT[0], "num").fillna(GOAL_DEFAULT[0])
    gy = col(df, CANDIDATES["goal_y"], GOAL_DEFAULT[1], "num").fillna(GOAL_DEFAULT[1])
    out["x"] = x; out["y"] = y; out["goal_x"] = gx; out["goal_y"] = gy

    # Angle/distance (compute if absent)
    ang = np.degrees(np.arctan2(gy - y, gx - x))
    out["angle_deg_signed"] = pd.to_numeric(df.get("angle_deg_signed", ang), errors="coerce")
    out["angle_deg_signed"] = out["angle_deg_signed"].clip(-90, 90)

    dist = np.sqrt((x - gx)**2 + (y - gy)**2)
    out["distance_m"] = pd.to_numeric(df.get("distance_m", dist), errors="coerce")

    out["shot_type"] = col(df, CANDIDATES["shot_type"], "", "str")
    out["pressure"]  = col(df, CANDIDATES["pressure"], "", "str")
    out["man_up"]    = normalize_man_up(col(df, CANDIDATES["man_up"], 0, None))
    out["goalie_x"]  = col(df, CANDIDATES["goalie_x"], np.nan, "num")
    out["goalie_y"]  = col(df, CANDIDATES["goalie_y"], np.nan, "num")
    out["quarter"]   = col(df, CANDIDATES["quarter"], "", "str")
    out["clock"]     = col(df, CANDIDATES["clock"], "", "str")
    out["notes"]     = col(df, CANDIDATES["notes"], "", "str")

    out["schema_version"] = 1

    # Reorder columns strictly to Schema v1
    out = out.reindex(columns=SCHEMA_V1)

    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False, encoding="utf-8")
    print(f"Wrote Schema v1 CSV: {outp}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
