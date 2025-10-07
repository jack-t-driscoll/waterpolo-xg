# tools/find_review_candidates.py
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

APP = Path(__file__).resolve().parents[1] / "app"
SHOTS = APP / "shots.csv"
FRAMES_MAN = APP / "reports" / "frames" / "frames_manifest.csv"
OUT = APP / "reports" / "review_candidates.csv"

NEEDED_SHOT_COLS = ["possession_id","game_id","event_type","video_file","video_timestamp_mmss",
                    "defender_count","goalie_present","shot_type","shot_result","man_state"]

def main():
    df = pd.read_csv(SHOTS, dtype=str)
    df["event_type"] = df.get("event_type","").astype(str).str.lower()
    shots = df[df["event_type"]=="shot"].copy()

    # Attach quick missingness flags
    for c in ["video_file","video_timestamp_mmss","defender_count","goalie_present","shot_type","shot_result","man_state"]:
        shots[f"miss_{c}"] = shots[c].isna() | (shots[c].astype(str).str.strip()=="")
    shots["needs_label_fix"] = shots[[f"miss_{c}" for c in ["defender_count","goalie_present","shot_type","shot_result","man_state"]]].any(axis=1)
    shots["needs_time_fix"]  = shots[[f"miss_{c}" for c in ["video_file","video_timestamp_mmss"]]].any(axis=1)

    # Join frame status (has_tm/t0/tp)
    man = pd.read_csv(FRAMES_MAN, dtype=str) if FRAMES_MAN.exists() else pd.DataFrame(columns=["possession_id","context"])
    has = man.pivot_table(index="possession_id", columns="context", values="frame_path", aggfunc="first")
    for k in ["tm","t0","tp"]:
        shots[f"has_{k}"] = shots["possession_id"].map(lambda pid: str(pid) in has.index and pd.notna(has.loc[str(pid)].get(k)))

    # Priority rules
    shots["priority"] = 0
    shots.loc[shots["needs_time_fix"], "priority"] = 3
    shots.loc[~shots["has_t0"], "priority"] = 3
    shots.loc[(~shots["has_tm"]) | (~shots["has_tp"]), "priority"] = shots[["priority"]].clip(lower=2)
    shots.loc[shots["needs_label_fix"] & (shots["priority"]<2), "priority"] = 2

    # Keep a compact table
    keep = ["possession_id","game_id","video_file","video_timestamp_mmss","priority",
            "needs_time_fix","needs_label_fix","has_tm","has_t0","has_tp",
            "defender_count","goalie_present","shot_type","shot_result","man_state"]
    for c in keep:
        if c not in shots.columns: shots[c] = pd.NA

    # Sort by priority desc, then game_id
    out = shots[keep].sort_values(["priority","game_id","possession_id"], ascending=[False, True, True])
    out.to_csv(OUT, index=False)
    print(f"✓ wrote {len(out)} rows → {OUT}")

if __name__ == "__main__":
    main()
