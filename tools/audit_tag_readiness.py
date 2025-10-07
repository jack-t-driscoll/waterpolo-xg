# tools/audit_tag_readiness.py
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

APP = Path(__file__).resolve().parents[1] / "app"
SHOTS = APP / "shots.csv"
MAN = APP / "reports" / "frames" / "frames_manifest.csv"

def main():
    if not SHOTS.exists():
        print(f"ERROR: missing {SHOTS}")
        return
    df = pd.read_csv(SHOTS, dtype=str)
    evt = df.get("event_type","").astype(str).str.lower()
    shots = df[evt=="shot"].copy()
    print(f"Shots total: {len(shots)}")

    # Simple missingness on timestamps/labels
    def miss(s): return s.isna() | (s.astype(str).str.strip()=="")
    cols_time = ["video_file","video_timestamp_mmss"]
    cols_lbl  = ["defender_count","goalie_present","shot_type","shot_result","man_state"]
    for c in cols_time + cols_lbl:
        if c not in shots.columns:
            print(f"WARNING: column missing in shots.csv → {c}")
            shots[c] = pd.NA

    m_time = shots[cols_time].apply(miss).any(axis=1)
    m_lbl  = shots[cols_lbl].apply(miss).any(axis=1)
    print(f"Missing time fields (any of {cols_time}): {int(m_time.sum())}")
    print(f"Missing label fields (any of {cols_lbl}): {int(m_lbl.sum())}")

    # Frames manifest audit
    if not MAN.exists():
        print(f"Frames manifest missing: {MAN} (run extract_keyframes.py)")
        return
    man = pd.read_csv(MAN, dtype=str)
    need_cols = {"possession_id","context","frame_path"}
    if not need_cols.issubset(set(man.columns)):
        print(f"ERROR: frames_manifest.csv must have columns {need_cols}. Found: {list(man.columns)}")
        return

    # contexts present
    print("Contexts present in manifest:", sorted(man["context"].dropna().unique().tolist()))
    has = man.pivot_table(index="possession_id", columns="context", values="frame_path", aggfunc="first")
    # expected contexts
    for k in ["tm","t0","tp"]:
        if k not in has.columns:
            has[k] = pd.NA

    # Join per-shot
    join = shots[["possession_id","game_id","video_file","video_timestamp_mmss"]].copy()
    join["has_tm"] = join["possession_id"].map(lambda pid: pd.notna(has.loc[pid]["tm"]) if pid in has.index else False)
    join["has_t0"] = join["possession_id"].map(lambda pid: pd.notna(has.loc[pid]["t0"]) if pid in has.index else False)
    join["has_tp"] = join["possession_id"].map(lambda pid: pd.notna(has.loc[pid]["tp"]) if pid in has.index else False)

    no_tm = (~join["has_tm"]).sum()
    no_t0 = (~join["has_t0"]).sum()
    no_tp = (~join["has_tp"]).sum()
    print(f"Missing frames: tm={no_tm}, t0={no_t0}, tp={no_tp}")

    # If suspicious zeros, show likely reasons
    if len(shots) > 0 and (no_tm+no_t0+no_tp) == 0 and int(m_time.sum())==0 and int(m_lbl.sum())==0:
        print("All good: no triage needed. (Zero missing frames/timestamps/labels on shots.)")
    else:
        # Show a few examples that need attention
        print("\nExamples needing time fixes:")
        print(join[m_time].head(10).to_string(index=False))
        print("\nExamples missing any frames:")
        needs_frames = ~(join["has_tm"] & join["has_t0"] & join["has_tp"])
        print(join[needs_frames].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
