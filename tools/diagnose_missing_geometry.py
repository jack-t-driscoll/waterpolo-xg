# tools/diagnose_missing_geometry.py
from pathlib import Path
import pandas as pd
import json
import os

APP = Path("app")
FEATURES = APP / "features_shots.csv"
SHOTS = APP / "shots.csv"
FRAMES_MAN = APP / "reports" / "frames" / "frames_manifest.csv"
HOMO_DIR = APP / "reports" / "homography"
CORR_XY = APP / "reports" / "corrections" / "corrections_shot_xy.csv"

def load_csv(path):
    return pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(dtype=str)

def has_homography(video_file: str) -> bool:
    if not video_file or pd.isna(video_file):
        return False
    # homography JSONs are named by source_video_id / video_file
    p = HOMO_DIR / f"{video_file}.json"
    return p.exists()

def frames_status(pid: str, frames_df: pd.DataFrame):
    # Return whether tm/t0/tp exist for this possession
    if frames_df.empty: return (None, None, None)
    row = frames_df[frames_df["possession_id"] == pid]
    if row.empty: return (False, False, False)
    have = {c: False for c in ["tm","t0","tp"]}
    for _, r in row.iterrows():
        ctx = r.get("context")
        if ctx in have and isinstance(r.get("frame_path"), str) and len(r.get("frame_path")) > 0:
            have[ctx] = True
    return have["tm"], have["t0"], have["tp"]

def main():
    f = load_csv(FEATURES)
    s = load_csv(SHOTS)
    fm = load_csv(FRAMES_MAN)
    if not f.shape[0]:
        raise SystemExit(f"Missing or empty {FEATURES}")

    # Only rows where geometry is missing
    f_num = f.copy()
    for col in ["distance_m","angle_deg","shooter_x","shooter_y","goalie_distance_m"]:
        if col in f_num.columns:
            f_num[col] = pd.to_numeric(f_num[col], errors="coerce")

    miss = f_num[f_num["distance_m"].isna() | f_num["angle_deg"].isna()].copy()

    # Join back to shots to get event_type etc.
    cols_from_shots = ["possession_id","event_type","video_file","video_timestamp_mmss","game_id","period","player_number",
                       "shooter_handedness","shot_type","attack_type"]
    s_small = s[cols_from_shots] if set(cols_from_shots).issubset(s.columns) else s
    miss = miss.merge(s_small, on="possession_id", how="left", suffixes=("","_shots"))

    # Reason flags
    reasons = []
    for idx, r in miss.iterrows():
        pid = r["possession_id"]
        et  = (r.get("event_type") or "").lower()
        vf  = r.get("video_file")
        # frames
        has_tm, has_t0, has_tp = frames_status(pid, fm)
        # homography
        has_H = has_homography(vf)
        # xy tag present?
        sx = pd.to_numeric(r.get("shooter_x"), errors="coerce")
        sy = pd.to_numeric(r.get("shooter_y"), errors="coerce")
        has_xy = pd.notna(sx) and pd.notna(sy)

        why = []
        if et != "shot":
            why.append("not_a_shot_in_shots.csv")
        if not has_xy:
            why.append("missing_xy_tag")
        if not has_t0:
            why.append("missing_t0_frame")
        if not has_H:
            why.append("missing_homography_json")
        if len(why) == 0:
            why.append("unknown_check_feature_formula/angles")
        reasons.append("|".join(why))
    miss["reason"] = reasons

    # Summaries
    print("\n=== Missing geometry summary ===")
    print(f"Total rows with missing distance OR angle: {len(miss)}")
    print("\nBy reason:")
    print(miss["reason"].value_counts().to_string())

    print("\nBy event_type (from shots.csv):")
    print(miss["event_type"].fillna("Unknown").value_counts().to_string())

    # Helpful short table
    show_cols = ["possession_id","event_type","video_file","video_timestamp_mmss",
                 "shooter_x","shooter_y","distance_m","angle_deg","reason"]
    print("\nExamples (first 30):")
    print(miss[show_cols].head(30).to_string(index=False))

    # Export full list
    out = APP / "reports" / "diagnostics" / "missing_geometry.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    miss.to_csv(out, index=False)
    print(f"\n✓ Wrote detailed list → {out.resolve()}")

if __name__ == "__main__":
    main()
