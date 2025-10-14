#!/usr/bin/env python3
"""
make_tag_queue.py
Build a prioritized tagging queue of possessions that most need shooter click.

Criteria (a possession is included if ALL apply):
  1) Has a t0 frame in frames_manifest.csv
  2) Has a homography JSON for its video_file
  3) EITHER:
     - distance_m or angle_deg is missing/NaN in features_shots.csv
     - OR no shooter_x/y override exists in corrections.csv
     - OR homography propagation audit shows low inliers for that video (likely drift)

Outputs:
  app/reports/tag_queue.csv   (possession_id list)
Also prints a per-game shortlist (top N per game).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

APP = Path("app")
FRAMES_MAN = APP / "reports" / "frames" / "frames_manifest.csv"
FEATURES = APP / "features_shots.csv"
CORR = APP / "reports" / "corrections" / "corrections.csv"
HOMO_DIR = APP / "reports" / "homography"
AUDIT = APP / "reports" / "homography" / "propagation_audit_multi.csv"
OUT = APP / "reports" / "tag_queue.csv"

def load_df(path: Path, cols=None) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame(columns=cols or [])
    return pd.read_csv(path, dtype=str)

def have_homography(video_file: str) -> bool:
    j = HOMO_DIR / (Path(video_file).stem + ".json")
    return j.exists()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_manifest", default=str(FRAMES_MAN))
    ap.add_argument("--features", default=str(FEATURES))
    ap.add_argument("--corrections", default=str(CORR))
    ap.add_argument("--audit", default=str(AUDIT))
    ap.add_argument("--out_csv", default=str(OUT))
    ap.add_argument("--per_game", type=int, default=8, help="Top N per game to surface")
    ap.add_argument("--low_inlier_thresh", type=int, default=250, help="Flag videos below this inliers")
    args = ap.parse_args()

    frames = load_df(Path(args.frames_manifest))
    feats  = load_df(Path(args.features))
    corr   = load_df(Path(args.corrections))
    audit  = load_df(Path(args.audit))

    if frames.empty or "context" not in frames.columns:
        raise SystemExit("frames_manifest.csv missing or malformed.")
    frames = frames[frames["context"] == "t0"].copy()

    # Minimal columns expectation
    for col in ["possession_id","video_file"]:
        if col not in frames.columns: raise SystemExit(f"manifest missing '{col}'")
    for col in ["possession_id","game_id","video_file","distance_m","angle_deg"]:
        if col not in feats.columns: raise SystemExit(f"features_shots.csv missing '{col}'")

    # Corrections present?
    have_corr = pd.Series(False, index=feats.index)
    if not corr.empty and "possession_id" in corr.columns:
        have_corr = feats["possession_id"].isin(corr["possession_id"])

    # NaN geometry flags
    d_is_nan = pd.to_numeric(feats["distance_m"], errors="coerce").isna()
    a_is_nan = pd.to_numeric(feats["angle_deg"], errors="coerce").isna()

    # Join frames (ensure t0 exists) and filter to rows that have homography JSON
    base = feats.merge(frames[["possession_id","video_file"]], on=["possession_id","video_file"], how="inner")
    base["has_homo"] = base["video_file"].map(lambda v: "1" if have_homography(v) else "0")
    base = base[base["has_homo"] == "1"].copy()

    # Low-inlier video penalty (if audit present)
    # audit rows have: game_id, video_file, status, inliers, good, anchor_used
    low_inlier_videos = set()
    if not audit.empty and {"video_file","inliers"}.issubset(audit.columns):
        aud = audit.copy()
        aud["inliers"] = pd.to_numeric(aud["inliers"], errors="coerce")
        aud = aud.groupby("video_file", as_index=False)["inliers"].median()
        low_inlier_videos = set(aud.loc[aud["inliers"] < args.low_inlier_thresh, "video_file"].astype(str).tolist())

    base["geom_nan"] = (feats.loc[base.index, "possession_id"].map(dict(zip(feats["possession_id"], d_is_nan | a_is_nan))).astype(bool))
    base["has_corr"] = feats.loc[base.index, "possession_id"].map(dict(zip(feats["possession_id"], have_corr))).fillna(False)

    # Need tag if geometry is NaN OR no correction yet OR video was low-inlier in audit
    base["low_inlier_video"] = base["video_file"].isin(low_inlier_videos)
    need = base[(base["geom_nan"]) | (~base["has_corr"]) | (base["low_inlier_video"])].copy()

    # Priority score:
    #   +2 if geom_nan
    #   +1 if no correction
    #   +1 if low-inlier video
    need["priority"] = (
        need["geom_nan"].astype(int) * 2
        + (~need["has_corr"]).astype(int)
        + need["low_inlier_video"].astype(int)
    )

    # Per-game shortlist
    need["priority"] = need["priority"].astype(int)
    need["game_id"] = need["game_id"].astype(str)
    need = need.sort_values(["game_id","priority"], ascending=[True, False])
    short = need.groupby("game_id").head(args.per_game).copy()

    # Write queue (possession_id only)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    short[["possession_id"]].drop_duplicates().to_csv(out_csv, index=False)

    # Console preview with reasons
    cols = ["possession_id","game_id","video_file","priority","geom_nan","has_corr","low_inlier_video"]
    print("\nTag queue (top N per game):")
    print(short[cols].to_string(index=False))
    print(f"\n✓ Wrote queue → {out_csv.resolve()}")
    if need.empty:
        print("No tagging needed by these criteria. You’re good!")

if __name__ == "__main__":
    main()
