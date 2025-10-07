# tools/propagate_homography.py
# -*- coding: utf-8 -*-
"""
Propagate homographies across clips by matching keyframes (tm/t0/tp)
using the frames manifest. For each game:
  - Choose a reference clip (prefer one with calibrated JSON)
  - Match ref frame -> target frame (t0 fallback to tm, then tp)
  - Save H as app/reports/homography/H_<video_file>.json
  - Write audit CSV with inliers/good matches and status

Inputs:
  app/reports/frames/frames_manifest.csv    (from extract_keyframes.py)
  app/reports/homography/H_<video_file>.json (optional, used as 'reference' if present)

Outputs:
  app/reports/homography/H_<video_file>.json
  app/reports/homography/propagation_audit.csv
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
FRAMES = APP / "reports" / "frames"
HOMO_DIR = APP / "reports" / "homography"
MANIFEST = FRAMES / "frames_manifest.csv"
AUDIT = HOMO_DIR / "propagation_audit.csv"

# ORB/FLANN params
ORB_N = 3000
FLANN_INDEX_LSH = 6
FLANN_PARAMS = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=12, key_size=20, multi_probe_level=2)

def read_img(p: Path) -> Optional[np.ndarray]:
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img

def orb_kp_desc(img: np.ndarray):
    orb = cv2.ORB_create(nfeatures=ORB_N)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def good_matches(desA, desB, k=2, ratio=0.75):
    if desA is None or desB is None:
        return []
    matcher = cv2.FlannBasedMatcher(FLANN_PARAMS, {})
    raw = matcher.knnMatch(desA, desB, k)
    goods = []
    for pair in raw:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            goods.append(pair[0])
    return goods

def estimate_h(kpA, kpB, matches, ransac_th=3.0):
    if len(matches) < 8:
        return None, 0
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransac_th)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers

def load_H_json(video_file: str) -> Optional[np.ndarray]:
    p = HOMO_DIR / f"H_{video_file}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        H = np.array(data["H"], dtype=np.float64)
        return H
    except Exception:
        return None

def save_H_json(video_file: str, H: np.ndarray):
    HOMO_DIR.mkdir(parents=True, exist_ok=True)
    data = {"H": H.tolist()}
    (HOMO_DIR / f"H_{video_file}.json").write_text(json.dumps(data, indent=2))

def pick_reference_for_game(gdf: pd.DataFrame) -> Tuple[str, Path]:
    """
    Prefer a clip with existing calibrated JSON; else the first clip with a t0 frame.
    Returns (video_file, frame_path_t0).
    """
    # try any with H json
    for v in gdf["video_file"].unique():
        if (HOMO_DIR / f"H_{v}.json").exists():
            # find its t0
            cand = gdf[(gdf["video_file"] == v) & (gdf["context"] == "t0")]
            if not cand.empty and Path(cand.iloc[0]["frame_path"]).exists():
                return v, Path(cand.iloc[0]["frame_path"])
    # fallback: first available t0
    cand = gdf[gdf["context"] == "t0"]
    if not cand.empty:
        return cand.iloc[0]["video_file"], Path(cand.iloc[0]["frame_path"])
    # last resort: any context
    r = gdf.iloc[0]
    return r["video_file"], Path(r["frame_path"])

def frame_for_video(gdf: pd.DataFrame, video_file: str, contexts=("t0","tm","tp")) -> Optional[Path]:
    for ctx in contexts:
        sub = gdf[(gdf["video_file"] == video_file) & (gdf["context"] == ctx)]
        if not sub.empty:
            p = Path(sub.iloc[0]["frame_path"])
            if p.exists():
                return p
    return None

def main():
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")
    df = pd.read_csv(MANIFEST, dtype=str)
    if df.empty:
        print("Manifest empty; nothing to propagate.")
        return

    # Ensure homography dir exists
    HOMO_DIR.mkdir(parents=True, exist_ok=True)

    # Group by game
    # Expect columns: game_id, possession_id, video_file, context, t_s, frame_path
    out_rows = []
    for gid, gdf in df.groupby("game_id"):
        gdf = gdf.copy()
        # Choose a reference
        ref_vid, ref_path = pick_reference_for_game(gdf)
        ref_img = read_img(ref_path)
        if ref_img is None:
            out_rows.append({"game_id": gid, "video_file": ref_vid, "status": "ref_image_missing", "inliers": 0, "good": 0})
            print(f"[game {gid}] Reference missing frame: {ref_path}")
            continue

        # Precompute features for reference
        ref_kp, ref_des = orb_kp_desc(ref_img)

        # If we already have H for ref, load it (used downstream if needed).
        ref_H = load_H_json(ref_vid)

        print(f"[game {gid}] Reference: {ref_vid} (ref frame: {ref_path.name})")

        # Propagate to all clips in this game
        for v in sorted(gdf["video_file"].unique()):
            tgt_path = frame_for_video(gdf, v, contexts=("t0","tm","tp"))
            if tgt_path is None:
                out_rows.append({"game_id": gid, "video_file": v, "status": "missing_any_frame", "inliers": 0, "good": 0})
                print(f"  - {v}: missing tm/t0/tp frame on disk")
                continue

            # Skip if already has H (keep existing)
            if (HOMO_DIR / f"H_{v}.json").exists():
                out_rows.append({"game_id": gid, "video_file": v, "status": "exists", "inliers": "", "good": ""})
                continue

            tgt_img = read_img(tgt_path)
            if tgt_img is None:
                out_rows.append({"game_id": gid, "video_file": v, "status": "read_fail", "inliers": 0, "good": 0})
                print(f"  - {v}: failed to read {tgt_path}")
                continue

            # Match
            kpB, desB = orb_kp_desc(tgt_img)
            goods = good_matches(ref_des, desB, k=2, ratio=0.75)
            H, inliers = estimate_h(ref_kp, kpB, goods)
            status = ""
            if H is None or inliers < 20:
                status = f"match FAIL (inliers={inliers}, good={len(goods)})"
                out_rows.append({"game_id": gid, "video_file": v, "status": status, "inliers": inliers, "good": len(goods)})
                print(f"  - {v}: {status}")
                continue

            # Save
            save_H_json(v, H)
            out_rows.append({"game_id": gid, "video_file": v, "status": "propagated", "inliers": inliers, "good": len(goods)})
            print(f"  ✓ {v}: propagated (inliers={inliers}, good={len(goods)})")

    # Write audit
    audit = pd.DataFrame(out_rows)
    audit.to_csv(AUDIT, index=False)
    print(f"\n✓ Wrote audit → {AUDIT.resolve()}")

if __name__ == "__main__":
    main()
