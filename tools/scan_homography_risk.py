# tools/scan_homography_risk.py
# -*- coding: utf-8 -*-
"""
Scan clips vs. a per-game reference to flag which ones likely need manual review.
It uses only the extracted frames (tm, t0, tp) and ORB keypoint matching
to estimate a homography from each clip frame to the game's reference frame.

Outputs:
  app/reports/homography/scan_risk.csv  with columns:
    game_id, video_file, possession_id, inliers_t0, inliers_tm, inliers_tp,
    drift_tm_tp, risk_score, risk_reasons

Usage:
  python tools/scan_homography_risk.py
Options:
  --game_id 1           # scan a single game_id
  --min_inliers 20      # baseline good threshold (default 20)
  --ratio 0.75          # Lowe ratio for ORB matching (default 0.75)
  --drift_warn 0.08     # warn if tm↔tp transform delta > this fraction (default 0.08)
"""

from __future__ import annotations
from pathlib import Path
import argparse, json
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
FRAMES = APP / "reports" / "frames"
MANIFEST = FRAMES / "frames_manifest.csv"
HOMO_DIR = APP / "reports" / "homography"
OUT = HOMO_DIR / "scan_risk.csv"

def load_reference_image_for_game(game_id: str, shots: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Pick the first video_file in this game that already has a homography JSON; load its t0 frame."""
    files = list(dict.fromkeys(shots["video_file"].dropna().astype(str)))
    for vf in files:
        if (HOMO_DIR / f"{vf}.json").exists():
            # load its t0 frame
            man = pd.read_csv(MANIFEST, dtype=str)
            row = man[(man["video_file"].astype(str) == vf) & (man["context"] == "t0")]
            if not row.empty:
                p = str(row.iloc[0]["frame_path"])
                img = cv2.imread(p)
                if img is not None:
                    return img, vf
    return None, None

def orb_match_homography(imgA, imgB, ratio=0.75):
    """Find H: B -> A (maps B into A). Returns (H, inliers, good_matches)."""
    orb = cv2.ORB_create(3000)
    kpa, da = orb.detectAndCompute(imgA, None)
    kpb, db = orb.detectAndCompute(imgB, None)
    if da is None or db is None or len(kpa) < 8 or len(kpb) < 8:
        return None, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(db, da, k=2)  # query=B, train=A
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, 0, len(good)

    ptsB = np.float32([kpb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsA = np.float32([kpa[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers, len(good)

def homography_delta(H1: np.ndarray, H2: np.ndarray) -> float:
    """
    Rough scalar "drift" between two H (clip→ref). We normalize by H[2,2] and compute
    Frobenius norm of difference after scale alignment, reported as a fraction of H1's norm.
    """
    if H1 is None or H2 is None:
        return 1.0
    # normalize by bottom-right to reduce arbitrary scale
    h1 = H1 / (H1[2, 2] if abs(H1[2, 2]) > 1e-9 else 1.0)
    h2 = H2 / (H2[2, 2] if abs(H2[2, 2]) > 1e-9 else 1.0)
    num = np.linalg.norm(h1 - h2)
    den = np.linalg.norm(h1) + 1e-9
    return float(num / den)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game_id", help="Only scan a single game_id")
    ap.add_argument("--min_inliers", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--drift_warn", type=float, default=0.08)
    args = ap.parse_args()

    # Load events & manifest
    shots = pd.read_csv(APP / "shots.csv", dtype=str)
    shots["game_id"] = shots["game_id"].astype(str)
    # We're scanning per possession; use whatever is present (shot rows best, but turnovers ok too)
    man = pd.read_csv(MANIFEST, dtype=str)

    games = sorted(shots["game_id"].dropna().unique())
    if args.game_id:
        games = [str(args.game_id)]

    rows = []
    for gid in games:
        sub = shots[shots["game_id"] == gid].copy()
        if sub.empty:
            continue

        ref_img, ref_file = load_reference_image_for_game(gid, sub)
        if ref_img is None:
            # No reference homography in this game
            rows.append({"game_id": gid, "video_file": None, "possession_id": None,
                         "risk_score": 100, "risk_reasons": "no_reference_homography"})
            continue

        # Map convenience indices for tm/t0/tp
        M = man[man["video_file"].isin(sub["video_file"].astype(str))].copy()
        pivot = M.pivot_table(index=["game_id", "possession_id", "video_file"],
                              columns="context", values="frame_path", aggfunc="first").reset_index()

        for _, r in pivot.iterrows():
            vf = str(r["video_file"])
            pid = str(r["possession_id"])
            # load frames if exist
            def read(path):
                try:
                    return cv2.imread(str(path)) if pd.notna(path) else None
                except Exception:
                    return None

            img_t0 = read(r.get("t0"))
            img_tm = read(r.get("tm"))
            img_tp = read(r.get("tp"))

            H_t0, inl_t0, g_t0 = (None, 0, 0)
            H_tm, inl_tm, g_tm = (None, 0, 0)
            H_tp, inl_tp, g_tp = (None, 0, 0)

            if img_t0 is not None:
                H_t0, inl_t0, g_t0 = orb_match_homography(ref_img, img_t0, ratio=args.ratio)
            if img_tm is not None:
                H_tm, inl_tm, g_tm = orb_match_homography(ref_img, img_tm, ratio=args.ratio)
            if img_tp is not None:
                H_tp, inl_tp, g_tp = orb_match_homography(ref_img, img_tp, ratio=args.ratio)

            # drift between tm and tp solutions (if both exist)
            drift = homography_delta(H_tm, H_tp) if (H_tm is not None and H_tp is not None) else np.nan

            # Risk rules
            reasons = []
            score = 0
            # low inliers at t0 (main anchor)
            if inl_t0 < args.min_inliers:
                score += 60
                reasons.append(f"low_inliers_t0({inl_t0}<{args.min_inliers})")
            # optional: low inliers on tm/tp
            if (img_tm is not None and inl_tm < max(8, int(0.5*args.min_inliers))):
                score += 10
                reasons.append(f"low_inliers_tm({inl_tm})")
            if (img_tp is not None and inl_tp < max(8, int(0.5*args.min_inliers))):
                score += 10
                reasons.append(f"low_inliers_tp({inl_tp})")
            # drift between tm and tp
            if not np.isnan(drift) and drift > args.drift_warn:
                score += 30
                reasons.append(f"drift_tm_tp({drift:.3f}>{args.drift_warn})")

            # If everything looks fine, tiny score & “ok”
            if not reasons:
                reasons = ["ok"]
                score = 0

            rows.append({
                "game_id": gid,
                "video_file": vf,
                "possession_id": pid,
                "inliers_t0": inl_t0,
                "inliers_tm": inl_tm,
                "inliers_tp": inl_tp,
                "drift_tm_tp": drift,
                "risk_score": score,
                "risk_reasons": ";".join(reasons),
            })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    # quick console summary
    print(f"✓ Wrote risk scan → {OUT.resolve()}")
    if not df.empty:
        bad = df.sort_values("risk_score", ascending=False).head(15)
        print("\nTop 15 to review:")
        print(bad[["game_id","video_file","possession_id","risk_score","risk_reasons"]].to_string(index=False))

if __name__ == "__main__":
    main()
