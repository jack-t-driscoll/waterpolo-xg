# tools/propose_shot_xy.py
# -*- coding: utf-8 -*-
"""
Propose shooter_x/y from tm→t0 motion with confidence + triage.
- Requires: frames_manifest (tm/t0), homography JSON per video_file
- Outputs:
  1) app/reports/corrections/corrections.csv   (proposed shooter_x / shooter_y)
  2) app/reports/cv/review_queue.csv           (only items that need human review)
Flags to review:
- Low motion at t0 (timestamp might be off)
- Blob/mapping low confidence
- Big disagreement vs existing x/y (> meters_threshold)
"""

from __future__ import annotations
from pathlib import Path
import json, math
import cv2
import numpy as np
import pandas as pd

# -------- Paths --------
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SHOTS = APP / "shots.csv"
FRAMES_DIR = APP / "reports" / "frames"
MANIFEST = FRAMES_DIR / "frames_manifest.csv"
HOMO_DIR = APP / "reports" / "homography"
CORR_DIR = APP / "reports" / "corrections"
CV_DIR = APP / "reports" / "cv"
CORR_DIR.mkdir(parents=True, exist_ok=True)
CV_DIR.mkdir(parents=True, exist_ok=True)
CORR_CSV = CORR_DIR / "corrections.csv"
REVIEW_CSV = CV_DIR / "review_queue.csv"

# -------- Pool geometry --------
POOL_W = 20.0  # x: -10..+10 m
POOL_L = 15.0  # y: 0..15 m

def meters_to_norm(x_m: float, y_m: float) -> tuple[float,float]:
    x_n = (x_m + POOL_W/2) / POOL_W
    y_n = y_m / POOL_L
    return float(np.clip(x_n, 0.0, 1.0)), float(np.clip(y_n, 0.0, 1.0))

def norm_to_meters(x_n: float, y_n: float) -> tuple[float,float]:
    return (x_n * POOL_W - POOL_W/2, y_n * POOL_L)

# -------- IO helpers --------
def load_manifest() -> pd.DataFrame:
    man = pd.read_csv(MANIFEST, dtype=str)
    need = {"possession_id","video_file","context","frame_path"}
    miss = need - set(man.columns)
    if miss:
        raise SystemExit(f"frames_manifest.csv missing columns: {miss}")
    return man

def load_homography(video_file: str) -> np.ndarray | None:
    hp = HOMO_DIR / f"{video_file}.json"
    if not hp.exists(): return None
    try:
        d = json.loads(hp.read_text())
        H = np.array(d["H"], dtype=float)
        return H if H.shape==(3,3) else None
    except Exception:
        return None

def apply_homography(H: np.ndarray, px_pts: np.ndarray) -> np.ndarray:
    p = np.hstack([px_pts, np.ones((len(px_pts),1))])
    q = p @ H.T
    q = q[:, :2] / np.clip(q[:, 2:3], 1e-9, None)
    return q

# -------- Motion detector --------
def detect_motion(img_tm: str, img_t0: str):
    """Return (centroid_px, area, energy, mask) or (None,0,0,None)."""
    a = cv2.imread(img_tm); b = cv2.imread(img_t0)
    if a is None or b is None:
        return None, 0.0, 0.0, None
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ga = cv2.GaussianBlur(ga, (5,5), 0)
    gb = cv2.GaussianBlur(gb, (5,5), 0)
    d = cv2.absdiff(ga, gb)
    energy = float(np.mean(d))  # simple motion energy proxy 0..255
    d2 = cv2.GaussianBlur(d, (5,5), 0)
    _, th = cv2.threshold(d2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0, energy, th
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, area, energy, th
    cx = float(M["m10"]/M["m00"])
    cy = float(M["m01"]/M["m00"])
    return (cx, cy), area, energy, th

# -------- Confidence & triage rules --------
def confidence_from(area: float, energy: float, mapped_xy: tuple[float,float] | None) -> float:
    """
    0..1 confidence. Heuristic:
    - area scaled (cap at ~2000 px)
    - energy scaled (cap at ~25)
    - mapping sanity bonus if within pool bounds
    """
    a_term = min(area/2000.0, 1.0)
    e_term = min(energy/25.0, 1.0)
    m_term = 0.2
    if mapped_xy is not None:
        x_m, y_m = mapped_xy
        if -12 <= x_m <= 12 and -2 <= y_m <= 17:
            m_term = 0.4
        if -10 <= x_m <= 10 and  0 <= y_m <= 15:
            m_term = 0.6
    raw = 0.5*a_term + 0.4*e_term + m_term
    return float(np.clip(raw, 0.0, 1.0))

def meters_distance(a: tuple[float,float], b: tuple[float,float]) -> float:
    dx = a[0]-b[0]; dy = a[1]-b[1]
    return float(math.hypot(dx, dy))

# -------- Main --------
def main(meters_threshold: float = 2.0, min_confidence: float = 0.45, only_missing_xy: bool = True, limit: int | None = None):
    # Load shots
    shots = pd.read_csv(SHOTS, dtype=str)
    shots = shots[shots.get("event_type","").astype(str).str.lower()=="shot"].copy()

    # If only_missing_xy, keep rows where x or y is non-numeric/missing
    if only_missing_xy:
        def is_num(s):
            try: float(s); return True
            except: return False
        mask = ~shots["shooter_x"].apply(is_num) | ~shots["shooter_y"].apply(is_num)
        shots = shots[mask].copy()

    if shots.empty:
        print("No candidate shots (already filled x/y or no shots).")
        return

    # Manifest pivot
    man = load_manifest()
    piv = man.pivot_table(index="possession_id", columns="context", values="frame_path", aggfunc="first")

    proposals = []
    reviews = []
    processed = 0

    for _, r in shots.iterrows():
        if limit is not None and processed >= limit: break
        pid = str(r["possession_id"])
        vid = str(r.get("video_file",""))
        if not vid or pid not in piv.index:
            continue
        H = load_homography(vid)
        if H is None:
            continue

        row = piv.loc[pid]
        img_tm = row.get("tm"); img_t0 = row.get("t0")
        if pd.isna(img_tm) or pd.isna(img_t0):
            continue
        img_tm = str(img_tm); img_t0 = str(img_t0)
        if not (Path(img_tm).exists() and Path(img_t0).exists()):
            continue

        centroid, area, energy, _ = detect_motion(img_tm, img_t0)
        if centroid is None:
            reviews.append({"possession_id": pid, "reason": "no_motion_blob", "detail": f"energy={energy:.2f}, area=0"})
            continue

        # Map pixel → meters
        x_px, y_px = centroid
        xy_m = apply_homography(H, np.array([[x_px, y_px]], dtype=float))[0]
        x_m, y_m = float(xy_m[0]), float(xy_m[1])
        x_n, y_n = meters_to_norm(x_m, y_m)

        # Confidence
        conf = confidence_from(area, energy, (x_m, y_m))

        # Existing x/y disagreement in meters (if any)
        disagreement_m = None
        reasons = []
        try:
            if str(r.get("shooter_x","")).strip() != "" and str(r.get("shooter_y","")).strip() != "":
                ex_xn = float(r["shooter_x"]); ex_yn = float(r["shooter_y"])
                ex_m = norm_to_meters(ex_xn, ex_yn)
                disagreement_m = meters_distance(ex_m, (x_m, y_m))
                if disagreement_m is not None and disagreement_m > meters_threshold:
                    reasons.append(f"disagree>{meters_threshold}m ({disagreement_m:.2f}m)")
        except Exception:
            pass

        # Low-motion timestamp suspicion
        if energy < 2.0:
            reasons.append(f"low_motion_energy={energy:.2f}")

        # Low confidence
        if conf < min_confidence:
            reasons.append(f"low_conf={conf:.2f}")

        # Record proposal
        proposals.append({"possession_id": pid, "field": "shooter_x", "value": f"{x_n:.4f}"})
        proposals.append({"possession_id": pid, "field": "shooter_y", "value": f"{y_n:.4f}"})

        # Record review row only if any reason exists
        if reasons:
            reviews.append({
                "possession_id": pid,
                "video_file": vid,
                "proposed_x_m": round(x_m,3),
                "proposed_y_m": round(y_m,3),
                "confidence": round(conf,3),
                "motion_energy": round(energy,3),
                "blob_area": round(area,1),
                "reasons": "; ".join(reasons)
            })

        processed += 1

    if not proposals:
        print("No proposals generated (need homography + tm/t0 frames).")
        return

    # Append/merge corrections
    out_df = pd.DataFrame(proposals, columns=["possession_id","field","value"])
    if CORR_CSV.exists():
        prev = pd.read_csv(CORR_CSV, dtype=str)
        corr_all = pd.concat([prev, out_df], ignore_index=True)
    else:
        corr_all = out_df
    corr_all.to_csv(CORR_CSV, index=False)
    print(f"✓ appended {len(out_df)} field updates to {CORR_CSV}")

    # Write review queue
    rev_df = pd.DataFrame(reviews, columns=[
        "possession_id","video_file","proposed_x_m","proposed_y_m",
        "confidence","motion_energy","blob_area","reasons"
    ])
    rev_df.to_csv(REVIEW_CSV, index=False)
    print(f"✓ wrote triage → {REVIEW_CSV}  (rows={len(rev_df)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Propose shot x/y with confidence + triage")
    ap.add_argument("--meters_threshold", type=float, default=2.0, help="disagreement threshold vs existing x/y (meters)")
    ap.add_argument("--min_confidence", type=float, default=0.45, help="flag if confidence below this")
    ap.add_argument("--all", action="store_true", help="include shots that already have x/y")
    ap.add_argument("--limit", type=int, default=None, help="max shots to process")
    args = ap.parse_args()
    main(meters_threshold=args.meters_threshold,
         min_confidence=args.min_confidence,
         only_missing_xy=not args.all,
         limit=args.limit)
