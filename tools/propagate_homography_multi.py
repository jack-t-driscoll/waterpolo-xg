# tools/propagate_homography_multi.py
# -*- coding: utf-8 -*-
"""
Propagate homography JSONs game-by-game, choosing the best anchor per target.
- Reads t0 frames from app/reports/frames/frames_manifest.csv
- Reads video mapping (video_file -> game_id) from app/videos.csv
- Uses ORB+FLANN to match target t0 frame to each available anchor t0 frame in the same game
- Computes T(target->anchor) and composes: H_target = H_anchor @ T(target->anchor)
- Writes target homography JSONs into app/reports/homography/<video_file>.json

Key options:
  --only_game N       process only a single game_id
  --overwrite         allow overwriting existing JSONs
  --min_inliers 40    minimum RANSAC inliers to accept a match
  --limit_kp 3000     ORB keypoint cap
"""

from __future__ import annotations
import json
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd


APP_DIR = Path("app")
FRAMES_DIR = APP_DIR / "reports" / "frames"
HOMO_DIR = APP_DIR / "reports" / "homography"
MANIFEST = FRAMES_DIR / "frames_manifest.csv"     # possession_id, context (tm/t0/tp), frame_path
VIDEOS_CSV = Path("data") / "videos.csv"          # video_file, full_path, fps, notes (schema flexible)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    man = pd.read_csv(manifest_path, dtype=str)
    # Expect columns: possession_id, context, frame_path, video_file, game_id (if present)
    # If video_file/game_id not present, derive from possession_id via shots file if needed (not used here).
    piv = man.pivot_table(
        index="possession_id",
        columns="context",
        values="frame_path",
        aggfunc="first"
    )
    piv = piv.reset_index()
    # Bring through video_file and game_id if present
    if "video_file" in man.columns:
        vf_map = man.dropna(subset=["video_file"]).drop_duplicates("possession_id").set_index("possession_id")["video_file"]
        piv["video_file"] = piv["possession_id"].map(vf_map)
    if "game_id" in man.columns:
        gid_map = man.dropna(subset=["game_id"]).drop_duplicates("possession_id").set_index("possession_id")["game_id"]
        piv["game_id"] = piv["possession_id"].map(gid_map)
    return piv


def load_videos(videos_csv: Path) -> pd.DataFrame:
    """
    Load videos.csv and normalize columns so downstream code can always use:
      - video_file
      - full_path

    Supports both legacy and new schemas:
      - legacy: source_video_id, source_video_path
      - new:    video_file, full_path
    """
    df = pd.read_csv(videos_csv, dtype=str)

    # Strip whitespace
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    # Map columns if needed
    col_map = {}
    if "video_file" in df.columns and "full_path" in df.columns:
        # already in new schema
        pass
    elif "source_video_id" in df.columns and "source_video_path" in df.columns:
        col_map = {
            "source_video_id": "video_file",
            "source_video_path": "full_path",
        }
        df = df.rename(columns=col_map)
    else:
        raise SystemExit(
            "videos.csv must include either "
            "[video_file, full_path] or [source_video_id, source_video_path]. "
            f"Found columns: {list(df.columns)}"
        )

    # Final sanity
    req = ["video_file", "full_path"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"videos.csv missing required columns after normalization: {missing}")

    # Normalize paths (leave as-is if they’re absolute Windows paths like D:\...)
    df["full_path"] = df["full_path"].str.replace("\\\\", "\\")
    return df


def list_anchors_for_game(game_id: str, manifest: pd.DataFrame) -> Dict[str, Dict]:
    """
    Anchors = videos with an existing homography JSON AND a t0 frame on disk.
    Returns mapping: video_file -> {"json": Path, "t0": Path}
    """
    anchors: Dict[str, Dict] = {}
    for _, row in manifest.iterrows():
        vf = row.get("video_file")
        gid = str(row.get("game_id"))
        t0 = row.get("t0")
        if not vf or gid != game_id or not t0 or not Path(t0).exists():
            continue
        jpath = HOMO_DIR / f"{vf}.json"
        if jpath.exists():
            anchors[vf] = {"json": jpath, "t0": Path(t0)}
    return anchors


def list_targets_for_game(game_id: str, manifest: pd.DataFrame) -> Dict[str, Path]:
    """
    Targets = videos in the game with a t0 frame on disk (we'll create/overwrite JSON for them).
    Returns mapping: video_file -> t0_frame_path
    """
    targets: Dict[str, Path] = {}
    for _, row in manifest.iterrows():
        vf = row.get("video_file")
        gid = str(row.get("game_id"))
        t0 = row.get("t0")
        if not vf or gid != game_id or not t0 or not Path(t0).exists():
            continue
        targets[vf] = Path(t0)
    return targets


def read_homography_json(p: Path) -> Optional[np.ndarray]:
    try:
        data = json.loads(p.read_text())
        H = np.array(data.get("H"), dtype=float).reshape(3, 3)
        return H
    except Exception:
        return None


def write_homography_json(p: Path, H: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"H": H.reshape(3, 3).tolist()}
    p.write_text(json.dumps(payload, indent=2))


def detect_and_match(imgA: np.ndarray, imgB: np.ndarray, limit_kp: int = 3000) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Match A->B (A=target, B=anchor). Returns (T_A2B, inliers, good_matches).
    """
    orb = cv2.ORB_create(nfeatures=int(limit_kp))
    kpA, desA = orb.detectAndCompute(imgA, None)
    kpB, desB = orb.detectAndCompute(imgB, None)
    if desA is None or desB is None or len(kpA) < 8 or len(kpB) < 8:
        return None, 0, 0

    index_params = dict(algorithm=6, table_number=6, key_size=10, multi_probe_level=1)  # FLANN LSH
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desA, desB, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return None, 0, len(good)

    src_pts = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    T, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return T, inliers, len(good)


def load_gray(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def main():
    ap = argparse.ArgumentParser(description="Propagate homography JSONs by matching t0 frames to per-game anchors.")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--homography_dir", default=str(HOMO_DIR))
    ap.add_argument("--frames_dir", default=str(FRAMES_DIR))
    ap.add_argument("--videos_csv", default=str(VIDEOS_CSV))
    ap.add_argument("--only_game", type=str, default=None, help="Process only this game_id")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONs")
    ap.add_argument("--min_inliers", type=int, default=40)
    ap.add_argument("--limit_kp", type=int, default=3000)
    args = ap.parse_args()

    homo_dir = Path(args.homography_dir)
    frames_dir = Path(args.frames_dir)

    # Load manifest (for t0 paths + video_file + game_id)
    man = load_manifest(Path(args.manifest))

    # If manifest lacks game_id, try to join from videos.csv (by video_file)
    if "game_id" not in man.columns or man["game_id"].isna().any():
        vids = load_videos(Path(args.videos_csv))
        if "game_id" in vids.columns:
            v2g = vids.set_index("video_file")["game_id"].to_dict()
            if "game_id" not in man.columns:
                man["game_id"] = man["video_file"].map(v2g)
            else:
                miss = man["game_id"].isna()
                man.loc[miss, "game_id"] = man.loc[miss, "video_file"].map(v2g)

    # Sanity
    if "video_file" not in man.columns or "game_id" not in man.columns or "t0" not in man.columns:
        raise SystemExit("frames_manifest must include video_file, game_id, and a t0 frame for matching.")

    games = sorted(man["game_id"].dropna().astype(str).unique())
    if args.only_game is not None:
        games = [g for g in games if str(g) == str(args.only_game)]

    audit_rows = []

    for gid in games:
        anchors = list_anchors_for_game(gid, man)
        targets = list_targets_for_game(gid, man)

        # If we have no anchors for this game, skip
        if not anchors:
            print(f"[game {gid}] Anchors: 0; targets: {len(targets)} (no anchors; skipping)")
            continue

        print(f"[game {gid}] Anchors: {len(anchors)}; targets: {len(targets)}")

        # Preload anchor images + homographies
        anchor_imgs: Dict[str, np.ndarray] = {}
        anchor_H: Dict[str, np.ndarray] = {}
        for vf, meta in anchors.items():
            img = load_gray(meta["t0"])
            H = read_homography_json(meta["json"])
            if img is None or H is None:
                continue
            anchor_imgs[vf] = img
            anchor_H[vf] = H

        for vf_target, t0_target in targets.items():
            out_json = homo_dir / f"{vf_target}.json"
            if out_json.exists() and not args.overwrite:
                # Already present; keep it
                audit_rows.append((gid, vf_target, "skip_existing", 0, 0))
                continue

            imgA = load_gray(t0_target)  # target image
            if imgA is None:
                audit_rows.append((gid, vf_target, "missing_t0", 0, 0))
                print(f"  - {vf_target}: missing t0 frame")
                continue

            best = {"inliers": -1, "good": 0, "vf_anchor": None, "H_target": None}

            # Try each anchor in this game
            for vf_anchor, imgB in anchor_imgs.items():
                T, inliers, good = detect_and_match(imgA, imgB, limit_kp=args.limit_kp)
                if T is None or inliers < args.min_inliers:
                    continue
                # Compose: H_target = H_anchor @ T(target->anchor)
                H_anchor = anchor_H[vf_anchor]
                H_target = H_anchor @ T

                if inliers > best["inliers"]:
                    best.update({"inliers": inliers, "good": good, "vf_anchor": vf_anchor, "H_target": H_target})

            if best["vf_anchor"] is None:
                audit_rows.append((gid, vf_target, "match FAIL", 0, 0))
                print(f"  - {vf_target}: match FAIL")
                continue

            # Write out
            write_homography_json(out_json, best["H_target"])
            audit_rows.append((gid, vf_target, "propagated", best["inliers"], best["good"]))
            print(f"  ✓ {vf_target}: propagated (inliers={best['inliers']}, good={best['good']}) from {best['vf_anchor']}")

    # Write audit
    audit = pd.DataFrame(audit_rows, columns=["game_id", "video_file", "status", "inliers", "good"])
    out_audit = APP_DIR / "reports" / "homography" / "propagation_audit_multi.csv"
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_audit, index=False)
    print(f"\n✓ Wrote audit → {out_audit.resolve()}")
    

if __name__ == "__main__":
    main()
