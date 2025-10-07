# tools/extract_keyframes.py
# -*- coding: utf-8 -*-
"""
Extracts reference frames around each labeled event for downstream CV tasks.

Outputs:
  - Images: app/reports/frames/<file-prefix>_<CONTEXT>.jpg   (CONTEXT ∈ {tm,t0,tp})
  - Manifest: app/reports/frames/frames_manifest.csv
      columns: game_id, possession_id, video_file, context, t_s, frame_path

Key features:
  - Safe, slugified filenames (no slashes/backslashes/# etc.)
  - Clamp timestamps via data/videos.csv (duration_s) or via video length
  - Backend fallback (FFMPEG → default → MSMF)
  - --force to overwrite existing frames and rebuild manifest
"""

from __future__ import annotations
from pathlib import Path
import argparse
import math
import re
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
DATA = ROOT / "data"
FRAMES_DIR = APP / "reports" / "frames"
MANIFEST_CSV = FRAMES_DIR / "frames_manifest.csv"
VIDEOS_CSV = DATA / "videos.csv"
SHOTS_CSV = APP / "shots.csv"

# ---------- Helpers ----------
_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
def slugify(s: str, max_len: int = 60) -> str:
    """Convert any string to a safe slug: lowercase, alnum + single dashes."""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", "/")  # normalize separators
    s = s.strip().lower()
    s = _SLUG_RE.sub("-", s).strip("-")
    return s[:max_len] if max_len else s

def parse_mmss(mmss: str) -> Optional[float]:
    """Parse 'MM:SS' (or 'M:SS') to seconds (float)."""
    if not isinstance(mmss, str):
        return None
    mmss = mmss.strip()
    if not mmss:
        return None
    parts = mmss.split(":")
    if len(parts) != 2:
        return None
    try:
        m = int(parts[0])
        s = float(parts[1])
        return max(0.0, m * 60.0 + s)
    except Exception:
        return None

def try_open_video(path: str) -> Optional[cv2.VideoCapture]:
    """Try several backends to open a video on Windows reliably."""
    # Prefer FFMPEG, then default API, then MSMF
    apis = [cv2.CAP_FFMPEG, 0, cv2.CAP_MSMF]
    for api in apis:
        cap = cv2.VideoCapture(path, api)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return None

def get_frame_at_second(cap: cv2.VideoCapture, t_s: float) -> Optional[np.ndarray]:
    """Grab a frame near t_s seconds using FPS to compute frame index."""
    if cap is None or not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        # fallback: just read the first frame
        ok, img = cap.read()
        return img if ok else None
    frame_idx = int(round(t_s * fps))
    # clamp
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_frames > 0:
        frame_idx = max(0, min(frame_idx, n_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    return img if ok else None

def clamp_time(t: float, dur: Optional[float]) -> float:
    if t < 0:
        return 0.0
    if dur is not None and dur > 0:
        return min(t, max(0.0, dur - 1e-3))
    return t

def load_videos_registry() -> Dict[str, Dict]:
    """Load data/videos.csv into a dict keyed by source_video_id (filename)."""
    reg = {}
    if VIDEOS_CSV.exists():
        vdf = pd.read_csv(VIDEOS_CSV, dtype=str)
        for _, r in vdf.iterrows():
            vid = str(r.get("source_video_id", "")).strip()
            pth = str(r.get("source_video_path", "")).strip()
            fps = r.get("fps", None)
            dur = r.get("duration_s", None)
            try:
                dur_val = float(dur) if dur not in (None, "", "nan") else None
            except Exception:
                dur_val = None
            if vid:
                reg[vid] = {
                    "path": pth,
                    "fps": fps,
                    "duration_s": dur_val
                }
    return reg

def build_file_prefix(row: pd.Series) -> str:
    """
    Construct a safe, informative prefix for JPG names, WITHOUT nested dirs.
    Example: 1_1-p0008-q1-4-post-bar
    """
    gid = slugify(row.get("game_id", ""))
    pid = slugify(row.get("possession_id", ""))
    qtr = slugify(row.get("period", ""))
    shooter = slugify(row.get("player_number", ""))
    stype = slugify(row.get("shot_type", ""))
    # Minimal but helpful
    parts = [gid, pid]
    if qtr: parts.append(f"q{qtr}")
    if shooter: parts.append(f"n{shooter}")
    if stype: parts.append(stype)
    return "-".join([p for p in parts if p])

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Extract keyframes around labeled timestamps into app/reports/frames/")
    ap.add_argument("--shots", default=str(SHOTS_CSV), help="Path to app/shots.csv")
    ap.add_argument("--videos", default=str(VIDEOS_CSV), help="Path to data/videos.csv")
    ap.add_argument("--out_dir", default=str(FRAMES_DIR), help="Output directory for frames")
    ap.add_argument("--context_s", type=float, default=2.0, help="Seconds before/after event for tm/tp")
    ap.add_argument("--limit", type=int, default=None, help="Max rows to process (for testing)")
    ap.add_argument("--only_missing", action="store_true", help="Skip rows that already have all frames")
    ap.add_argument("--force", action="store_true", help="Overwrite existing frames and rebuild manifest")
    args = ap.parse_args()

    shots_path = Path(args.shots)
    videos_path = Path(args.videos)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    if not shots_path.exists():
        raise SystemExit(f"Missing shots CSV: {shots_path}")
    df = pd.read_csv(shots_path, dtype=str)

    # Keep only rows with required time fields present
    need_cols = ["video_file", "video_timestamp_mmss", "possession_id", "game_id"]
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"shots.csv missing required column: {c}")

    # Filter to rows with valid timestamps
    df["t0"] = df["video_timestamp_mmss"].map(parse_mmss)
    df = df[~df["t0"].isna()].copy()
    if df.empty:
        print("No rows with valid MM:SS timestamps; nothing to extract.")
        return

    # Optionally limit rows
    if args.limit is not None:
        df = df.iloc[:int(args.limit)].copy()

    # Videos registry
    vreg = load_videos_registry()
    if not vreg:
        print("WARNING: data/videos.csv empty or missing; path clamping by duration may be limited.")

    # Prepare manifest accumulation
    manifest_rows = []
    written = 0
    skipped_existing = 0
    failed = 0

    # If forcing, clear old manifest (frames remain but will be overwritten)
    if args.force and MANIFEST_CSV.exists():
        MANIFEST_CSV.unlink(missing_ok=True)

    # If not forcing, load existing manifest to know what “already exists”
    existing = pd.DataFrame()
    if MANIFEST_CSV.exists() and not args.force:
        try:
            existing = pd.read_csv(MANIFEST_CSV, dtype=str)
        except Exception:
            existing = pd.DataFrame()
    have = set()
    if not existing.empty:
        for _, r in existing.iterrows():
            k = (str(r.get("game_id","")), str(r.get("possession_id","")), str(r.get("video_file","")), str(r.get("context","")))
            have.add(k)

    # Per-row extraction
    for _, row in df.iterrows():
        gid = str(row["game_id"])
        pid = str(row["possession_id"])
        vfile = str(row["video_file"]).strip()
        t0 = float(row["t0"])

        reg = vreg.get(vfile, {})
        vpath = reg.get("path")
        dur_s = reg.get("duration_s", None)

        if not vpath or not Path(vpath).exists():
            print(f"ERROR: video path not found for {vfile} → {vpath}")
            failed += 1
            continue

        # Compute tm/t0/tp times (clamped)
        ctx = float(args.context_s)
        t_tm = clamp_time(t0 - ctx, dur_s)
        t_t0 = clamp_time(t0, dur_s)
        t_tp = clamp_time(t0 + ctx, dur_s)

        # output filenames (flat; safe)
        prefix = build_file_prefix(row)
        if not prefix:
            # minimal fallback (still safe)
            prefix = f"{slugify(gid)}-{slugify(pid)}-{slugify(vfile)}"

        targets = {
            "tm": out_dir / f"{prefix}_tm.jpg",
            "t0": out_dir / f"{prefix}_t0.jpg",
            "tp": out_dir / f"{prefix}_tp.jpg",
        }

        # skip if all exist and only_missing
        already = all(p.exists() for p in targets.values())
        if args.only_missing and already:
            skipped_existing += 1
            # also append to manifest from existing (ensures continuity)
            for ctx_name, pth in targets.items():
                manifest_rows.append({
                    "game_id": gid,
                    "possession_id": pid,
                    "video_file": vfile,
                    "context": ctx_name,
                    "t_s": {"tm": t_tm, "t0": t_t0, "tp": t_tp}[ctx_name],
                    "frame_path": str(pth),
                })
            continue

        # Open the video (prefer FFMPEG)
        cap = try_open_video(vpath)
        if cap is None:
            print(f"ERROR opening {vpath}: could not open with available backends")
            failed += 1
            continue

        # Extract frames
        try:
            triplet = [("tm", t_tm), ("t0", t_t0), ("tp", t_tp)]
            for name, t_s in triplet:
                img = get_frame_at_second(cap, t_s)
                if img is None:
                    print(f"  - {vfile}: failed to read frame at {t_s:.2f}s ({name})")
                    continue
                # Ensure overwrite if --force set; otherwise cv2.imwrite will overwrite anyway
                ok = cv2.imwrite(str(targets[name]), img)
                if ok:
                    written += 1
                    manifest_rows.append({
                        "game_id": gid,
                        "possession_id": pid,
                        "video_file": vfile,
                        "context": name,
                        "t_s": t_s,
                        "frame_path": str(targets[name]),
                    })
                else:
                    print(f"  - {vfile}: cv2.imwrite failed for {targets[name]}")
        finally:
            cap.release()

    # Write manifest (append to existing unless --force)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(manifest_rows)
    if not out_df.empty:
        if MANIFEST_CSV.exists() and not args.force:
            # append unique rows
            old = pd.read_csv(MANIFEST_CSV, dtype=str)
            merged = pd.concat([old, out_df], ignore_index=True)
            # Drop perfect duplicates
            merged = merged.drop_duplicates(
                subset=["game_id", "possession_id", "video_file", "context", "frame_path"],
                keep="last",
            )
            merged.to_csv(MANIFEST_CSV, index=False)
        else:
            out_df.to_csv(MANIFEST_CSV, index=False)

    print(f"\n✓ Extraction complete. Wrote {written} frames.")
    if skipped_existing:
        print(f"  (Skipped {skipped_existing} rows that already had frames; use --force to overwrite.)")
    if failed:
        print(f"  (Encountered {failed} video open/path issues — see errors above.)")
    print(f"→ Manifest: {MANIFEST_CSV.resolve()}")

if __name__ == "__main__":
    main()
