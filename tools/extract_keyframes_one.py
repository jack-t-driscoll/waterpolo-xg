# tools/extract_keyframes_one.py
# -*- coding: utf-8 -*-
from pathlib import Path
import math
import cv2
import pandas as pd

APP = Path(__file__).resolve().parents[1] / "app"
SHOTS = APP / "shots.csv"
VIDEOS = Path("data/videos.csv")
FRAMES_DIR = APP / "reports" / "frames"
MANIFEST = FRAMES_DIR / "frames_manifest.csv"

def mmss_to_seconds(s: str) -> float:
    s = str(s).strip()
    if not s:
        return math.nan
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m, sec = int(parts[0]), float(parts[1])
            return m * 60 + sec
        elif len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + sec
        return float(s)
    except:
        return math.nan

def open_cap(path: str):
    # Try FFmpeg, MSMF, then default
    for api in [cv2.CAP_FFMPEG, cv2.CAP_MSMF, 0]:
        cap = cv2.VideoCapture(path, apiPreference=api)
        if cap.isOpened():
            return cap
        cap.release()
    return None

def main(pid: str, context_s: float = 2.0):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    shots = pd.read_csv(SHOTS, dtype=str)
    row = shots[shots["possession_id"].astype(str) == pid]
    if row.empty:
        raise SystemExit(f"possession_id not found: {pid}")
    row = row.iloc[0]
    if str(row.get("event_type", "")).lower() != "shot":
        raise SystemExit(f"possession_id {pid} is not a shot (event_type={row.get('event_type')})")

    video_file = str(row.get("video_file", "")).strip()
    t0_str = str(row.get("video_timestamp_mmss", "")).strip()
    if not video_file or not t0_str:
        raise SystemExit(f"Missing video_file or video_timestamp_mmss for {pid}")

    vids = pd.read_csv(VIDEOS, dtype=str)
    m = vids[vids["source_video_id"].astype(str) == video_file]
    if m.empty:
        raise SystemExit(f"videos.csv has no row with source_video_id={video_file}")
    video_path = str(m.iloc[0]["source_video_path"]).strip()
    if not Path(video_path).exists():
        raise SystemExit(f"Video path does not exist: {video_path}")

    t0 = mmss_to_seconds(t0_str)
    if math.isnan(t0):
        raise SystemExit(f"Bad timestamp for {pid}: {t0_str}")

    cap = open_cap(video_path)
    if cap is None:
        raise SystemExit(f"Failed to open video with available backends: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if fps > 0 else math.inf

    def grab(ts, tag):
        if ts < 0 or ts > duration_s:
            return None
        frame_idx = max(0, min(int(round(ts * fps)), max(0, total_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None
        out_path = FRAMES_DIR / f"{pid}_{tag}.jpg"
        ok = cv2.imwrite(str(out_path), frame)
        return str(out_path) if ok else None

    tm = grab(t0 - context_s, "tm")
    t0p = grab(t0, "t0")
    tp = grab(t0 + context_s, "tp")
    cap.release()

    # Update manifest (upsert)
    entries = []
    for tag, path in [("tm", tm), ("t0", t0p), ("tp", tp)]:
        if path is not None:
            entries.append({"possession_id": pid, "video_file": video_file, "context": tag, "frame_path": path})

    if not entries:
        print("No frames written.")
        return

    man_cols = ["possession_id", "video_file", "context", "frame_path"]
    man = pd.read_csv(MANIFEST, dtype=str) if MANIFEST.exists() else pd.DataFrame(columns=man_cols)
    man = man[man["possession_id"] != pid]
    man = pd.concat([man, pd.DataFrame(entries, columns=man_cols)], ignore_index=True)
    man.to_csv(MANIFEST, index=False)

    print(
        f"✓ wrote frames for {pid}:",
        f"tm={'ok' if tm else 'miss'}",
        f"t0={'ok' if t0p else 'miss'}",
        f"tp={'ok' if tp else 'miss'}",
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--possession_id", required=True)
    ap.add_argument("--context_s", type=float, default=2.0)
    args = ap.parse_args()
    main(args.possession_id, context_s=args.context_s)
