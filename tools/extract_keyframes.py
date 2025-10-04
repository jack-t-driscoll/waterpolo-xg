# tools/extract_keyframes.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
DATA_DIR = ROOT / "data"
OUT_DIR_DEFAULT = APP_DIR / "reports" / "frames"

# --- Time parsing ---
def mmss_to_seconds(s: str) -> float:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m, sec = int(parts[0]), float(parts[1])
            return m * 60 + sec
        elif len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + sec
        else:
            return float(s)  # already seconds
    except Exception:
        return np.nan

# --- Path utils ---
def clean_path(p: str) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    s = str(p).strip()
    # strip wrapping quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s

# --- Video helpers ---
def open_video(path: Path):
    """Open video via FFmpeg backend only; raise clear errors if missing or unsupported."""
    p = Path(clean_path(str(path)))
    if not p.exists():
        raise RuntimeError(f"Path does not exist: {repr(str(p))}")
    cap = cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 0:
            fps = 30.0
        return cap, float(fps)
    raise RuntimeError(f"Failed to open video with FFmpeg: {repr(str(p))}")

def grab_frame_at_time(cap, fps: float, t_s: float):
    if np.isnan(t_s):
        return None
    frame_idx = max(0, int(round(t_s * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --- Main ---
def main():
    ap = argparse.ArgumentParser(description="Extract per-shot keyframes (and context) from source videos")
    ap.add_argument("--shots", default=str(APP_DIR / "shots.csv"), help="Path to app/shots.csv")
    ap.add_argument("--videos", default=str(DATA_DIR / "videos.csv"), help="Path to data/videos.csv")
    ap.add_argument("--out_dir", default=str(OUT_DIR_DEFAULT), help="Output directory for frames")
    ap.add_argument("--context_s", type=float, default=0.0, help="Also extract frames at ±context seconds (0 to disable)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N shots (0 = all)")
    args = ap.parse_args()

    shots_path = Path(args.shots)
    videos_path = Path(args.videos)
    out_dir = Path(args.out_dir)

    if not shots_path.exists():
        raise FileNotFoundError(f"Missing shots CSV: {shots_path}")
    if not videos_path.exists():
        raise FileNotFoundError(f"Missing videos CSV: {videos_path}")

    ensure_dir(out_dir)

    shots = pd.read_csv(shots_path, dtype=str)
    vids = pd.read_csv(videos_path, dtype=str)

    # Keep only shots
    if "event_type" in shots.columns:
        shots = shots[shots["event_type"].astype(str).str.lower() == "shot"].copy()

    # Parse times
    if "video_timestamp_mmss" in shots.columns:
        shots["t_s"] = shots["video_timestamp_mmss"].map(mmss_to_seconds)
    else:
        shots["t_s"] = np.nan

    # Join for video paths
    required_shot_cols = {"video_file"}
    required_vids_cols = {"source_video_id", "source_video_path"}
    if not required_shot_cols.issubset(shots.columns):
        missing = required_shot_cols - set(shots.columns)
        raise ValueError(f"shots.csv missing columns: {missing}")
    if not required_vids_cols.issubset(vids.columns):
        missing = required_vids_cols - set(vids.columns)
        raise ValueError(f"videos.csv missing columns: {missing}")

    df = shots.merge(
        vids[["source_video_id", "source_video_path"]],
        left_on="video_file", right_on="source_video_id", how="left"
    )
    df["source_video_path"] = df["source_video_path"].map(clean_path)

    # Report unmapped
    unmapped = df["source_video_path"].isna() | (df["source_video_path"].astype(str).str.strip() == "")
    if unmapped.any():
        n = int(unmapped.sum())
        print(f"Warning: {n} shots have no matching source_video_path in data/videos.csv")

    # Identify optional columns for filename tags
    poss_col = "possession_id" if "possession_id" in df.columns else None
    game_col = "game_id" if "game_id" in df.columns else None
    period_col = "period" if "period" in df.columns else None
    player_col = "player_number" if "player_number" in df.columns else None
    outcome_col = "shot_result" if "shot_result" in df.columns else ("outcome" if "outcome" in df.columns else None)

    # Optionally limit
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    manifest_rows = []

    # Group by video to minimize open/close
    for vid_path_str, group in df.groupby("source_video_path", dropna=False):
        if pd.isna(vid_path_str) or str(vid_path_str).strip() == "":
            continue
        vid_path = Path(vid_path_str)

        print(f"Opening: {repr(str(vid_path))}")
        try:
            cap, fps = open_video(vid_path)
        except Exception as e:
            print(f"ERROR opening {vid_path}: {e}")
            continue

        try:
            for i, row in group.iterrows():
                t0 = row.get("t_s", np.nan)
                if np.isnan(t0):
                    continue

                # Build filename base
                tags = []
                if game_col:   tags.append(str(row.get(game_col, "")))
                if poss_col:   tags.append(str(row.get(poss_col, "")))
                if period_col: tags.append(f"Q{str(row.get(period_col, ''))}")
                if player_col: tags.append(f"#{str(row.get(player_col, ''))}")
                if outcome_col:tags.append(str(row.get(outcome_col, "")))
                base = "_".join([x for x in tags if x]) or f"row{i}"

                # t0
                frame = grab_frame_at_time(cap, fps, t0)
                if frame is not None:
                    out_path = out_dir / f"{base}_t0.jpg"
                    cv2.imwrite(str(out_path), frame)
                    manifest_rows.append({
                        "row_idx": i,
                        "video": str(vid_path),
                        "t_s": float(t0),
                        "context": "t0",
                        "frame_path": str(out_path),
                        "possession_id": str(row.get("possession_id", "")),
                        "video_file": str(row.get("video_file", "")),
                    })

                # context frames (clamp tm to 0.0 if negative)
                ctx = float(args.context_s)
                if ctx > 0:
                    for label, t in [("tm", t0 - ctx), ("tp", t0 + ctx)]:
                        t_eff = max(0.0, float(t))
                        clipped = (t_eff != t)
                        f2 = grab_frame_at_time(cap, fps, t_eff)
                        if f2 is not None:
                            out_path = out_dir / f"{base}_{label}.jpg"
                            cv2.imwrite(str(out_path), f2)
                            row_dict = {
                                "row_idx": i,
                                "video": str(vid_path),
                                "t_s": float(t_eff),
                                "context": label,
                                "frame_path": str(out_path),
                                "possession_id": str(row.get("possession_id", "")),
                                "video_file": str(row.get("video_file", "")),
                            }
                            if label == "tm":
                                row_dict["clipped_to_zero"] = str(bool(clipped))
                            manifest_rows.append(row_dict)
        finally:
            cap.release()

    # Write manifest
    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        out_csv = out_dir / "frames_manifest.csv"
        manifest.to_csv(out_csv, index=False)
        print(f"✓ Wrote {len(manifest)} frames to {out_dir} and manifest {out_csv}")
    else:
        print("No frames extracted (check timestamps and mapping).")

if __name__ == "__main__":
    main()
