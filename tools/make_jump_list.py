# tools/make_jump_list.py
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
DATA_DIR = ROOT / "data"

def mmss_to_seconds(mmss: str) -> float:
    try:
        s = str(mmss).strip()
        if not s or s.lower() == "nan":
            return np.nan
        parts = s.split(":")
        if len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return m * 60 + s
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + s
        else:
            return float(s)
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser(description="Build VLC jump list from app/shots.csv (+ data/videos.csv if present)")
    ap.add_argument("--shots", default=str(APP_DIR / "shots.csv"), help="Path to shots.csv")
    ap.add_argument("--videos", default=str(DATA_DIR / "videos.csv"), help="Path to videos.csv (optional)")
    ap.add_argument("--out", default=str(APP_DIR / "reports" / "downloads" / "jump_list_vlc.csv"), help="Output CSV path")
    ap.add_argument("--write-bat", action="store_true", help="Also write a Windows .bat file with one VLC command per line")
    args = ap.parse_args()

    shots_path = Path(args.shots)
    vids_path  = Path(args.videos)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(shots_path, dtype=str)
    # shots only
    if "event_type" in df.columns:
        df = df[df["event_type"].astype(str).str.lower() == "shot"].copy()

    # map video path
    video_file_col = "video_file" if "video_file" in df.columns else None
    if vids_path.exists():
        vids = pd.read_csv(vids_path, dtype=str)
        if video_file_col and {"source_video_id","source_video_path"}.issubset(vids.columns):
            df = df.merge(vids[["source_video_id","source_video_path"]],
                          left_on=video_file_col, right_on="source_video_id", how="left")
            df["video_path"] = df["source_video_path"].fillna(df[video_file_col])
        else:
            df["video_path"] = df[video_file_col] if video_file_col else ""
    else:
        df["video_path"] = df[video_file_col] if video_file_col else ""

    # time
    ts_col = "video_timestamp_mmss" if "video_timestamp_mmss" in df.columns else None
    df["start_time_s"] = df[ts_col].map(mmss_to_seconds) if ts_col else np.nan

    # label (Team • # • Qn if available)
    lbl = []
    if "our_team_name" in df.columns:
        lbl.append(df["our_team_name"].astype(str))
    elif "team" in df.columns:
        lbl.append(df["team"].astype(str))
    if "player_number" in df.columns:
        lbl.append("#" + df["player_number"].astype(str))
    if "period" in df.columns:
        lbl.append("Q" + df["period"].astype(str))

    if lbl:
        label = lbl[0]
        for p in lbl[1:]:
            label = label + " • " + p
        df["label"] = label
    else:
        df["label"] = ""

    df["vlc_cmd"] = df.apply(
        lambda r: f'vlc --qt-start-minimized --play-and-exit --start-time="{r["start_time_s"]}" "{r["video_path"]}"',
        axis=1
    )

    out_cols = ["video_path","start_time_s","label","vlc_cmd"]
    jump_df = df[out_cols]
    jump_df.to_csv(out_path, index=False)
    print(f"✓ Wrote {len(jump_df)} rows to {out_path}")

    if args.write_bat:
        bat_path = out_path.with_suffix(".bat")
        with open(bat_path, "w", encoding="utf-8") as f:
            for cmd in jump_df["vlc_cmd"]:
                f.write(cmd + "\n")
        print(f"✓ Wrote batch file: {bat_path}")

if __name__ == "__main__":
    sys.exit(main())
