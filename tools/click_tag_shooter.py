#!/usr/bin/env python3
"""
click_tag_shooter.py
One-click shooter tagging using propagated homographies.

What it does
------------
- Loads t0 keyframes from app/reports/frames/frames_manifest.csv
- Loads homography JSON per video from app/reports/homography/*.json
- Lets you click the shooter location once; maps pixel→pool coords (meters)
- Writes/updates app/reports/corrections/corrections.csv with shooter_x, shooter_y

Controls
--------
- Left click: set point (drawn as a circle)
- ENTER/RETURN: accept & save correction
- R: reset the click for this frame
- S: skip this frame
- Q or ESC: quit the session

Notes
-----
- Pool coords: x in [-10..+10], y in [0..15] (goal center at x=0, goal line at y=0)
- If a correction already exists for a possession_id, it will be overwritten.
- Window auto-scales up to ~90% of your screen size (preserving aspect). You can override with --scale.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# ---------- Paths & constants ----------
APP = Path("app")
FRAMES_DIR = APP / "reports" / "frames"
MANIFEST_CSV = FRAMES_DIR / "frames_manifest.csv"
HOMO_DIR = APP / "reports" / "homography"
CORR_DIR = APP / "reports" / "corrections"
CORR_CSV = CORR_DIR / "corrections.csv"

POOL_X_MIN, POOL_X_MAX = -10.0, 10.0
POOL_Y_MIN, POOL_Y_MAX = 0.0, 15.0

# ---------- Screen size helpers ----------
def get_screen_size() -> Tuple[int, int]:
    """
    Best-effort screen resolution getter. Prefers tkinter (cross-platform).
    Falls back to Windows ctypes; last resort 1920x1080.
    """
    # Try tkinter (works on Windows/macOS/Linux when display available)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w and h:
            return int(w), int(h)
    except Exception:
        pass

    # Fallback for Windows
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        if w and h:
            return int(w), int(h)
    except Exception:
        pass

    # Last resort
    return 1920, 1080

def compute_display_scale(img_w: int, img_h: int, user_scale: Optional[float]) -> float:
    """
    If user provides --scale, use it. Otherwise compute a scale so the image fits
    within ~90% of the screen (preserving aspect).
    """
    if user_scale and user_scale > 0:
        return float(user_scale)
    scr_w, scr_h = get_screen_size()
    max_w = int(scr_w * 0.9)
    max_h = int(scr_h * 0.9)
    sw = max_w / img_w
    sh = max_h / img_h
    scale = min(sw, sh)
    # Avoid upscaling tiny amounts or downscaling too much
    return float(scale)

# ---------- Data I/O ----------
def load_manifest() -> pd.DataFrame:
    if not MANIFEST_CSV.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST_CSV}")
    mf = pd.read_csv(MANIFEST_CSV, dtype=str)
    mf = mf[mf["context"] == "t0"].copy()
    return mf

def load_shots(shots_csv: Path) -> pd.DataFrame:
    if not shots_csv.exists():
        raise SystemExit(f"Missing shots file: {shots_csv}")
    df = pd.read_csv(shots_csv, dtype=str)
    return df

def load_homography(video_file: str) -> Optional[np.ndarray]:
    jpath = HOMO_DIR / f"{Path(video_file).stem}.json"
    if not jpath.exists():
        return None
    try:
        data = json.loads(jpath.read_text())
        H = np.array(data.get("H"), dtype=float)
        if H.shape == (3, 3):
            return H
    except Exception:
        return None
    return None

# ---------- Geometry ----------
def pixel_to_pool(H: np.ndarray, pt: Tuple[int, int]) -> Tuple[float, float]:
    """Map image pixel (u, v) → pool coords (x_m, y_m) via homography H."""
    u, v = float(pt[0]), float(pt[1])
    vec = np.array([u, v, 1.0], dtype=float)
    w = H @ vec
    if abs(w[2]) < 1e-9:
        raise ValueError("Degenerate homography projection (w≈0)")
    x = w[0] / w[2]
    y = w[1] / w[2]
    x = float(np.clip(x, POOL_X_MIN, POOL_X_MAX))
    y = float(np.clip(y, POOL_Y_MIN, POOL_Y_MAX))
    return x, y

# ---------- Corrections ----------
def ensure_corr_header():
    CORR_DIR.mkdir(parents=True, exist_ok=True)
    if not CORR_CSV.exists():
        pd.DataFrame(columns=["possession_id", "shooter_x", "shooter_y"]).to_csv(CORR_CSV, index=False)

def upsert_correction(possession_id: str, shooter_x: float, shooter_y: float):
    ensure_corr_header()
    corr = pd.read_csv(CORR_CSV, dtype=str) if CORR_CSV.exists() else pd.DataFrame(columns=["possession_id","shooter_x","shooter_y"])
    corr = corr[corr["possession_id"] != possession_id].copy()
    corr = pd.concat([
        corr,
        pd.DataFrame([{
            "possession_id": possession_id,
            "shooter_x": f"{shooter_x:.3f}",
            "shooter_y": f"{shooter_y:.3f}",
        }])
    ], ignore_index=True)
    corr.to_csv(CORR_CSV, index=False)

# ---------- Queue building ----------
def pick_rows(df_shots: pd.DataFrame, mf: pd.DataFrame,
              only_game: Optional[str], only_pos_csv: Optional[Path]) -> pd.DataFrame:
    key = ["possession_id", "video_file"]
    j = pd.merge(
        df_shots[["possession_id", "game_id", "video_file"]].drop_duplicates(),
        mf[["possession_id", "video_file", "frame_path"]],
        on=key, how="inner"
    )
    if only_game:
        j = j[j["game_id"].astype(str) == str(only_game)].copy()
    if only_pos_csv:
        if not only_pos_csv.exists():
            raise SystemExit(f"--only_pos_csv not found: {only_pos_csv}")
        want = pd.read_csv(only_pos_csv, dtype=str)["possession_id"].astype(str).tolist()
        j = j[j["possession_id"].astype(str).isin(want)].copy()
    j = j.sort_values(["game_id", "possession_id"]).reset_index(drop=True)
    return j

# ---------- UI helpers ----------
def draw_overlay(img: np.ndarray, click_pt: Optional[Tuple[int,int]], xym: Optional[Tuple[float,float]]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    # black strip for text
    cv2.rectangle(out, (10,10), (max(600, w-10), 110), (0,0,0), -1)
    cv2.putText(out, "Click the SHOOTER at release (t0). ENTER=save  R=reset  S=skip  Q=quit",
                (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, "Purpose: capture shooter_x/y (meters) → recompute distance/angle accurately.",
                (20,85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2, cv2.LINE_AA)

    if xym is not None:
        cv2.putText(out, f"x={xym[0]:.2f} m, y={xym[1]:.2f} m", (20, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
    if click_pt is not None:
        cv2.circle(out, click_pt, 10, (0,255,255), -1, cv2.LINE_AA)
        cv2.circle(out, click_pt, 18, (0,255,255), 2, cv2.LINE_AA)
    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="One-click shooter tagging → corrections.csv")
    ap.add_argument("--shots_csv", default=str(APP / "shots.csv"))
    ap.add_argument("--only_game", default=None, help="Just one game_id")
    ap.add_argument("--only_pos_csv", default=None, help="CSV with possession_id column to limit the queue")
    ap.add_argument("--scale", type=float, default=None, help="Force display scale (else auto-fit to screen)")
    args = ap.parse_args()

    df_shots = load_shots(Path(args.shots_csv))
    mf = load_manifest()
    queue = pick_rows(df_shots, mf,
                      only_game=args.only_game,
                      only_pos_csv=Path(args.only_pos_csv) if args.only_pos_csv else None)

    if queue.empty:
        print("Nothing to tag (no matching frames).")
        return

    ensure_corr_header()
    cv2.namedWindow("tag", cv2.WINDOW_NORMAL)

    print(f"Tagging {len(queue)} frames. Controls: click → ENTER save, R reset, S skip, Q quit.")
    for _, row in queue.iterrows():
        poss = row["possession_id"]
        game = row["game_id"]
        vfile = row["video_file"]
        fpath = row["frame_path"]

        H = load_homography(vfile)
        if H is None:
            print(f"[skip] {poss} (game {game}, {vfile}): no homography JSON")
            continue
        if not Path(fpath).exists():
            print(f"[skip] {poss} (game {game}, {vfile}): missing frame {fpath}")
            continue

        img = cv2.imread(fpath)
        if img is None:
            print(f"[skip] {poss} (game {game}, {vfile}): failed to read {fpath}")
            continue

        # ----- compute display scale -----
        img_h, img_w = img.shape[:2]
        disp_scale = compute_display_scale(img_w, img_h, args.scale)
        if disp_scale != 1.0:
            disp_img = cv2.resize(img, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_LINEAR)
        else:
            disp_img = img

        # Size the window to the image
        cv2.resizeWindow("tag", disp_img.shape[1], disp_img.shape[0])

        click_pt: Optional[Tuple[int,int]] = None
        xym: Optional[Tuple[float,float]] = None

        def _on_mouse(event, x, y, flags, userdata):
            nonlocal click_pt, xym
            if event == cv2.EVENT_LBUTTONDOWN:
                click_pt = (int(x), int(y))
                # Map back to original pixel if scaled
                px = int(round(x / disp_scale))
                py = int(round(y / disp_scale))
                try:
                    x_m, y_m = pixel_to_pool(H, (px, py))
                    xym = (x_m, y_m)
                except Exception:
                    xym = None
        cv2.setMouseCallback("tag", _on_mouse)

        while True:
            vis = draw_overlay(disp_img, click_pt, xym)
            title = f"{poss} (game {game})  •  {vfile}"
            cv2.setWindowTitle("tag", title)
            cv2.imshow("tag", vis)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):  # ENTER
                if click_pt is None or xym is None:
                    continue
                upsert_correction(possession_id=poss, shooter_x=xym[0], shooter_y=xym[1])
                print(f"✓ saved {poss}: shooter_x={xym[0]:.3f}, shooter_y={xym[1]:.3f}")
                break
            elif key in (ord('r'), ord('R')):
                click_pt = None
                xym = None
            elif key in (ord('s'), ord('S')):
                print(f"- skipped {poss}")
                break
            elif key in (27, ord('q'), ord('Q')):  # ESC or Q
                print("Exiting.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"Done. Corrections at: {CORR_CSV.resolve()}")

if __name__ == "__main__":
    main()
