#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

APP = Path("app")
MANIFEST = APP / "reports" / "frames" / "frames_manifest.csv"
HOMO_DIR = APP / "reports" / "homography"
OVERRIDE = APP / "reports" / "corrections" / "shot_xy_overrides.csv"

def load_homography(json_path: Path) -> np.ndarray:
    data = json.loads(json_path.read_text())
    # be flexible with key names; prefer image->pool homography
    if "H_image_to_pool" in data:
        H = np.array(data["H_image_to_pool"], dtype=float)
    elif "H" in data:
        # calibration saved as image->pool
        H = np.array(data["H"], dtype=float)
    elif "H_pool_to_image" in data:
        # invert if only pool->image is present
        H = np.linalg.inv(np.array(data["H_pool_to_image"], dtype=float))
    else:
        raise ValueError(f"{json_path} missing homography (expected 'H' or 'H_image_to_pool').")
    if H.shape != (3,3):
        raise ValueError(f"Homography in {json_path} has wrong shape {H.shape}, expected (3,3).")
    return H

def img_to_pool(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Map pixel (u,v) → pool coords (x_m,y_m) using image->pool homography."""
    pt = np.array([u, v, 1.0], dtype=float)
    xyw = H @ pt
    if abs(xyw[2]) < 1e-9:
        raise ValueError("Homography mapping returned degenerate w≈0.")
    x = xyw[0] / xyw[2]
    y = xyw[1] / xyw[2]
    return float(x), float(y)

def fit_scale_for_screen(img_w, img_h, target_w=1920, target_h=1080):
    # simple fit within target box; keep aspect ratio
    sw = target_w / img_w
    sh = target_h / img_h
    return min(sw, sh)

def pick_context_row(df_one: pd.DataFrame, preferred: str) -> pd.Series:
    # choose preferred context if exists, else fallback order
    order = [preferred, "t0", "tm", "tp"]
    for c in order:
        row = df_one[df_one["context"] == c]
        if len(row) == 1:
            return row.iloc[0]
    raise SystemExit(f"No frame found for possession {df_one['possession_id'].iloc[0]} (contexts tried: {order}).")

def ensure_overrides_csv():
    OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
    if not OVERRIDE.exists():
        pd.DataFrame(columns=["possession_id","x_m","y_m","source","at"]).to_csv(OVERRIDE, index=False)

def append_override(pid: str, x_m: float, y_m: float, source: str = "manual_click"):
    ensure_overrides_csv()
    df = pd.read_csv(OVERRIDE, dtype=str)
    # drop any existing entry for this possession (keep latest)
    df = df[df["possession_id"] != pid]
    row = {
        "possession_id": pid,
        "x_m": f"{x_m:.3f}",
        "y_m": f"{y_m:.3f}",
        "source": source,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(OVERRIDE, index=False)

def main():
    ap = argparse.ArgumentParser(description="Tag one shot location (in pool meters) for a possession_id.")
    ap.add_argument("--possession_id", required=True, help="e.g., 10_P0003")
    ap.add_argument("--context", default="t0", choices=["t0","tm","tp"], help="frame context to display (default: t0)")
    ap.add_argument("--scale", type=float, default=None, help="optional UI scale (e.g., 1.5). If omitted, fits into 1920x1080.")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"Missing frames manifest: {MANIFEST}")

    mf = pd.read_csv(MANIFEST, dtype=str)
    mf = mf[mf["possession_id"] == args.possession_id]
    if mf.empty:
        raise SystemExit(f"possession_id not found in manifest: {args.possession_id}")

    row = pick_context_row(mf, preferred=args.context)
    frame_path = Path(row["frame_path"])
    video_file = row["video_file"]

    if not frame_path.exists():
        raise SystemExit(f"Frame not found on disk: {frame_path}")

    H_json = HOMO_DIR / f"{video_file}.json"
    if not H_json.exists():
        raise SystemExit(f"Missing homography for video: {H_json}")

    H = load_homography(H_json)

    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read frame image: {frame_path}")

    h0, w0 = img.shape[:2]
    if args.scale is None:
        sc = fit_scale_for_screen(w0, h0, 1920, 1080)
    else:
        sc = float(args.scale)

    W, Hh = int(w0 * sc), int(h0 * sc)
    disp = cv2.resize(img, (W, Hh), interpolation=cv2.INTER_AREA)

    click_pt = {"u": None, "v": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt["u"] = x / sc
            click_pt["v"] = y / sc
            # draw a small marker
            cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("tag_shot_xy", disp)

    cv2.namedWindow("tag_shot_xy", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("tag_shot_xy", W, Hh)
    cv2.imshow("tag_shot_xy", disp)
    cv2.setMouseCallback("tag_shot_xy", on_mouse)

    print(f"\nTagging possession_id={args.possession_id}  video={video_file}  context={row['context']}")
    print("Instructions:")
    print("  • Click once on the shooter’s location at release (the player’s body center at release).")
    print("  • Press 's' to save override, or 'q' to cancel.\n")

    while True:
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            print("Canceled. No changes written.")
            break
        if k == ord('s'):
            if click_pt["u"] is None:
                print("No click registered yet. Click on the image, then press 's'.")
                continue
            x_m, y_m = img_to_pool(H, click_pt["u"], click_pt["v"])
            append_override(args.possession_id, x_m, y_m, source="manual_click")
            print(f"✓ Saved override → {OVERRIDE}")
            print(f"  possession_id={args.possession_id}  x_m={x_m:.3f}  y_m={y_m:.3f}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
