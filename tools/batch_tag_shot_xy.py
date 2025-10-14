import argparse, json, os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

APP = Path("app")
FRAMES = APP / "reports" / "frames" / "frames_manifest.csv"
HOMO_DIR = APP / "reports" / "homography"
OUT_DIR = APP / "reports" / "corrections"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OVERRIDES = OUT_DIR / "shot_xy_overrides.csv"

POOL_W, POOL_H = 20.0, 15.0  # meters (x in [-10,+10], y in [0,15])

def read_manifest():
    df = pd.read_csv(FRAMES, dtype=str)
    need = {"possession_id","video_file","context","frame_path"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{FRAMES} missing columns: {missing}")
    return df

def load_homography(video_file: str):
    jf = HOMO_DIR / f"{video_file}.json"
    if not jf.exists():
        raise FileNotFoundError(f"No homography file found: {jf}")
    data = json.loads(jf.read_text())
    if "H" in data:
        H = np.array(data["H"], dtype=np.float64)
        if data.get("H_dir") == "pool_to_image":
            H = np.array(data.get("H_inv", np.linalg.inv(H)), dtype=np.float64)
        return H
    if "H_img_to_pool" in data:
        return np.array(data["H_img_to_pool"], dtype=np.float64)
    if "H_inv" in data:
        return np.array(data["H_inv"], dtype=np.float64)
    raise ValueError(f"Homography JSON lacks usable matrix: {jf}")

def perspective_img_to_pool(H, x, y):
    pts = np.array([[[x, y]]], dtype=np.float64)
    dst = cv2.perspectiveTransform(pts, H)
    X, Y = float(dst[0,0,0]), float(dst[0,0,1])
    return X, Y

def clamp_pool(X, Y):
    Xc = max(-POOL_W/2, min(POOL_W/2, X))
    Yc = max(0.0, min(POOL_H, Y))
    return Xc, Yc

def upsert_override(pid, x_m, y_m):
    row = {"possession_id": pid, "x_m": f"{x_m:.4f}", "y_m": f"{y_m:.4f}"}
    if OVERRIDES.exists() and OVERRIDES.stat().st_size > 0:
        df = pd.read_csv(OVERRIDES, dtype=str)
        if "possession_id" not in df.columns:
            raise SystemExit(f"{OVERRIDES} exists but is malformed (no possession_id).")
        df = df[["possession_id","x_m","y_m"]]
        if (df["possession_id"] == pid).any():
            df.loc[df["possession_id"] == pid, ["x_m","y_m"]] = [row["x_m"], row["y_m"]]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=["possession_id","x_m","y_m"])
    df.to_csv(OVERRIDES, index=False)

def load_pid_list(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str)
        if "possession_id" not in df.columns:
            raise SystemExit(f"{path} must have a possession_id column.")
        pids = df["possession_id"].dropna().astype(str).unique().tolist()
    else:
        # plain text: one pid per line
        pids = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return pids

def tag_one(df_manifest, pid, scale):
    row = df_manifest[(df_manifest["possession_id"]==pid) & (df_manifest["context"]=="t0")]
    if row.empty:
        print(f"  - {pid}: no t0 frame in manifest → skip")
        return "skip"
    frame_path = row.iloc[0]["frame_path"]
    video_file = row.iloc[0]["video_file"]
    if not os.path.exists(frame_path):
        print(f"  - {pid}: t0 frame missing on disk → skip")
        return "skip"

    try:
        H = load_homography(video_file)
    except Exception as e:
        print(f"  - {pid}: homography load error for {video_file}: {e} → skip")
        return "skip"

    img = cv2.imread(frame_path)
    if img is None:
        print(f"  - {pid}: failed to read frame {frame_path} → skip")
        return "skip"

    img_disp = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1.0 else img.copy()

    print(f"  • {pid}  (video={video_file})")
    print("    Click where the ball left the shooter’s hand (plan-view location).")
    print("    Keys: Enter=save  r=redo  s=skip  q=quit batch")

    clicked = {"pt": None}
    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    win = f"XY Tag: {pid}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    overlay = img_disp.copy()
    result = None
    while True:
        disp = overlay.copy()
        if clicked["pt"] is not None:
            cv2.circle(disp, clicked["pt"], 6, (0,255,0), -1)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('r'):
            clicked["pt"] = None
            overlay = img_disp.copy()
        elif key == ord('s'):
            result = "skip"
            break
        elif key == ord('q'):
            result = "quit"
            break
        elif key in (13, 10):  # Enter
            if clicked["pt"] is None:
                print("    (no click yet)")
                continue
            sx, sy = 1.0/scale, 1.0/scale
            x_img = clicked["pt"][0] * sx
            y_img = clicked["pt"][1] * sy
            X_m, Y_m = perspective_img_to_pool(H, x_img, y_img)
            X_m, Y_m = clamp_pool(X_m, Y_m)
            upsert_override(pid, X_m, Y_m)
            print(f"    ✓ saved x_m={X_m:.3f}, y_m={Y_m:.3f}")
            result = "saved"
            break

    cv2.destroyWindow(win)
    return result

def main():
    ap = argparse.ArgumentParser(description="Batch tag shot (x_m,y_m) from t0 frames using homography.")
    ap.add_argument("--list", required=True, help="Path to a txt list (one possession_id per line) or CSV with a possession_id column.")
    ap.add_argument("--scale", type=float, default=1.6, help="Display scale for the image window.")
    args = ap.parse_args()

    pids = load_pid_list(Path(args.list))
    if not pids:
        raise SystemExit("No possession_ids found in the list.")
    df_manifest = read_manifest()

    print(f"Loaded {len(pids)} targets. Overrides file: {OVERRIDES}")
    done = 0
    for i, pid in enumerate(pids, 1):
        print(f"[{i}/{len(pids)}]")
        res = tag_one(df_manifest, pid, args.scale)
        if res == "quit":
            print("Batch aborted by user.")
            break
        if res == "saved":
            done += 1
    print(f"\nCompleted. Saved {done} of {len(pids)}.")

    print("\nTo materialize overrides into features, run:")
    print("  python tools\\build_shot_xy_corrections.py")
    print("  python tools\\normalize_and_export.py --input app\\shots.csv --output app\\features_shots.csv --corrections app\\reports\\corrections\\corrections_shot_xy.csv")

if __name__ == "__main__":
    main()
