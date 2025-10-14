# -*- coding: utf-8 -*-
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

APP = Path("app")
MANIFEST = APP / "reports" / "frames" / "frames_manifest.csv"
HOMO_DIR = APP / "reports" / "homography"
SHOTS_CSV = APP / "shots.csv"

def load_manifest():
    df = pd.read_csv(MANIFEST, dtype=str)
    piv = df.pivot_table(index="possession_id", columns="context",
                         values="frame_path", aggfunc="first")
    return piv

def load_homography(video_file: str):
    base = Path(video_file).with_suffix(".json").name
    path = HOMO_DIR / base
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    mat = data.get("H") or data.get("homography") or data.get("matrix")
    if mat is None:
        return None, None
    H = np.array(mat, dtype=float).reshape(3,3)
    return H, data

def warp_to_pool(H, uv):
    uv1 = np.array([uv[0], uv[1], 1.0], dtype=float)
    XYw = H @ uv1
    if abs(XYw[2]) < 1e-9:
        return None
    X = XYw[0] / XYw[2]
    Y = XYw[1] / XYw[2]
    return float(X), float(Y)

def get_screen_size():
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080

def resolve_full_frame(frame_path: str) -> Path | None:
    """If the manifest points to a 'bar_' thumbnail, switch to the full-size sibling."""
    p = Path(frame_path)
    if not p.exists():
        return None
    name = p.name
    if name.startswith("bar_"):
        full_name = name.replace("bar_", "", 1)
        full_path = p.with_name(full_name)
        if full_path.exists():
            return full_path
    # Otherwise just use the original
    return p

def choose_click(img, win_name="tag", scale=None):
    h, w = img.shape[:2]

    # Compute display size
    if scale is None:
        sw, sh = get_screen_size()
        s = min((sw * 0.92) / w, (sh * 0.92) / h)
    else:
        s = float(scale)

    disp_w = max(1, int(round(w * s)))
    disp_h = max(1, int(round(h * s)))
    disp = cv2.resize(img, (disp_w, disp_h)) if s != 1.0 else img.copy()

    clicked = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    # Resizable window + force the size
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)
    cv2.imshow(win_name, disp)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        canvas = disp.copy()
        if clicked["pt"] is not None:
            cv2.circle(canvas, clicked["pt"], 6, (0, 0, 255), -1)
            cv2.putText(canvas, f"{clicked['pt'][0]},{clicked['pt'][1]}",
                        (clicked["pt"][0]+8, clicked["pt"][1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, canvas)
        k = cv2.waitKey(16) & 0xFF

        # accept (y or Enter)
        if k in (ord('y'), 13):
            if clicked["pt"] is not None:
                break
        # redo (r or Backspace)
        elif k in (ord('r'), 8):
            clicked["pt"] = None
        # skip (s or Esc)
        elif k in (ord('s'), 27):
            clicked["pt"] = None
            break
        # quit
        elif k in (ord('q'),):
            cv2.destroyWindow(win_name)
            return None, None, "quit"

    cv2.destroyWindow(win_name)

    if clicked["pt"] is None:
        return None, None, "skip"

    if s != 1.0:
        u = int(round(clicked["pt"][0] / s))
        v = int(round(clicked["pt"][1] / s))
    else:
        u, v = clicked["pt"]

    return u, v, "ok"

def main():
    ap = argparse.ArgumentParser(description="Batch-tag shooter XY for missing shots")
    ap.add_argument("--list", required=True,
                    help="CSV from diagnostics (must include possession_id, video_file)")
    ap.add_argument("--scale", type=float, default=None,
                    help="Scale factor (e.g. 1.6). If omitted, auto-fit to your screen.")
    ap.add_argument("--context", default="t0", choices=["t0","tm","tp"],
                    help="Which context image to show")
    args = ap.parse_args()

    diag = pd.read_csv(args.list, dtype=str)
    if "possession_id" not in diag.columns:
        sys.exit("List CSV must include possession_id")

    if not SHOTS_CSV.exists():
        sys.exit(f"Missing {SHOTS_CSV}")
    shots = pd.read_csv(SHOTS_CSV, dtype=str)

    piv = load_manifest()
    if args.context not in piv.columns:
        sys.exit(f"Manifest has no '{args.context}' frames; re-run keyframe extraction.")

    todo = diag["possession_id"].dropna().unique().tolist()
    print(f"Tagging {len(todo)} possessions... (y/Enter accept, r redo, s/Esc skip, q quit)")

    updated = skipped = failed = 0

    for pid in todo:
        row = shots[shots["possession_id"] == pid]
        if row.empty:
            print(f"  - {pid}: missing from shots.csv, skipping")
            skipped += 1
            continue

        # manifest frame
        try:
            manifest_path = piv.loc[pid, args.context]
        except KeyError:
            manifest_path = None

        if not isinstance(manifest_path, str):
            print(f"  - {pid}: no frame in manifest for {args.context}, skipping")
            skipped += 1
            continue

        frame_path = resolve_full_frame(manifest_path)
        if frame_path is None or not frame_path.exists():
            print(f"  - {pid}: frame file not found ({manifest_path}), skipping")
            skipped += 1
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  - {pid}: failed to read frame {frame_path}, skipping")
            skipped += 1
            continue

        # video file -> homography
        if "video_file" in row.columns and pd.notna(row["video_file"].values[0]):
            video_file = row["video_file"].values[0]
        else:
            # fallback: infer from folder name
            try:
                folder = Path(manifest_path).parts[-2]
                video_file = folder.split("_")[0] + ".MOV"
            except Exception:
                print(f"  - {pid}: cannot infer video_file, skipping")
                skipped += 1
                continue

        H, _ = load_homography(video_file)
        if H is None:
            print(f"  - {pid}: no homography for {video_file}, skipping")
            skipped += 1
            continue

        u, v, status = choose_click(img, win_name=f"{pid} ({video_file})", scale=args.scale)
        if status == "quit":
            print("  ! quit requested — stopping")
            break
        if status != "ok":
            print(f"  - {pid}: skipped")
            skipped += 1
            continue

        XY = warp_to_pool(H, (u, v))
        if XY is None or not np.isfinite(XY[0]) or not np.isfinite(XY[1]):
            print(f"  - {pid}: homography mapping failed for pixel ({u},{v})")
            failed += 1
            continue

        x_m, y_m = XY
        shots.loc[shots["possession_id"] == pid, "shooter_x"] = f"{x_m:.4f}"
        shots.loc[shots["possession_id"] == pid, "shooter_y"] = f"{y_m:.4f}"
        updated += 1
        print(f"  + {pid}: ({u},{v}) -> ({x_m:.2f} m, {y_m:.2f} m) [{frame_path.name}]")

    shots.to_csv(SHOTS_CSV, index=False)
    print(f"\nDone. Updated={updated}, Skipped={skipped}, Failed={failed}")
    print(f"Saved -> {SHOTS_CSV.resolve()}")
    print("Next: python tools/normalize_and_export.py --input app/shots.csv --output app/features_shots.csv")

if __name__ == "__main__":
    main()
