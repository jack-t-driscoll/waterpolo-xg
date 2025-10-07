# tools/homography_calibrator.py
# -*- coding: utf-8 -*-
"""
Homography calibrator (OpenCV) with reliable fullscreen fit, zoom, letterbox, and dialog prompts.
Controls:
  • Left-click: add point → dialog asks for "x_m, y_m" (or "x_m y_m")
  • u: undo last point
  • + / - : zoom in / out
  • 0: reset zoom to fit screen
  • f: toggle fullscreen
  • q: finish & save JSON to app/reports/homography/<video_file>.json
"""

from __future__ import annotations
from pathlib import Path
import sys, json, argparse, ctypes
import cv2, numpy as np, pandas as pd

# Tkinter dialogs (avoid console focus issues)
try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
    TK_OK = True
except Exception:
    TK_OK = False

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
FRAMES_DIR = APP / "reports" / "frames"
MANIFEST = FRAMES_DIR / "frames_manifest.csv"
HOMO_DIR = APP / "reports" / "homography"
HOMO_DIR.mkdir(parents=True, exist_ok=True)

# ---------- screen helpers ----------
def get_screen_size():
    """Physical screen size (px), DPI-aware."""
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080

def fit_scale(w, h, max_w, max_h, target_h=None):
    """
    Scale to fit into max_w x max_h.
    If target_h is provided and image is small, allow upscaling up to target_h (but not beyond screen).
    """
    if w <= 0 or h <= 0:
        return 1.0
    s_fit = min(max_w / float(w), max_h / float(h))
    if target_h is not None:
        s_up = max(1.0, target_h / float(h))
        return min(s_fit, s_up)
    return s_fit

# ---------- data ----------
def pick_t0_for_video(video_file: str) -> str | None:
    if not MANIFEST.exists():
        print(f"ERROR: Missing manifest: {MANIFEST}")
        return None
    man = pd.read_csv(MANIFEST, dtype=str)
    row = man[(man["video_file"].astype(str) == video_file) & (man["context"] == "t0")]
    if row.empty:
        print(f"ERROR: No t0 frame found for video_file={video_file}")
        return None
    return str(row.iloc[0]["frame_path"])

# ---------- UI ----------
class CalibUI:
    def __init__(self, img_path: str, start_fullscreen=True):
        img = cv2.imread(img_path)
        if img is None:
            raise SystemExit(f"ERROR: could not read image: {img_path}")
        self.img_path = img_path
        self.img_orig = img
        self.H0, self.W0 = img.shape[:2]
        sw, sh = get_screen_size()
        self.max_w = int(sw * 0.98)
        self.max_h = int(sh * 0.98)

        # window
        self.win = "Homography Calibrator (click; + - zoom; 0 reset; u undo; f fullscreen; q save)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        self.fullscreen = False
        if start_fullscreen:
            self.toggle_fullscreen(force=True)

        # scale & letterbox offsets
        self.scale = fit_scale(self.W0, self.H0, self.max_w, self.max_h,
                               target_h=min(1080, int(self.max_h*0.95)))
        self.offset_x = 0
        self.offset_y = 0

        # collected points
        self.pts_px: list[list[float]] = []  # in original pixels
        self.pts_m:  list[list[float]] = []  # in meters

        # dialogs root
        self.tk_root = None

        cv2.setMouseCallback(self.win, self.on_mouse)
        self.render()

    # --- fullscreen toggle ---
    def toggle_fullscreen(self, force=None):
        new_state = (not self.fullscreen) if force is None else force
        self.fullscreen = new_state
        prop = cv2.WINDOW_FULLSCREEN if new_state else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, prop)

    # --- Tk root ---
    def ensure_tk(self):
        if not TK_OK:
            return None
        if self.tk_root is None:
            self.tk_root = tk.Tk()
            self.tk_root.withdraw()
        return self.tk_root

    # --- canvas composition ---
    def current_canvas(self):
        disp_w = int(round(self.W0 * self.scale))
        disp_h = int(round(self.H0 * self.scale))
        canvas_w = min(self.max_w, max(disp_w, 320))
        canvas_h = min(self.max_h, max(disp_h, 240))
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        self.offset_x = (canvas_w - disp_w) // 2 if canvas_w > disp_w else 0
        self.offset_y = (canvas_h - disp_h) // 2 if canvas_h > disp_h else 0

        disp = cv2.resize(self.img_orig, (disp_w, disp_h),
                          interpolation=cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR)
        canvas[self.offset_y:self.offset_y+disp_h, self.offset_x:self.offset_x+disp_w] = disp
        return canvas

    def draw_points(self, canvas):
        for i, (x_px, y_px) in enumerate(self.pts_px):
            x_d = int(round(x_px * self.scale)) + self.offset_x
            y_d = int(round(y_px * self.scale)) + self.offset_y
            cv2.circle(canvas, (x_d, y_d), 7, (0,255,0), -1)
            cv2.putText(canvas, f"{i+1}", (x_d+8, y_d-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,255,50), 2, cv2.LINE_AA)

    def render(self):
        canvas = self.current_canvas()
        self.draw_points(canvas)
        cv2.imshow(self.win, canvas)
        if not self.fullscreen:
            ch, cw = canvas.shape[:2]
            cv2.resizeWindow(self.win, cw, ch)

    # --- prompt for meters ---
    def prompt_xy(self, idx: int):
        if not TK_OK:
            return None
        root = self.ensure_tk()
        msg = (f"Point #{idx}\n\n"
               "Enter pool coordinates in meters:\n"
               "  x_m ∈ [-10..+10]  (0 = goal center; left < 0, right > 0)\n"
               "  y_m ∈ [0..15]     (0 = goal line; 15 = frontcourt limit)\n\n"
               "Formats:  '-3.5, 8'  or  '-3.5 8'")
        while True:
            ans = simpledialog.askstring("Pool coordinates", msg, parent=root)
            if ans is None:
                return None
            t = ans.strip().replace(",", " ").split()
            if len(t) != 2:
                messagebox.showerror("Invalid", "Please enter two numbers (x y) or (x, y).")
                continue
            try:
                x_m, y_m = float(t[0]), float(t[1])
            except Exception:
                messagebox.showerror("Invalid", "Could not parse numbers.")
                continue
            if not (-10.0 <= x_m <= 10.0 and 0.0 <= y_m <= 15.0):
                ok = messagebox.askyesno("Out of range",
                                         f"Received (x_m={x_m}, y_m={y_m}). Continue anyway?")
                if not ok:
                    continue
            return [x_m, y_m]

    # --- mouse handler ---
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # map display → original px
            xo = (x - self.offset_x) / max(self.scale, 1e-9)
            yo = (y - self.offset_y) / max(self.scale, 1e-9)
            if 0 <= xo < self.W0 and 0 <= yo < self.H0:
                idx = len(self.pts_px) + 1
                self.pts_px.append([xo, yo])
                self.render()

                xym = self.prompt_xy(idx) if TK_OK else None
                if xym is None:
                    # cancelled; remove point
                    self.pts_px.pop()
                    self.render()
                else:
                    self.pts_m.append(xym)

    # --- zoom controls ---
    def zoom(self, factor: float):
        old = self.scale
        self.scale = float(np.clip(self.scale * factor, 0.2, 6.0))
        if abs(self.scale - old) > 1e-6:
            self.render()

    def reset_zoom(self):
        self.scale = fit_scale(self.W0, self.H0, self.max_w, self.max_h,
                               target_h=min(1080, int(self.max_h*0.95)))
        self.render()

# ---------- save ----------
def compute_and_save(video_file: str, img_path: str, px_pts, m_pts, img_shape):
    if len(px_pts) != len(m_pts) or len(px_pts) < 4:
        print("ERROR: Need >= 4 matching points.")
        return False
    px = np.array(px_pts, dtype=float)
    mt = np.array(m_pts, dtype=float)
    H, mask = cv2.findHomography(px, mt, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        print("ERROR: cv2.findHomography failed.")
        return False
    out = {
        "video_file": video_file,
        "image_path": str(Path(img_path).as_posix()),
        "image_shape": {"h": int(img_shape[0]), "w": int(img_shape[1])},
        "H": H.tolist(),
        "points_px": px.tolist(),
        "points_m": mt.tolist(),
        "notes": "Pixel→meter homography. x_m in [-10..+10], y_m in [0..15]."
    }
    out_path = HOMO_DIR / f"{video_file}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"✓ Saved homography → {out_path.resolve()}")
    return True

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Homography calibrator (fit, zoom, letterbox, dialogs)")
    ap.add_argument("--video_file", help="e.g., IMG_6154.MOV (must exist in frames_manifest with t0)")
    ap.add_argument("--frame_path", help="Optional explicit path to a t0 image (overrides --video_file)")
    args = ap.parse_args()

    if args.frame_path:
        img_path = args.frame_path
        # Best effort for JSON name; doesn’t affect math
        video_file = Path(img_path).stem.split("_")[0] + ".MOV"
    elif args.video_file:
        video_file = args.video_file
        img_path = pick_t0_for_video(video_file)
        if img_path is None:
            sys.exit(1)
    else:
        print("Provide either --video_file or --frame_path")
        sys.exit(1)

    ui = CalibUI(img_path, start_fullscreen=True)

    print("\nInstructions:")
    print("  • Click landmark → dialog appears → enter 'x_m, y_m' or 'x_m y_m' → OK.")
    print("  • Keys: + / - zoom, 0 reset, u undo, f fullscreen, q finish.")
    print("  • Need at least 4 points; 6–10 recommended.\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            if ui.pts_px:
                ui.pts_px.pop()
                if ui.pts_m: ui.pts_m.pop()
                ui.render()
        elif key in (ord('+'), ord('=')):
            ui.zoom(1.25)
        elif key in (ord('-'), ord('_')):
            ui.zoom(0.8)
        elif key == ord('0'):
            ui.reset_zoom()
        elif key == ord('f'):
            ui.toggle_fullscreen()

    cv2.destroyAllWindows()

    if len(ui.pts_px) < 4:
        print("Not enough points. No homography saved.")
        sys.exit(1)

    ok = compute_and_save(video_file, ui.img_path, ui.pts_px, ui.pts_m, ui.img_orig.shape)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
