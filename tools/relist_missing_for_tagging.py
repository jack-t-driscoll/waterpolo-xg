import pandas as pd
from pathlib import Path
import os

APP = Path("app")
SHOTS = APP / "shots.csv"
HOMO_DIR = APP / "homography"
OUTDIR = APP / "reports" / "diagnostics"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "missing_geometry.csv"

df = pd.read_csv(SHOTS, dtype=str)
for c in ["shooter_x","shooter_y"]:
    if c not in df.columns:
        df[c] = None
miss = df.copy()
# Only shots with missing XY
miss = miss[(miss.get("event_type","shot").str.lower()=="shot") &
            (miss["shooter_x"].isna() | (miss["shooter_x"].astype(str)=="") |
             miss["shooter_y"].isna() | (miss["shooter_y"].astype(str)==""))].copy()

def has_h_json(vf):
    if not isinstance(vf, str) or vf.strip()=="":
        return False
    cand = HOMO_DIR / (Path(vf).stem + ".json")
    return cand.exists()

miss["reason"] = "missing_xy_tag"
miss["has_homography_json"] = miss["video_file"].apply(has_h_json)
miss.loc[miss["has_homography_json"]==False, "reason"] = "missing_xy_tag|missing_homography_json"

keep_cols = ["possession_id","event_type","video_file","video_timestamp_mmss",
             "shooter_x","shooter_y","reason"]
missing_now = miss[keep_cols].sort_values(["video_file","possession_id"])
missing_now.to_csv(OUT, index=False)

print(f"Shots total: {len(df[df.get('event_type','shot').str.lower()=='shot'])}")
print("Missing XY after refresh:", len(missing_now))
print("Reasons:")
print(missing_now["reason"].value_counts(dropna=False).to_string())
print(f"\n✓ Wrote → {OUT.resolve()}")
