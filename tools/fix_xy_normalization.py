# tools/fix_xy_normalization.py
# Convert any meter-based x/y rows in app/shots.csv to normalized fractions.
# - Assumes "shooter_x","shooter_y" either already in [0,1] or given as meters:
#     x_m in [-10, +10]  -> x_n = (x_m + 10)/20
#     y_m in [0, 15]     -> y_n = y_m / 15
# Writes a backup and fixes in place.

import pandas as pd
from pathlib import Path

IN = Path("app/shots.csv")
BAK = Path("app/shots.backup_before_fix.csv")

def looks_like_fraction(s):
    try:
        v = float(s)
        return 0.0 <= v <= 1.0
    except:
        return False

def looks_like_meter_x(s):
    try:
        v = float(s)
        return -10.5 <= v <= 10.5
    except:
        return False

def looks_like_meter_y(s):
    try:
        v = float(s)
        return -0.5 <= v <= 15.5
    except:
        return False

def main():
    if not IN.exists():
        print(f"ERROR: {IN} not found.")
        return
    df = pd.read_csv(IN, dtype=str)
    if "shooter_x" not in df.columns or "shooter_y" not in df.columns:
        print("ERROR: shots.csv missing shooter_x/shooter_y.")
        return

    df.to_csv(BAK, index=False)
    print(f"Backup written → {BAK}")

    fixed = 0
    for idx, row in df.iterrows():
        sx, sy = row.get("shooter_x", ""), row.get("shooter_y", "")
        if sx == "" and sy == "":
            continue

        # Decide if meters → convert to fractions
        convert_x = (not looks_like_fraction(sx)) and looks_like_meter_x(sx)
        convert_y = (not looks_like_fraction(sy)) and looks_like_meter_y(sy)

        if convert_x:
            x_m = float(sx)
            x_n = (x_m + 10.0) / 20.0
            df.at[idx, "shooter_x"] = f"{x_n:.3f}"
            fixed += 1

        if convert_y:
            y_m = float(sy)
            y_n = (y_m / 15.0)
            df.at[idx, "shooter_y"] = f"{y_n:.3f}"
            fixed += 1

    df.to_csv(IN, index=False)
    print(f"✓ Fixed {fixed} field(s). Saved → {IN}")

if __name__ == "__main__":
    main()
