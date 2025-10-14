# tools/repair_geometry_from_xy.py
from pathlib import Path
import math
import pandas as pd
import numpy as np

IN = Path("app/features_shots.csv")
OUT = IN  # in-place

def compute_distance_angle_from_xy(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    d = df.copy()

    # Coerce to numeric (safe even if already numeric)
    for c in ["shooter_x", "shooter_y", "distance_m", "angle_deg"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan

    # Identify rows that can be repaired (have XY but missing distance/angle)
    need = d["shooter_x"].notna() & d["shooter_y"].notna() & (
        d["distance_m"].isna() | d["angle_deg"].isna()
    )

    # Compute:
    # distance_m = sqrt( x^2 + y^2 )
    # angle_deg = atan2( x , y ) in degrees (x=lateral, y=toward goal)
    x = d.loc[need, "shooter_x"].astype(float)
    y = d.loc[need, "shooter_y"].astype(float)

    d.loc[need, "distance_m"] = (x.pow(2) + y.pow(2)).pow(0.5)
    d.loc[need, "angle_deg"] = np.degrees(np.arctan2(x, y))

    fixed = int(need.sum())
    print(f"Repaired rows (computed distance_m/angle_deg from XY): {fixed}")
    return d

def main():
    df = pd.read_csv(IN, dtype=str)
    before_missing_dist = df["distance_m"].isna().sum() if "distance_m" in df.columns else "NA"
    before_missing_ang  = df["angle_deg"].isna().sum() if "angle_deg" in df.columns else "NA"
    print(f"Before → missing distance_m: {before_missing_dist} | angle_deg: {before_missing_ang}")

    df2 = compute_distance_angle_from_xy(df)
    df2.to_csv(OUT, index=False)

    after_missing_dist = df2["distance_m"].isna().sum() if "distance_m" in df2.columns else "NA"
    after_missing_ang  = df2["angle_deg"].isna().sum() if "angle_deg" in df2.columns else "NA"
    print(f"After  → missing distance_m: {after_missing_dist} | angle_deg: {after_missing_ang}")
    print(f"✓ Wrote {OUT}")

if __name__ == "__main__":
    main()
