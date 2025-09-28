# tools/local_calibration_report.py
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Check local calibration by distance/angle bins.")
    ap.add_argument("--shots_features", default="app/features_shots.csv")
    ap.add_argument("--shots_preds", default="app/reports/xg_by_shot_all_calibrated_hgb.csv")
    ap.add_argument("--out_dir", default="app/reports")
    args = ap.parse_args()

    feats = pd.read_csv(args.shots_features, dtype=str)
    preds = pd.read_csv(args.shots_preds, dtype=str)

    for df in (feats, preds):
        for c in ["possession_id","game_id"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

    merged = preds.merge(feats, on=["possession_id","game_id"], how="left", validate="m:1")

    merged["xg"] = pd.to_numeric(merged["xg"], errors="coerce")
    merged["distance_m"] = pd.to_numeric(merged.get("distance_m"), errors="coerce")
    merged["angle_deg"] = pd.to_numeric(merged.get("angle_deg"), errors="coerce")
    merged["goal"] = (merged["shot_result"].str.lower() == "goal").astype(int)

    # Bins
    dist_bins = [0,5,10,15,20,30]
    ang_bins = [0,15,30,45,90]
    merged["distance_bin"] = pd.cut(merged["distance_m"], bins=dist_bins, right=False)
    merged["angle_bin"] = pd.cut(merged["angle_deg"].abs(), bins=ang_bins, right=False)

    def agg(df, key):
        g = (
            df.groupby(key, observed=True)  # <- silence future warning & use future default
              .agg(shots=("xg","size"), goals=("goal","sum"), xg_sum=("xg","sum"))
              .reset_index()
        )
        g["goal_rate"] = g["goals"]/g["shots"]
        g["xg_rate"] = g["xg_sum"]/g["shots"]
        g["residual"] = g["goals"] - g["xg_sum"]
        return g

    by_dist = agg(merged, "distance_bin")
    by_angle = agg(merged, "angle_bin")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    by_dist.to_csv(out_dir/"local_calibration_by_distance.csv", index=False)
    by_angle.to_csv(out_dir/"local_calibration_by_angle.csv", index=False)

    print("✓ Wrote:")
    print(" ", (out_dir/"local_calibration_by_distance.csv").resolve())
    print(" ", (out_dir/"local_calibration_by_angle.csv").resolve())

if __name__ == "__main__":
    main()
