# tools/audit_pipeline.py
import argparse
from pathlib import Path
import pandas as pd

EXPECTED_FEATURES = {
    "our_team_level",
    "opponent_team_level",
    "distance_m",
    "angle_deg",
    "defender_count",
    "goalie_distance_m",
    "goalie_lateral",
    "attack_type",
    "shot_type",
    "shooter_handedness",
    "player_number",
    "possession_passes",
    "shot_result",
}

def main():
    ap = argparse.ArgumentParser(description="Audit pipeline outputs for missing or dropped features.")
    ap.add_argument("--shots", default="app/shots.csv", help="Path to shots.csv")
    ap.add_argument("--features", default="app/features_shots.csv", help="Path to features_shots.csv")
    ap.add_argument("--reports", default="app/reports", help="Directory containing model reports")
    args = ap.parse_args()

    shots_path = Path(args.shots)
    features_path = Path(args.features)
    reports_dir = Path(args.reports)

    # --- Check shots.csv exists
    if not shots_path.exists():
        print(f"! shots.csv not found at {shots_path}")
        return

    # --- Check features_shots.csv
    if not features_path.exists():
        print(f"! features_shots.csv not found at {features_path}")
    else:
        fs = pd.read_csv(features_path, nrows=0)  # just headers
        fs_cols = set([c.strip().lower() for c in fs.columns])
        missing = [f for f in EXPECTED_FEATURES if f.lower() not in fs_cols]
        if missing:
            print("! Missing expected features in features_shots.csv:", missing)
        else:
            print("✓ All expected features present in features_shots.csv")
        print("Columns found:", sorted(fs_cols))

    # --- Check reports dir
    if not reports_dir.exists():
        print(f"! reports dir not found at {reports_dir}")
        return

    expected_reports = [
        "metrics_validation.csv",
        "xg_by_game_all_calibrated_hgb.csv",
        "xg_by_shot_all_calibrated_hgb.csv",
    ]
    for fname in expected_reports:
        path = reports_dir / fname
        if path.exists():
            print(f"✓ Found {fname}")
        else:
            print(f"! Missing {fname}")

if __name__ == "__main__":
    main()
