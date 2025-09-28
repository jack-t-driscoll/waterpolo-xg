# tools/normalize_and_export.py
import argparse
import math
from pathlib import Path
import pandas as pd

POOL_WIDTH_M = 20.0        # x spans -10..+10
FRONTCOURT_LENGTH_M = 15.0 # y spans 0..15

EXPECTED_COLS = [
    "possession_id",
    "game_id",
    "our_team_name",
    "opponent_team_name",
    "our_team_level",
    "opponent_team_level",
    "period",
    "time_remaining",
    "man_state",                 # e.g., 6v6, 6v5, 6v4, 7v6 (pulled goalie/empty net), 6v7 (opponent pulled...), 5v5, other
    "event_type",                # shot | turnover | ejection_drawn | 5m_drawn
    "turnover_type",
    "turnover_player_number",
    "drawn_by_player_number",
    "shooter_x",                 # normalized 0..1 (string in CSV)
    "shooter_y",                 # normalized 0..1 (string in CSV)
    "defender_count",
    "goalie_present",            # "true"/"false" on shots; blank otherwise
    "goalie_distance_m",
    "goalie_lateral",
    "possession_passes",
    "attack_type",               # perimeter, set, drive, man-up, counter, 5m penalty, broken play, other
    "shot_type",
    "shot_result",
    "shooter_handedness",
    "player_number",
    "video_file",
    "video_timestamp_mmss",
]

# Our numerical-advantage states (exclusions). Pulled-goalie scenarios are not counted as "man-up".
MAN_UP_STATES_OUR = {"6v5", "6v4"}

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # Ensure expected columns exist
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    # Keep stable order; preserve any extras at end
    extras = [c for c in df.columns if c not in EXPECTED_COLS]
    df = df[EXPECTED_COLS + extras]
    return df

def apply_corrections(df: pd.DataFrame, corr_path: Path) -> pd.DataFrame:
    """
    corrections.csv format:
      possession_id,field,value
      2025GAME1_P0017,event_type,turnover
      2025GAME1_P0017,turnover_type,steal/interception
    """
    corr = pd.read_csv(corr_path, dtype=str)
    needed = {"possession_id", "field", "value"}
    if not needed.issubset(set(corr.columns)):
        raise ValueError(f"Corrections file must have columns {needed}")
    for pid, rows in corr.groupby("possession_id"):
        mask = (df["possession_id"].astype(str) == str(pid))
        if not mask.any():
            continue
        for _, r in rows.iterrows():
            field = r["field"]
            val = r["value"]
            if field not in df.columns:
                df[field] = pd.NA
            df.loc[mask, field] = val
    return df

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Trim strings
    for c in ["attack_type", "goalie_present", "event_type", "man_state"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Normalize attack_type legacy label
    df.loc[df["attack_type"].str.lower() == "man-up (6v5)", "attack_type"] = "man-up"

    # If a shot row has blank goalie_present, default to true
    is_shot = df["event_type"] == "shot"
    blank_goalie = df["goalie_present"].isna() | (df["goalie_present"].astype(str).str.strip() == "")
    df.loc[is_shot & blank_goalie, "goalie_present"] = "true"

    # Standardize goalie_present casing
    df.loc[df["goalie_present"].str.lower() == "true", "goalie_present"] = "true"
    df.loc[df["goalie_present"].str.lower() == "false", "goalie_present"] = "false"

    return df

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def compute_distance_angle(row):
    try:
        x_norm = to_float(row.get("shooter_x", ""))
        y_norm = to_float(row.get("shooter_y", ""))
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            return pd.Series({"distance_m": float("nan"), "angle_deg": float("nan")})
        dx = (x_norm - 0.5) * POOL_WIDTH_M
        dy = y_norm * FRONTCOURT_LENGTH_M
        distance = math.hypot(dx, dy)
        base_angle_rad = math.atan2(abs(dx), dy) if dy != 0 else (math.pi / 2)
        base_angle_deg = math.degrees(base_angle_rad)
        angle = -base_angle_deg if dx < 0 else base_angle_deg
        return pd.Series({"distance_m": round(distance, 3), "angle_deg": round(angle, 2)})
    except Exception:
        return pd.Series({"distance_m": float("nan"), "angle_deg": float("nan")})

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_labels(df)

    # Shots only for xG features
    shots = df[df["event_type"] == "shot"].copy()

    # Derived flags
    shots["is_man_up"] = (
        shots["man_state"].isin(MAN_UP_STATES_OUR) |
        (shots["attack_type"].astype(str).str.lower() == "man-up")
    )
    shots["is_penalty_shot"] = (shots["shot_type"].astype(str).str.lower() == "5m penalty")
    shots["empty_net"] = (shots["goalie_present"].astype(str).str.lower() == "false")

    # Numeric casts (safe; leave NaN if not numeric)
    for col in ["defender_count", "goalie_distance_m", "player_number", "possession_passes"]:
        shots[col] = pd.to_numeric(shots[col], errors="coerce")

    # Distance/angle
    geom = shots.apply(compute_distance_angle, axis=1)
    shots = pd.concat([shots, geom], axis=1)

    # Export columns (feel free to add/remove later)
    export_cols = [
        "possession_id",
        "game_id",
        "period",
        "time_remaining",
        "man_state",
        "is_man_up",
        "empty_net",
        "shooter_x",
        "shooter_y",
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
        "video_file",
        "video_timestamp_mmss",
    ]
    for c in export_cols:
        if c not in shots.columns:
            shots[c] = pd.NA

    # Sort for readability
    shots = shots[export_cols].sort_values(["game_id", "possession_id"], kind="stable")

    return shots

def audit_report(df: pd.DataFrame, features: pd.DataFrame):
    print("\n=== Audit ===")
    print(
        f"Rows total: {len(df)} | Shots: {int((df['event_type']=='shot').sum())} | "
        f"Turnovers: {int((df['event_type']=='turnover').sum())} | "
        f"Ejections drawn: {int((df['event_type']=='ejection_drawn').sum())} | "
        f"5m drawn: {int((df['event_type']=='5m_drawn').sum())}"
    )

    # Missing across ALL events (non-shot rows will naturally be blank for shot-only fields)
    miss_cols = ["man_state", "attack_type", "goalie_present", "defender_count", "shot_type", "shot_result"]
    for c in miss_cols:
        m = df[c].isna() | (df[c].astype(str).str.strip() == "")
        print(f"Missing {c}: {int(m.sum())}")

    print("\nFeature missingness (shots only):")
    print(f"  distance_m: {int(features['distance_m'].isna().sum())} missing")
    print(f"  angle_deg: {int(features['angle_deg'].isna().sum())} missing")
    print(f"  defender_count: {int(features['defender_count'].isna().sum())} missing")

    # Only count goalie_distance_m where a goalie is present (i.e., not empty-net shots)
    need_goalie_dist = ~features["empty_net"].fillna(False)
    missing_goalie_dist = features.loc[need_goalie_dist, "goalie_distance_m"].isna().sum()
    print(f"  goalie_distance_m (when goalie_present=true): {int(missing_goalie_dist)} missing")

def main():
    ap = argparse.ArgumentParser(description="Normalize and export modeling features from shots.csv")
    ap.add_argument("--input", default="shots.csv", help="Path to shots.csv")
    ap.add_argument("--output", default="features_shots.csv", help="Output features CSV")
    ap.add_argument("--corrections", default=None, help="Optional corrections CSV (possession_id,field,value)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    df = load_csv(in_path)
    if args.corrections:
        df = apply_corrections(df, Path(args.corrections))

    features = build_features(df)
    features.to_csv(out_path, index=False)

    audit_report(df, features)
    print(f"\n✓ Wrote {len(features)} shot rows to {out_path.resolve()}")

if __name__ == "__main__":
    main()
