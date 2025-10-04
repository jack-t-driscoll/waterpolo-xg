# tools/normalize_and_export.py
import argparse
import math
from pathlib import Path
import pandas as pd

# =========================
# Existing constants/schema
# =========================
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
    et = df.get("event_type", pd.Series(index=df.index, dtype=str)).astype(str).str.lower()

    shots     = df[et == "shot"].copy()
    turnovers = df[et == "turnover"].copy()
    draws     = df[et.isin(["ejection_drawn", "5m_drawn"])].copy()

    print("\n=== Audit ===")
    print(f"Rows total: {len(df)} | Shots: {len(shots)} | Turnovers: {len(turnovers)} | Ejections/5m drawn: {len(draws)}")

    def _miss(s: pd.Series) -> int:
        return int((s.isna() | (s.astype(str).str.strip() == "")).sum())

    # Shot-only fields: report missingness on SHOTS ONLY
    shot_fields = ["man_state", "attack_type", "goalie_present", "defender_count", "shot_type", "shot_result"]
    print("\nMissing (shots only):")
    for c in shot_fields:
        if c in shots.columns:
            print(f"  {c}: {_miss(shots[c])}")
        else:
            print(f"  {c}: (column not present)")

    # Turnover-specific fields: report on TURNOVERS ONLY
    to_fields = ["turnover_type", "turnover_player_number"]
    print("\nMissing (turnovers only):")
    if len(turnovers):
        for c in to_fields:
            if c in turnovers.columns:
                print(f"  {c}: {_miss(turnovers[c])}")
            else:
                print(f"  {c}: (column not present)")
    else:
        print("  (no turnover rows)")

    # Drawn events (ejection/5m): report on those ONLY
    draw_fields = ["drawn_by_player_number"]
    print("\nMissing (ejection_drawn/5m_drawn only):")
    if len(draws):
        for c in draw_fields:
            if c in draws.columns:
                print(f"  {c}: {_miss(draws[c])}")
            else:
                print(f"  {c}: (column not present)")
    else:
        print("  (no ejection_drawn or 5m_drawn rows)")

    # Geometry missingness (shots only) — from features frame you already build
    print("\nFeature missingness (shots only):")
    for c in ["distance_m", "angle_deg", "defender_count"]:
        if c in features.columns:
            print(f"  {c}: {int(features[c].isna().sum())} missing")
        else:
            print(f"  {c}: (column not present)")

    # Goalie distance only when goalie is present (shots only)
    if "goalie_distance_m" in features.columns and "empty_net" in features.columns:
        need_goalie_dist = ~features["empty_net"].fillna(False)
        missing_goalie_dist = features.loc[need_goalie_dist, "goalie_distance_m"].isna().sum()
        print(f"  goalie_distance_m (when goalie_present=true): {int(missing_goalie_dist)} missing")


# =========================
# New: app normalization & downloads (opt-in, non-destructive)
# =========================

def _mmss_to_seconds(s: str):
    if not isinstance(s, str) or ":" not in s:
        return None
    try:
        m, ss = s.split(":")
        return int(m) * 60 + float(ss)
    except Exception:
        return None

def build_shots_norm_full(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Create a normalized copy of the FULL dataset (not shots-only) for the app:
      - Adds aliases: x,y,outcome,handed
      - Adds angle_deg_signed (alias from your computed angle_deg when available)
      - Computes distance_m/angle_deg for rows that have shooter_x/y in [0,1]
      - Adds video_timecode from video_timestamp_mmss (non-destructive)
    Keeps ALL original columns.
    """
    df = df_all.copy()

    # Non-destructive aliases
    if "shooter_x" in df.columns: df["x"] = pd.to_numeric(df["shooter_x"], errors="coerce")
    if "shooter_y" in df.columns: df["y"] = pd.to_numeric(df["shooter_y"], errors="coerce")
    if "shot_result" in df.columns: df["outcome"] = df["shot_result"]
    if "shooter_handedness" in df.columns: df["handed"] = df["shooter_handedness"]

    # Geometry for any row with shooter_x/y present (uses same logic as compute_distance_angle)
    def _geom_row(r):
        try:
            x_norm = float(r["shooter_x"])
            y_norm = float(r["shooter_y"])
        except Exception:
            return pd.Series({"distance_m": pd.NA, "angle_deg": pd.NA})
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            return pd.Series({"distance_m": pd.NA, "angle_deg": pd.NA})
        dx = (x_norm - 0.5) * POOL_WIDTH_M
        dy = y_norm * FRONTCOURT_LENGTH_M
        distance = math.hypot(dx, dy)
        base_angle_rad = math.atan2(abs(dx), dy) if dy != 0 else (math.pi / 2)
        base_angle_deg = math.degrees(base_angle_rad)
        angle = -base_angle_deg if dx < 0 else base_angle_deg
        return pd.Series({"distance_m": round(distance, 3), "angle_deg": round(angle, 2)})

    geom = df.apply(_geom_row, axis=1)
    for col in ["distance_m", "angle_deg"]:
        if col not in df.columns:
            df[col] = pd.NA
        df.loc[geom[col].notna(), col] = geom.loc[geom[col].notna(), col]

    # angle_deg_signed for app (alias; keep your angle_deg)
    if "angle_deg" in df.columns:
        df["angle_deg_signed"] = pd.to_numeric(df["angle_deg"], errors="coerce")

    # video_timecode from mm:ss (non-destructive)
    if "video_timestamp_mmss" in df.columns:
        if "video_timecode" not in df.columns:
            df["video_timecode"] = pd.NA
        vt = df["video_timestamp_mmss"].apply(_mmss_to_seconds)
        df.loc[df["video_timecode"].isna() & pd.notna(vt), "video_timecode"] = vt

    # If you used video_file as your ID, mirror to source_video_id (non-destructive)
    if "video_file" in df.columns and "source_video_id" not in df.columns:
        df["source_video_id"] = df["video_file"]

    return df

def export_downloads(df_norm: pd.DataFrame, out_dir="app/reports/downloads"):
    """
    Write coach-friendly CSVs: overall, by_angle, by_distance, by_shooter
    Works even if some derived columns are missing (skips gracefully).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"shots":[len(df_norm[df_norm["event_type"]=="shot"])]}).to_csv(out/"overall.csv", index=False)

    if "angle_deg_signed" in df_norm.columns:
        bins = pd.cut(df_norm.loc[df_norm["event_type"]=="shot","angle_deg_signed"],
                      bins=[-90,-60,-30,0,30,60,90], include_lowest=True)
        by_angle = bins.value_counts(dropna=False).rename_axis("bin").reset_index(name="shots").sort_values("bin")
        by_angle.to_csv(out/"by_angle.csv", index=False)

    if "distance_m" in df_norm.columns:
        bins = pd.cut(df_norm.loc[df_norm["event_type"]=="shot","distance_m"],
                      bins=[0,3,6,9,12,15,21], include_lowest=True)
        by_dist = bins.value_counts(dropna=False).rename_axis("bin").reset_index(name="shots").sort_values("bin")
        by_dist.to_csv(out/"by_distance.csv", index=False)

    shooter_key = "player_number" if "player_number" in df_norm.columns else None
    if shooter_key:
        by_shooter = (
            df_norm[df_norm["event_type"]=="shot"]
            .groupby(shooter_key, dropna=False).size().rename("shots").reset_index()
        )
        by_shooter.to_csv(out/"by_shooter.csv", index=False)

# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Normalize and export modeling features from shots.csv")
    ap.add_argument("--input", default="shots.csv", help="Path to shots.csv")
    ap.add_argument("--output", default="features_shots.csv", help="Output features CSV (shots-only features)")
    ap.add_argument("--corrections", default=None, help="Optional corrections CSV (possession_id,field,value)")

    # New, optional flags (all backward-compatible; do nothing unless provided)
    ap.add_argument("--write-shots-norm", default=None,
                    help="If set, write a normalized FULL dataset CSV for the app (e.g., app/shots_norm.csv)")
    ap.add_argument("--export-downloads", action="store_true",
                    help="If set, write coach downloads CSVs under app/reports/downloads/ using the normalized dataset")

    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    df = load_csv(in_path)
    if args.corrections:
        df = apply_corrections(df, Path(args.corrections))

    # Build features (shots-only) — existing behavior
    features = build_features(df)
    features.to_csv(out_path, index=False)

    audit_report(df, features)
    print(f"\n\u2713 Wrote {len(features)} shot rows to {out_path.resolve()}")

    # New, optional: write normalized full dataset for the app
    if args.write_shots_norm:
        df_norm = build_shots_norm_full(df)
        norm_path = Path(args.write_shots_norm)
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        df_norm.to_csv(norm_path, index=False)
        print(f"\u2713 Wrote normalized full dataset to {norm_path.resolve()}")

        # Optional: export downloads based on normalized data
        if args.export_downloads:
            export_downloads(df_norm)
            print("\u2713 Wrote coach downloads to app/reports/downloads/")

if __name__ == "__main__":
    main()
