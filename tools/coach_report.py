# tools/coach_report.py
# Purpose: Produce coach-facing summaries from features + model outputs.
# Robust to: by_shot having xg/xg_cal/xg_raw; by_game being pre-aggregated or per-shot.
# Outputs: CSVs into --out_dir (shooters, overview, angle_distance, heatmap-friendly tables)

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_csv_safe(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")
    return pd.read_csv(p, dtype=str)

def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def pick_xg_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a numeric 'xg' column:
      - Prefer existing 'xg'
      - else use 'xg_cal'
      - else use 'xg_raw'
    Raises if none found.
    """
    for cand in ["xg", "xg_cal", "xg_raw"]:
        if cand in df.columns:
            df["xg"] = to_float(df[cand])
            return df
    raise SystemExit(f"'by_shot' must contain one of ['xg','xg_cal','xg_raw']; columns={list(df.columns)}")

def coerce_boolish(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    m = {True:1, False:0, "true":1, "false":0, "True":1, "False":0, 1:1, 0:0}
    return s.map(m).fillna(0).astype(int)

def maybe_aggregate_by_game(by_game: pd.DataFrame, by_shot: pd.DataFrame) -> pd.DataFrame:
    """
    If by_game already has 'game_id','xg_sum','shots' we accept it.
    Otherwise, compute a per-game aggregate from by_shot.
    """
    cols = set(by_game.columns)
    if {"game_id", "xg_sum", "shots"}.issubset(cols):
        # Ensure numeric types
        by_game["xg_sum"] = to_float(by_game["xg_sum"])
        by_game["shots"] = pd.to_numeric(by_game["shots"], errors="coerce").fillna(0).astype(int)
        return by_game[["game_id","xg_sum","shots"]]

    # Build from by_shot instead
    if "game_id" not in by_shot.columns:
        raise SystemExit("Cannot aggregate by_game: 'by_shot' missing 'game_id'")
    agg = by_shot.groupby("game_id", dropna=False).agg(
        shots=("possession_id", "count"),
        xg_sum=("xg", "sum"),
    ).reset_index()
    return agg

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric geometry present for summaries
    for c in ["distance_m","angle_deg","shooter_x","shooter_y","defender_count","goalie_distance_m","possession_passes"]:
        if c in df.columns:
            df[c] = to_float(df[c])
    # Normalize shot_result labels a bit
    if "shot_result" in df.columns:
        df["shot_result"] = df["shot_result"].astype(str).str.strip().str.lower()
        df["is_goal"] = (df["shot_result"] == "goal").astype(int)
    else:
        df["is_goal"] = np.nan
    # Useful bins
    if "distance_m" in df.columns:
        df["distance_bin_model"] = pd.cut(df["distance_m"],
                                          bins=[-1,2,4,6,8,10,50],
                                          labels=["0-2","2-4","4-6","6-8","8-10","10+"],
                                          include_lowest=True)
    if "angle_deg" in df.columns:
        df["angle_bin"] = pd.cut(df["angle_deg"],
                                 bins=[-181,-60,-30,-10,10,30,60,181],
                                 labels=["<-60","-60..-30","-30..-10","-10..10","10..30","30..60",">60"],
                                 include_lowest=True)
    return df

def make_shooter_summary(df: pd.DataFrame, by_shot: pd.DataFrame) -> pd.DataFrame:
    # Join xg back to features on possession_id
    if "possession_id" not in df.columns or "possession_id" not in by_shot.columns:
        raise SystemExit("Both features and by_shot must have 'possession_id'.")
    M = df.merge(by_shot[["possession_id","xg","game_id"]], on="possession_id", how="left")
    # Shooter identity (prefer team-aware if present)
    shooter_key = None
    for cand in ["shooter_name_team","shooter_name","player_number"]:
        if cand in M.columns:
            shooter_key = cand
            break
    if shooter_key is None:
        # Fall back so we still produce something
        shooter_key = "player_number"
        M[shooter_key] = df.get("player_number", "Unknown")

    g = M.groupby([shooter_key], dropna=False).agg(
        shots=("possession_id","count"),
        goals=("is_goal","sum"),
        xg_sum=("xg","sum"),
        dist_med=("distance_m","median"),
        ang_med=("angle_deg","median"),
    ).reset_index()
    g["sh_pct"] = (g["goals"] / g["shots"]).round(3)
    g["xg_per_shot"] = (g["xg_sum"] / g["shots"]).round(3)
    # Order
    g = g.sort_values(["shots","xg_per_shot"], ascending=[False,False])
    return g.rename(columns={shooter_key:"shooter"})

def make_overview(by_game: pd.DataFrame) -> pd.DataFrame:
    bg = by_game.copy()
    bg["xg_per_shot"] = (bg["xg_sum"] / bg["shots"]).replace([np.inf, -np.inf], np.nan).round(3)
    return bg.sort_values("game_id")

def make_angle_distance(df: pd.DataFrame, by_shot: pd.DataFrame, min_bin_size: int) -> pd.DataFrame:
    M = df.merge(by_shot[["possession_id","xg","game_id"]], on="possession_id", how="left")
    # Choose signed/unsigned angle groupings; stick with signed bins if available
    angle_col = "angle_bin" if "angle_bin" in M.columns else None
    if angle_col is None:
        # fallback rough binning from numeric angle if present
        if "angle_deg" in M.columns:
            M["angle_bin"] = pd.cut(M["angle_deg"], bins=[-181,-60,-30,-10,10,30,60,181],
                                    labels=["<-60","-60..-30","-30..-10","-10..10","10..30","30..60",">60"],
                                    include_lowest=True)
            angle_col = "angle_bin"

    dist_col = "distance_bin_model" if "distance_bin_model" in M.columns else None
    if dist_col is None and "distance_m" in M.columns:
        M["distance_bin_model"] = pd.cut(M["distance_m"],
                                         bins=[-1,2,4,6,8,10,50],
                                         labels=["0-2","2-4","4-6","6-8","8-10","10+"],
                                         include_lowest=True)
        dist_col = "distance_bin_model"

    keys = []
    if angle_col: keys.append(angle_col)
    if dist_col: keys.append(dist_col)
    if not keys:
        # Nothing to aggregate on; return empty
        return pd.DataFrame(columns=["bin_key","shots","goals","xg_sum","goal_rate","xg_per_shot"])

    g = M.groupby(keys, dropna=False).agg(
        shots=("possession_id","count"),
        goals=("is_goal","sum"),
        xg_sum=("xg","sum"),
    ).reset_index()
    g["goal_rate"] = (g["goals"] / g["shots"]).round(3)
    g["xg_per_shot"] = (g["xg_sum"] / g["shots"]).round(3)

    # Filter small bins for stability
    g = g[g["shots"] >= int(min_bin_size)].copy()

    # Human-friendly key
    g["bin_key"] = g.apply(lambda r: "|".join(str(r[c]) for c in keys), axis=1)
    cols = ["bin_key","shots","goals","xg_sum","goal_rate","xg_per_shot"] + keys
    return g[cols].sort_values(["shots","xg_per_shot"], ascending=[False, False])

def main():
    ap = argparse.ArgumentParser(
        description="Coach report generator (schema-aware: xg/xg_cal/xg_raw; auto-aggregates by_game if needed)"
    )
    ap.add_argument("--features", required=True, help="features_shots.csv")
    ap.add_argument("--by_shot", required=True, help="xg_by_shot_*.csv (per-shot predictions; column xg/xg_cal/xg_raw)")
    ap.add_argument("--by_game", required=True, help="xg_by_game_*.csv (pre-agg or per-shot; tool will adapt)")
    ap.add_argument("--out_dir", required=True, help="Output directory for coach CSVs")
    ap.add_argument("--min_bin_size", type=int, default=5, help="Minimum bin size for angle-distance table")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))

    # Load
    features = read_csv_safe(Path(args.features))
    by_shot  = read_csv_safe(Path(args.by_shot))
    by_game  = read_csv_safe(Path(args.by_game))

    # Normalize schemas
    by_shot = pick_xg_column(by_shot)
    by_game = maybe_aggregate_by_game(by_game, by_shot)
    features = enrich_features(features)

    # Build outputs
    shooters = make_shooter_summary(features, by_shot)
    overview = make_overview(by_game)
    angdist  = make_angle_distance(features, by_shot, min_bin_size=args.min_bin_size)

    # Write
    shooters.to_csv(out_dir / "shooters.csv", index=False)
    overview.to_csv(out_dir / "overview.csv", index=False)
    angdist.to_csv(out_dir / "angle_distance.csv", index=False)

    print(f"✓ Wrote coach summaries → {out_dir.resolve()}")

if __name__ == "__main__":
    main()
