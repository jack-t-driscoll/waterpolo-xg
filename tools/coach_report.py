# tools/coach_report.py
# Coach-facing summaries from calibrated by-shot/by-game outputs.
# - Merges features to recover labels & context
# - Derives is_goal from features['shot_result'] == 'goal'
# - Adds low_data flags for small counts
# - Shooters grouped by our_team_name + player_number (adds shooter_label)
# - Angle bins are SIGNED (left negative, right positive), recomputed unconditionally

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def add_low_data_flag(df, count_col="n", threshold=10):
    df = df.copy()
    if count_col in df.columns:
        df["low_data"] = df[count_col] < threshold
    else:
        if "shots" in df.columns and len(df) == 1:
            df["low_data"] = df["shots"] < threshold
        else:
            df["low_data"] = False
    return df


def recompute_bins(df):
    """Recompute distance_bin and SIGNED angle_bin (overwrite any existing)."""
    out = df.copy()

    # Distance bins: [0-5, 5-8, 8-10, 10+]
    if "distance_m" in out.columns:
        out["distance_m"] = to_num(out["distance_m"])
        out["distance_bin"] = pd.cut(
            out["distance_m"],
            bins=[0, 5, 8, 10, 30],
            labels=["0-5", "5-8", "8-10", "10+"],
            right=False,
        )
    else:
        out["distance_bin"] = pd.NA

    # SIGNED angle bins: [-90,-45), [-45,-30), [-30,-15), [-15,15), [15,30), [30,45), [45,90]
    if "angle_deg" in out.columns:
        ang = to_num(out["angle_deg"])  # keep sign
        bins = [-90, -45, -30, -15, 15, 30, 45, 90]
        labels = ["-90:-45", "-45:-30", "-30:-15", "-15:+15", "+15:+30", "+30:+45", "+45:+90"]
        out["angle_bin"] = pd.cut(
            ang, bins=bins, labels=labels, right=False, include_lowest=True
        )
    else:
        out["angle_bin"] = pd.NA

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)    # app/features_shots.csv
    ap.add_argument("--by_shot", required=True)     # app/reports/xg_by_shot_all_calibrated_logreg_final.csv
    ap.add_argument("--by_game", required=True)     # app/reports/xg_by_game_all_calibrated_logreg_final.csv
    ap.add_argument("--out_dir", required=True)     # app/reports/coach
    ap.add_argument("--min_bin_size", type=int, default=10)
    args = ap.parse_args()

    feats = pd.read_csv(args.features, dtype=str)
    by_shot = pd.read_csv(args.by_shot, dtype=str)
    by_game = pd.read_csv(args.by_game, dtype=str)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predictions
    if "xg" not in by_shot.columns:
        raise ValueError(f"by_shot missing 'xg'. Columns: {list(by_shot.columns)}")
    by_shot["xg"] = to_num(by_shot["xg"])

    # Labels from features (shot_result == 'goal')
    if "shot_result" not in feats.columns:
        raise ValueError("features_shots.csv missing 'shot_result' column needed to derive goals.")
    feats["is_goal"] = (feats["shot_result"].astype(str).str.lower() == "goal").astype(int)

    # Context to carry
    need = [
        "possession_id","game_id","shot_result",
        "distance_m","angle_deg",  # numeric sources for bins
        "is_man_up","defender_count_bin","attack_type",
        "opponent_team_level","our_team_level",
        "our_team_name","opponent_team_name",
        "player_number","shooter_handedness","event_type"
    ]
    for c in need:
        if c not in feats.columns:
            feats[c] = pd.NA

    # Merge predictions with labels & context
    df = by_shot.merge(feats[need + ["is_goal"]], on="possession_id", how="left")
    df = recompute_bins(df)  # <<< overwrite bins with signed-angle version
    df["is_goal"] = to_num(df["is_goal"]).fillna(0).astype(int)

    # ---------- Overall ----------
    overall = pd.DataFrame([{
        "shots": int(df.shape[0]),
        "goals": int(df["is_goal"].sum()),
        "xg_sum": float(df["xg"].sum()),
        "xg_per_shot": float(df["xg"].mean()),
        "goal_rate": float(df["is_goal"].mean()),
    }])
    overall = add_low_data_flag(overall, count_col="shots", threshold=args.min_bin_size)
    overall.to_csv(out_dir / "coach_overall.csv", index=False)

    # ---------- By game ----------
    bg = by_game.copy()
    for c in ["shots","xg_sum","goals","xg_diff"]:
        if c in bg.columns:
            bg[c] = to_num(bg[c])
    bg["xg_per_shot"] = (bg["xg_sum"] / bg["shots"]).round(3)
    bg["goal_rate"] = (bg["goals"] / bg["shots"]).round(3)
    bg = add_low_data_flag(bg, "shots", args.min_bin_size)
    bg.to_csv(out_dir / "coach_by_game.csv", index=False)

    # ---------- Helper ----------
    def summarize(key):
        g = df.groupby(key, dropna=False).agg(
            n=("is_goal","size"),
            goals=("is_goal","sum"),
            xg_sum=("xg","sum"),
        ).reset_index()
        g["xg_ps"] = (g["xg_sum"] / g["n"]).round(3)
        g["goal_rate"] = (g["goals"] / g["n"]).round(3)
        g["xg_minus_goals"] = (g["xg_sum"] - g["goals"]).round(3)
        g = add_low_data_flag(g, "n", args.min_bin_size)
        g.sort_values("n", ascending=False, inplace=True)
        return g

    # ---------- Context tables ----------
    summarize("distance_bin").to_csv(out_dir / "coach_by_distance.csv", index=False)
    summarize("angle_bin").to_csv(out_dir / "coach_by_angle.csv", index=False)
    summarize("is_man_up").to_csv(out_dir / "coach_by_manup.csv", index=False)
    summarize("attack_type").to_csv(out_dir / "coach_by_attack_type.csv", index=False)
    summarize("defender_count_bin").to_csv(out_dir / "coach_by_defenders.csv", index=False)
    summarize("opponent_team_level").to_csv(out_dir / "coach_by_opponent_level.csv", index=False)

    # ---------- Shooters: group by team + player number ----------
    if "player_number" in df.columns:
        shooters = df.copy()
        shooters["player_number"] = shooters["player_number"].fillna("").astype(str).str.strip()
        shooters["our_team_name"] = shooters["our_team_name"].fillna("").astype(str).str.strip()
        shooters = shooters[shooters["player_number"] != ""]
        grp = shooters.groupby(["our_team_name","player_number"], dropna=False).agg(
            n=("is_goal","size"),
            goals=("is_goal","sum"),
            xg_sum=("xg","sum"),
        ).reset_index()
        grp["xg_ps"] = (grp["xg_sum"] / grp["n"]).round(3)
        grp["goal_rate"] = (grp["goals"] / grp["n"]).round(3)
        grp["xg_minus_goals"] = (grp["xg_sum"] - grp["goals"]).round(3)
        grp = add_low_data_flag(grp, "n", args.min_bin_size)
        grp["shooter_label"] = grp["our_team_name"].fillna("Team") + " #" + grp["player_number"].fillna("?")
        grp = grp.sort_values(["our_team_name","n"], ascending=[True, False])
        grp.to_csv(out_dir / "coach_shooters.csv", index=False)

    print(f"✓ Wrote coach summaries (signed angles, team-aware shooters) → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
