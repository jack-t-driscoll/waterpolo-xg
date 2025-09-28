# tools/export_all_views.py
import argparse
import math
from pathlib import Path
import pandas as pd

# Geometry constants
POOL_WIDTH_M = 20.0          # goal-to-goal width used for lateral spread (consistent with prior convo)
FRONTCOURT_LENGTH_M = 15.0   # y from goal line to 15m

# Our man-up states (from offense perspective)
MAN_UP_STATES_OUR = {"6v5", "6v4"}

# Reporting bins: collapse long-range into 10+ so desperation shots don't dominate
DISTANCE_BINS = [0, 5, 8, 10, 30]
DISTANCE_LABELS = ["0-5", "5-8", "8-10", "10+"]
ANGLE_BINS = [0, 15, 30, 45, 90]
ANGLE_LABELS = ["0-15", "15-30", "30-45", "45-90"]

EXPECTED_COLS = [
    "possession_id","game_id","our_team_name","opponent_team_name",
    "our_team_level","opponent_team_level","period","time_remaining",
    "man_state","event_type","turnover_type","turnover_player_number",
    "drawn_by_player_number","shooter_x","shooter_y","defender_count",
    "goalie_present","goalie_distance_m","goalie_lateral","possession_passes",
    "attack_type","shot_type","shot_result","shooter_handedness","player_number",
    "video_file","video_timestamp_mmss",
]

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # ensure expected columns exist
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in ["event_type","attack_type","man_state","goalie_present","shot_type","shot_result",
              "our_team_level","opponent_team_level","time_remaining"]:
        df[c] = df[c].astype(str).str.strip()
    return df

def to_float(x):
    try:
        return float(x)
    except:
        return float("nan")

def parse_time_to_seconds(s: str):
    if not isinstance(s, str) or s.strip() == "" or s.lower() == "nan":
        return float("nan")
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m, sec = int(parts[0]), int(parts[1])
            if 0 <= m <= 59 and 0 <= sec <= 59:
                return m*60 + sec
        return float("nan")
    except:
        return float("nan")

def compute_distance_angle(x_norm, y_norm):
    """
    Compute distance & signed angle given normalized coordinates:
      - x_norm in [0,1], where 0.5 is goal center; <0.5 is goalie-left (negative), >0.5 goalie-right (positive)
      - y_norm in [0,1], with 0 at goal line, 1 at 15m
    Returns (distance_m, signed_angle_deg)
    """
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return (float("nan"), float("nan"))
    dx = (x_norm - 0.5) * POOL_WIDTH_M
    dy = y_norm * FRONTCOURT_LENGTH_M
    distance = math.hypot(dx, dy)
    base_angle_rad = math.atan2(abs(dx), dy) if dy != 0 else (math.pi / 2)
    base_angle_deg = math.degrees(base_angle_rad)
    angle = -base_angle_deg if dx < 0 else base_angle_deg
    return (round(distance, 3), round(angle, 2))

def build_features_possessions(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f["_row_order"] = range(len(f))
    f = f.sort_values(["game_id", "_row_order"], kind="stable")

    # Event flags
    f["is_shot"] = (f["event_type"] == "shot")
    f["is_turnover"] = (f["event_type"] == "turnover")
    f["is_ejection_drawn"] = (f["event_type"] == "ejection_drawn")
    f["is_5m_drawn"] = (f["event_type"] == "5m_drawn")
    f["is_man_up"] = f["man_state"].isin(MAN_UP_STATES_OUR) | (f["attack_type"].str.lower() == "man-up")
    f["empty_net"] = (f["goalie_present"].str.lower() == "false")

    # Geometry (shots only)
    x = f["shooter_x"].apply(to_float)
    y = f["shooter_y"].apply(to_float)
    dist, ang = [], []
    for xi, yi, shot in zip(x, y, f["is_shot"]):
        if shot:
            d, a = compute_distance_angle(xi, yi)
        else:
            d, a = (float("nan"), float("nan"))
        dist.append(d)
        ang.append(a)
    f["distance_m"] = dist
    f["angle_deg"] = ang

    # NEW: smooth geometry features (shots only; NaN otherwise)
    f["abs_angle"] = f["angle_deg"].abs()
    f["angle_x_distance"] = f["abs_angle"] * f["distance_m"]

    # Bins
    f["distance_bin"] = pd.cut(f["distance_m"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS, right=False)
    f["angle_bin"] = pd.cut(f["angle_deg"].abs(), bins=ANGLE_BINS, labels=ANGLE_LABELS, right=False)

    # Defender count bin (0, 1, 2+)
    f["defender_count"] = pd.to_numeric(f["defender_count"], errors="coerce")
    def bin_def(c):
        if pd.isna(c): return pd.NA
        c = int(c)
        if c <= 0: return "def0"
        if c == 1: return "def1"
        return "def2p"
    f["defender_count_bin"] = f["defender_count"].apply(bin_def)

    # Clock bins
    secs = f["time_remaining"].apply(parse_time_to_seconds)
    f["seconds_remaining"] = secs
    def clock_bucket(s):
        if pd.isna(s): return pd.NA
        s = int(s)
        if s <= 3: return "le3s"
        if s <= 8: return "4to8s"
        return "gt8s"
    f["clock_bin"] = f["seconds_remaining"].apply(clock_bucket)

    # Flags
    f["is_long_range"] = (f["distance_m"] >= 10.0).astype("boolean")
    f["is_heave"] = (
        ((f["distance_m"] >= 10.0) | (f["seconds_remaining"] <= 3)) &
        (f["empty_net"] == False)
    ).astype("boolean")

    # Crosses (shots only)
    f["dist_x_manup"] = pd.NA
    f.loc[f["is_shot"], "dist_x_manup"] = (
        f.loc[f["is_shot"], "distance_bin"].astype(str) + "|" +
        f.loc[f["is_shot"], "is_man_up"].astype(int).astype(str).radd("MU")
    )
    f["dist_x_defbin"] = pd.NA
    f.loc[f["is_shot"], "dist_x_defbin"] = (
        f.loc[f["is_shot"], "distance_bin"].astype(str) + "|" +
        f.loc[f["is_shot"], "defender_count_bin"].astype(str)
    )

    # Outcomes
    def outcome_simple(row):
        if row["event_type"] == "shot":
            return "goal" if (row.get("shot_result","").lower() == "goal") else "shot_no_goal"
        if row["event_type"] == "turnover": return "turnover"
        if row["event_type"] == "ejection_drawn": return "ejection_drawn"
        if row["event_type"] == "5m_drawn": return "5m_drawn"
        return "other"

    def outcome_detail(row):
        if row["event_type"] == "shot": return row.get("shot_result","")
        if row["event_type"] == "turnover": return row.get("turnover_type","turnover")
        return row["event_type"]

    f["possession_outcome_simple"] = f.apply(outcome_simple, axis=1)
    f["possession_outcome_detail"] = f.apply(outcome_detail, axis=1)

    # In-game priors (leakage-safe)
    f["poss_idx_in_game"] = f.groupby("game_id").cumcount() + 1
    f["prior_poss_before"] = f.groupby("game_id").cumcount()

    s_mask = f["is_shot"].fillna(False)
    shot_idx = f.index[s_mask]
    shots = f.loc[s_mask, ["game_id","shot_result"]].copy()
    shots["is_goal"] = (shots["shot_result"].str.lower() == "goal").astype(int)

    shots["prior_shots_before"] = shots.groupby("game_id").cumcount()
    shots["prior_goals_before"] = shots.groupby("game_id")["is_goal"] \
        .transform(lambda s: s.shift(1).fillna(0).cumsum()).astype(int)
    shots["last3_shots_before"] = shots["prior_shots_before"].clip(upper=3)
    shots["last3_goals_before"] = shots.groupby("game_id")["is_goal"] \
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()) \
        .fillna(0).astype(int)

    f.loc[shot_idx, "prior_shots_before"] = shots["prior_shots_before"].values
    f.loc[shot_idx, "prior_goals_before"] = shots["prior_goals_before"].values
    f.loc[shot_idx, "last3_shots_before"] = shots["last3_shots_before"].values
    f.loc[shot_idx, "last3_goals_before"] = shots["last3_goals_before"].values

    def safe_rate(numer, denom, a=1.0, b=2.0):
        numer = pd.to_numeric(numer, errors="coerce").fillna(0).astype(float)
        denom = pd.to_numeric(denom, errors="coerce").fillna(0).astype(float)
        return (numer + a) / (denom + a + b)

    f["prior_shot_rate_in_game"] = safe_rate(f["prior_shots_before"], f["prior_poss_before"])
    f["prior_goal_rate_in_game"] = safe_rate(f["prior_goals_before"], f["prior_shots_before"])
    f["last3_goal_rate_in_game"] = safe_rate(f["last3_goals_before"], f["last3_shots_before"])

    # coerce some numerics
    for col in ["goalie_distance_m","player_number","possession_passes",
                "prior_poss_before","prior_shots_before","prior_goals_before",
                "last3_shots_before","last3_goals_before","poss_idx_in_game"]:
        f[col] = pd.to_numeric(f[col], errors="coerce")

    # Final column order
    cols = [
        "possession_id","game_id","our_team_name","opponent_team_name",
        "period","time_remaining","seconds_remaining","clock_bin",
        "our_team_level","opponent_team_level",
        "man_state","attack_type","event_type","possession_outcome_simple","possession_outcome_detail",
        "is_shot","is_turnover","is_ejection_drawn","is_5m_drawn","is_man_up","empty_net","is_heave","is_long_range",
        "defender_count","defender_count_bin","goalie_distance_m","goalie_lateral","shooter_x","shooter_y",
        "distance_m","angle_deg","abs_angle","angle_x_distance","distance_bin","angle_bin",
        "dist_x_manup","dist_x_defbin",
        "poss_idx_in_game","prior_poss_before","prior_shots_before","prior_goals_before",
        "prior_shot_rate_in_game","prior_goal_rate_in_game","last3_shots_before","last3_goals_before","last3_goal_rate_in_game",
        "shot_type","shot_result","shooter_handedness","player_number",
        "possession_passes","video_file","video_timestamp_mmss",
    ]
    for c in cols:
        if c not in f.columns: f[c] = pd.NA

    f = f.drop(columns=["_row_order"])
    return f[cols].sort_values(["game_id","poss_idx_in_game"], kind="stable")

def build_features_shots_from_possessions(fp: pd.DataFrame) -> pd.DataFrame:
    s = fp[fp["is_shot"]].copy()
    cols = [
        "possession_id","game_id","our_team_name","opponent_team_name",
        "period","time_remaining","seconds_remaining","clock_bin",
        "our_team_level","opponent_team_level",
        "man_state","is_man_up","empty_net","is_heave","is_long_range",
        "shooter_x","shooter_y","distance_m","angle_deg","abs_angle","angle_x_distance",
        "distance_bin","angle_bin",
        "defender_count","defender_count_bin","goalie_distance_m","goalie_lateral",
        "dist_x_manup","dist_x_defbin",
        "poss_idx_in_game","prior_poss_before","prior_shots_before","prior_goals_before",
        "prior_shot_rate_in_game","prior_goal_rate_in_game","last3_shots_before","last3_goals_before","last3_goal_rate_in_game",
        "attack_type","shot_type","shooter_handedness","player_number",
        "possession_passes","shot_result","video_file","video_timestamp_mmss",
    ]
    for c in cols:
        if c not in s.columns: s[c] = pd.NA
    return s[cols].sort_values(["game_id","poss_idx_in_game"], kind="stable")

def summarize_offense(fp: pd.DataFrame, weight_col=None):
    df = fp.copy()
    w = df[weight_col] if (weight_col is not None and weight_col in df.columns) else pd.Series(1.0, index=df.index)
    g = df.groupby("game_id", dropna=False)
    rows = []
    for gid, d in g:
        ww = w.loc[d.index]
        poss = float(ww.sum())
        shots = float((d["is_shot"] * ww).sum())
        goals = float(((d["possession_outcome_detail"] == "goal") * ww).sum())
        turnovers = float((d["is_turnover"] * ww).sum())
        ejects = float((d["is_ejection_drawn"] * ww).sum())
        five_m = float((d["is_5m_drawn"] * ww).sum())
        rows.append({
            "game_id": gid,
            "possessions_weighted": round(poss,3),
            "shots_weighted": round(shots,3),
            "goals_weighted": round(goals,3),
            "shot_rate": round(shots/poss,3) if poss else None,
            "goal_rate": round(goals/poss,3) if poss else None,
            "turnovers_weighted": round(turnovers,3),
            "turnover_rate": round(turnovers/poss,3) if poss else None,
            "ejections_drawn_weighted": round(ejects,3),
            "ejections_per_poss": round(ejects/poss,3) if poss else None,
            "five_m_drawn_weighted": round(five_m,3),
            "5m_per_poss": round(five_m/poss,3) if poss else None,
        })
    per_game = pd.DataFrame(rows)

    poss = float(w.sum()) or 1.0
    shots = float((df["is_shot"] * w).sum())
    goals = float(((df["possession_outcome_detail"] == "goal") * w).sum())
    turnovers = float((df["is_turnover"] * w).sum())
    ejects = float((df["is_ejection_drawn"] * w).sum())
    five_m = float((df["is_5m_drawn"] * w).sum())
    overall = pd.DataFrame([{
        "possessions_weighted": round(poss,3),
        "shots_weighted": round(shots,3), "goals_weighted": round(goals,3),
        "shot_rate": round(shots/poss,3), "goal_rate": round(goals/poss,3),
        "turnovers_weighted": round(turnovers,3), "turnover_rate": round(turnovers/poss,3),
        "ejections_drawn_weighted": round(ejects,3), "ejections_per_poss": round(ejects/poss,3),
        "five_m_drawn_weighted": round(five_m,3), "5m_per_poss": round(five_m/poss,3),
    }])
    return per_game, overall

def load_curation_option_a(curation_path: Path) -> pd.DataFrame:
    if not curation_path or not curation_path.exists():
        return pd.DataFrame(columns=["possession_id","sample_weight"])
    cur = pd.read_csv(curation_path, dtype=str)
    cur.columns = [c.strip().lower() for c in cur.columns]
    if "possession_id" not in cur.columns or "curation_tag" not in cur.columns:
        print("! Curation file missing required columns; ignoring:", curation_path)
        return pd.DataFrame(columns=["possession_id","sample_weight"])
    tag = cur["curation_tag"].astype(str).str.lower().str.strip()
    tag_to_weight = {"organic":1.0, "lightly_curated":0.70, "highly_curated":0.33}
    w = tag.map(tag_to_weight).fillna(1.0)
    return pd.DataFrame({"possession_id": cur["possession_id"], "sample_weight": w})

def main():
    ap = argparse.ArgumentParser(description="Export features + offense summaries (with collapsed long-range bin + smooth geometry).")
    ap.add_argument("--input", default="shots.csv")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--curation", default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(in_path)
    fp = build_features_possessions(df)
    fs = build_features_shots_from_possessions(fp)

    # Optional curation weights (for offense summaries only; trainer can also load curation separately)
    if args.curation:
        cur = load_curation_option_a(Path(args.curation))
        if not cur.empty:
            fp = fp.merge(cur, on="possession_id", how="left")
            fp["sample_weight"] = fp["sample_weight"].fillna(1.0)
        else:
            fp["sample_weight"] = 1.0
    else:
        fp["sample_weight"] = 1.0

    # Write features
    (out_dir / "features_possessions.csv").write_text(fp.to_csv(index=False), encoding="utf-8")
    (out_dir / "features_shots.csv").write_text(fs.to_csv(index=False), encoding="utf-8")

    # Offense summaries (unweighted & weighted)
    un_per_game, un_overall = summarize_offense(fp, weight_col=None)
    (out_dir / "offense_report_by_game.csv").write_text(un_per_game.to_csv(index=False), encoding="utf-8")
    (out_dir / "offense_report_overall.csv").write_text(un_overall.to_csv(index=False), encoding="utf-8")

    w_per_game, w_overall = summarize_offense(fp, weight_col="sample_weight")
    (out_dir / "offense_report_by_game_weighted.csv").write_text(w_per_game.to_csv(index=False), encoding="utf-8")
    (out_dir / "offense_report_overall_weighted.csv").write_text(w_overall.to_csv(index=False), encoding="utf-8")

    print("✓ Wrote:")
    for fname in ["features_possessions.csv","features_shots.csv",
                  "offense_report_by_game.csv","offense_report_overall.csv",
                  "offense_report_by_game_weighted.csv","offense_report_overall_weighted.csv"]:
        print(" -", (out_dir / fname).resolve())

if __name__ == "__main__":
    main()
