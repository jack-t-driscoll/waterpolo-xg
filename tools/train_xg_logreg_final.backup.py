# tools/train_xg_logreg_final.py
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if m.sum() == 0: continue
        ece += (m.mean()) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)

def oof_target_encode(series, y, groups, n_splits=5, min_count=5, global_prior=0.5):
    s = series.fillna("Unknown").astype(str)
    y = np.asarray(y).astype(int)
    gkf = GroupKFold(n_splits=min(n_splits, max(2, len(np.unique(groups)))))
    oof = np.zeros(len(s), dtype=float)
    global_mean = float(y.mean()) if len(y) else global_prior
    for tr, va in gkf.split(s, y, groups):
        tmp = pd.DataFrame({"cat": s.iloc[tr].values, "y": y[tr]})
        grp = tmp.groupby("cat", sort=False).agg(cnt=("y","size"), mean=("y","mean"))
        smooth = (grp["mean"] * grp["cnt"] + global_mean * min_count) / (grp["cnt"] + min_count)
        m = smooth.to_dict()
        oof[va] = [m.get(k, global_mean) for k in s.iloc[va]]
    return oof

def load_curation_weights(curation_path: Path):
    if not curation_path or not curation_path.exists():
        return None
    cur = pd.read_csv(curation_path, dtype=str)
    cur.columns = [c.strip().lower() for c in cur.columns]
    if "possession_id" not in cur.columns or "curation_tag" not in cur.columns:
        return None
    tag = cur["curation_tag"].astype(str).str.lower().str.strip()
    tag_to_weight = {"organic":1.0, "lightly_curated":0.70, "highly_curated":0.33}
    w = tag.map(tag_to_weight).fillna(1.0)
    return dict(zip(cur["possession_id"], w))

def map_bool(s):
    return pd.Series(s).map({True:1, False:0, "true":1, "false":0, 1:1, 0:0}).fillna(0).astype(int)

def build_design(df: pd.DataFrame, y: np.ndarray, folds: int):
    # Numerics (includes smooth geometry)
    base_num = [
        "distance_m","angle_deg","abs_angle","angle_x_distance",
        "defender_count","goalie_distance_m","possession_passes","shooter_x","shooter_y"
    ]
    for c in base_num: df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # In-game priors
    prior_num = [
        "poss_idx_in_game","prior_poss_before","prior_shots_before","prior_goals_before",
        "prior_shot_rate_in_game","prior_goal_rate_in_game",
        "last3_shots_before","last3_goals_before","last3_goal_rate_in_game"
    ]
    for c in prior_num: df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)

    # Booleans → ints
    for b in ["is_man_up","empty_net","is_heave","is_long_range"]:
        df[b] = map_bool(df.get(b, 0))

    # Categorical features
    for c in ["goalie_lateral","attack_type","shot_type","shooter_handedness",
              "opponent_team_level","our_team_level","distance_bin","angle_bin",
              "defender_count_bin","dist_x_manup","dist_x_defbin","clock_bin",
              "opponent_team_name","our_team_name","game_id"]:
        df[c] = df.get(c, "Unknown").fillna("Unknown").replace("", "Unknown")

    # Collapsed distance bin + crosses for the model
    df["distance_bin_model"] = df["distance_bin"]
    df["dist_x_manup_model"] = df["distance_bin_model"].astype(str) + "|" + df["is_man_up"].astype(str).radd("MU")
    df["dist_x_defbin_model"] = df["distance_bin_model"].astype(str) + "|" + df["defender_count_bin"].astype(str)

    # Group key
    if "game_id" not in df.columns: raise SystemExit("features_shots.csv missing 'game_id'")
    groups = df["game_id"].astype(str)

    # OOF opponent mean encoding (leakage-safe)
    df["opp_oof_goal_rate"] = oof_target_encode(
        df["opponent_team_name"], y, groups=groups, n_splits=folds, min_count=5, global_prior=0.5
    )

    # Preprocessor
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    cat_cols = ["goalie_lateral","attack_type","shot_type","shooter_handedness",
                "opponent_team_level","distance_bin_model","angle_bin",
                "defender_count_bin","dist_x_manup_model","dist_x_defbin_model","clock_bin"]
    num_with_bools = base_num + prior_num + ["is_man_up","empty_net","is_heave","is_long_range","opp_oof_goal_rate"]

    pre = ColumnTransformer([("num", numeric_pipeline, num_with_bools),
                             ("cat", categorical_pipeline, cat_cols)])
    return df, groups, pre

def main():
    ap = argparse.ArgumentParser(description="FINAL: LogReg (C=0.3) with OOF isotonic calibration and final exports")
    ap.add_argument("--input", default="app/features_shots.csv")
    ap.add_argument("--out_dir", default="app/models")
    ap.add_argument("--reports_dir", default="app/reports")
    ap.add_argument("--curation", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--downweight_long_range", type=float, default=None,
                    help="Optional extra downweight for distance>=10m shots, e.g. 0.5")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str)
    if "shot_result" not in df.columns:
        raise SystemExit("features_shots.csv missing 'shot_result' column.")
    y = (df["shot_result"].str.lower() == "goal").astype(int).values

    # Base sample weights
    if "sample_weight" in df.columns:
        sw = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).values.astype(float)
    else:
        wm = load_curation_weights(Path(args.curation)) if args.curation else None
        sw = df["possession_id"].map(wm).fillna(1.0).values.astype(float) if wm else np.ones(len(df), dtype=float)

    # Optional extra downweighting for long-range
    if args.downweight_long_range is not None and "is_long_range" in df.columns:
        is_long = df["is_long_range"].map({True:1,False:0,"true":1,"false":0,1:1,0:0}).fillna(0).astype(int).values
        sw = sw * np.where(is_long==1, float(args.downweight_long_range), 1.0)

    # Design
    df, groups, pre = build_design(df, y, folds=args.folds)

    # Model: Logistic Regression with fixed C=0.3
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=0.3))])

    # Holdout-by-game metrics (quick sense check)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    tr_idx, va_idx = next(gss.split(df, y, groups))
    df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    sw_tr, sw_va = sw[tr_idx], sw[va_idx]

    model.fit(df_tr, y_tr, **{"clf__sample_weight": sw_tr})
    p_va = model.predict_proba(df_va)[:, 1]
    metrics_row = {
        "model": "logreg_final_C0.3",
        "roc_auc": roc_auc_score(y_va, p_va),
        "logloss": log_loss(y_va, p_va, labels=[0,1]),
        "brier": brier_score_loss(y_va, p_va),
        "ece": ece_score(y_va, p_va, n_bins=10),
    }
    pd.DataFrame([metrics_row]).to_csv(reports_dir / "metrics_validation_final.csv", index=False)

    # OOF predictions (by game)
    gkf = GroupKFold(n_splits=min(args.folds, max(2, df["game_id"].nunique())))
    oof = np.zeros(len(df), dtype=float)
    for tr, va in gkf.split(df, y, groups):
        pipe_cv = clone(model)
        pipe_cv.fit(df.iloc[tr], y[tr], **{"clf__sample_weight": sw[tr]})
        oof[va] = pipe_cv.predict_proba(df.iloc[va])[:, 1]
    oof_df = pd.DataFrame({"possession_id": df["possession_id"], "game_id": df["game_id"], "xg": oof, "goal": y})
    oof_df.to_csv(reports_dir / "xg_oof_preds_logreg_final.csv", index=False)

    # Fit isotonic calibration: global → by level → by (level, distance_bin)
    iso_global = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    level_col = "opponent_team_level"
    df[level_col] = df[level_col].fillna("Unknown").replace("", "Unknown")
    df["distance_bin_model"] = df["distance_bin"].fillna("Unknown").astype(str)

    iso_by_level, iso_by_key = {}, {}
    for lv in sorted(df[level_col].astype(str).unique()):
        idx = (df[level_col].astype(str) == lv)
        if idx.sum() >= 40 and len(np.unique(y[idx])) > 1:
            iso_by_level[lv] = IsotonicRegression(out_of_bounds="clip").fit(oof[idx], y[idx])
    for key in sorted(set(zip(df[level_col].astype(str), df["distance_bin_model"]))):
        idx = (df[level_col].astype(str) == key[0]) & (df["distance_bin_model"] == key[1])
        if idx.sum() >= 40 and len(np.unique(y[idx])) > 1:
            iso_by_key[key] = IsotonicRegression(out_of_bounds="clip").fit(oof[idx], y[idx])

    # Train final on ALL data + save
    model_full = clone(model).fit(df, y, **{"clf__sample_weight": sw})
    joblib.dump(model_full, out_dir / "xg_logreg_final.pkl")
    joblib.dump(iso_global, out_dir / "xg_logreg_final_isotonic.pkl")
    joblib.dump(iso_by_level, out_dir / "xg_logreg_final_isotonic_by_level.pkl")
    joblib.dump(iso_by_key, out_dir / "xg_logreg_final_isotonic_by_level_and_dist.pkl")

    # Raw & calibrated predictions for ALL rows
    p_all_raw = model_full.predict_proba(df)[:, 1]
    def apply_cal(p, lv, db):
        lv_key = str(lv).strip() if isinstance(lv, str) and lv.strip() else "Unknown"
        db_key = str(db).strip() if isinstance(db, str) and db.strip() else "Unknown"
        iso = iso_by_key.get((lv_key, db_key)) or iso_by_level.get(lv_key) or iso_global
        return float(iso.predict([p])[0])
    p_all_cal = np.array([apply_cal(p, lv, db) for p, lv, db in zip(p_all_raw, df[level_col], df["distance_bin_model"])], dtype=float)

    # Write by-shot
    by_shot_raw = df[["possession_id","game_id","video_file","video_timestamp_mmss"]].copy()
    by_shot_raw["xg"] = p_all_raw
    by_shot_raw.to_csv(reports_dir / "xg_by_shot_all_logreg_final.csv", index=False)

    by_shot_cal = by_shot_raw.copy()
    by_shot_cal["xg"] = p_all_cal
    by_shot_cal.to_csv(reports_dir / "xg_by_shot_all_calibrated_logreg_final.csv", index=False)

    # Prepare goals by game once
    goals_all = df.assign(goal=y).groupby("game_id", as_index=False)["goal"].sum()

    # Write by-game (FIXED: rename 'goal'->'goals' before computing xg_diff)
    def agg_game(xg_series):
        g = by_shot_raw.assign(xg=xg_series).groupby("game_id", as_index=False).agg(
            shots=("xg","size"), xg_sum=("xg","sum")
        )
        g = g.merge(goals_all, on="game_id", how="left").rename(columns={"goal":"goals"})
        g["xg_diff"] = g["goals"] - g["xg_sum"]
        return g

    by_game_raw = agg_game(p_all_raw)
    by_game_cal = agg_game(p_all_cal)
    by_game_raw.to_csv(reports_dir / "xg_by_game_all_logreg_final.csv", index=False)
    by_game_cal.to_csv(reports_dir / "xg_by_game_all_calibrated_logreg_final.csv", index=False)

    # Final summary metrics on ALL (calibrated)
    final_metrics = {
        "model": "logreg_final_C0.3",
        "roc_auc_holdout": metrics_row["roc_auc"],
        "logloss_holdout": metrics_row["logloss"],
        "brier_holdout": metrics_row["brier"],
        "ece_holdout": metrics_row["ece"],
        "n_rows": int(len(df))
    }
    (reports_dir / "metrics_validation_final.json").write_text(json.dumps(final_metrics, indent=2))

    print("Validation (holdout by game):", metrics_row)
    print("✓ wrote models →", (out_dir / "xg_logreg_final.pkl").resolve())
    print("✓ wrote reports →", (reports_dir / "xg_by_shot_all_calibrated_logreg_final.csv").resolve())
    print("✓ wrote OOF →", (reports_dir / "xg_oof_preds_logreg_final.csv").resolve())

if __name__ == "__main__":
    main()
