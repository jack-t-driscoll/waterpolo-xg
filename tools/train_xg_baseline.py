import argparse, json
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
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

def plot_calibration(y_true, y_prob, out_path):
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob)
    bins = np.linspace(0,1,11); digitized = np.digitize(y_prob, bins) - 1
    xs, ys, ns = [], [], []
    for i in range(10):
        m = (digitized == i)
        if m.sum()==0: continue
        xs.append(y_prob[m].mean()); ys.append(y_true[m].mean()); ns.append(m.sum())
    import matplotlib.pyplot as plt
    plt.figure(); plt.plot([0,1],[0,1],"--"); plt.scatter(xs,ys,s=np.array(ns))
    plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title("Calibration")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

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
        print("! Curation file missing required columns; ignoring:", curation_path)
        return None
    tag = cur["curation_tag"].astype(str).str.lower().str.strip()
    tag_to_weight = {"organic":1.0, "lightly_curated":0.70, "highly_curated":0.33}
    w = tag.map(tag_to_weight).fillna(1.0)
    return dict(zip(cur["possession_id"], w))

def main():
    ap = argparse.ArgumentParser(description="xG with collapsed long-range bin + smooth geometry + in-game priors + OOF opponent encoding")
    ap.add_argument("--input", default="app/features_shots.csv")
    ap.add_argument("--out_dir", default="app/models")
    ap.add_argument("--curation", default=None)
    ap.add_argument("--downweight_long_range", type=float, default=None,
                    help="e.g. 0.5 to reduce weight of distance>=10m shots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir.parent / "reports"; reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str)
    if "shot_result" not in df.columns:
        raise SystemExit("features_shots.csv missing 'shot_result' column.")
    y = (df["shot_result"].str.lower() == "goal").astype(int).values

    # Sample weights
    if "sample_weight" in df.columns:
        sw = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).values.astype(float)
    else:
        weights_map = load_curation_weights(Path(args.curation)) if args.curation else None
        sw = df["possession_id"].map(weights_map).fillna(1.0).values.astype(float) if weights_map else np.ones(len(df), dtype=float)

    # Optional extra downweighting for long-range
    if args.downweight_long_range is not None and "is_long_range" in df.columns:
        is_long = df["is_long_range"].map({True:1,False:0,"true":1,"false":0,1:1,0:0}).fillna(0).astype(int).values
        sw = sw * np.where(is_long==1, float(args.downweight_long_range), 1.0)

    # Numeric base (NOW includes abs_angle & angle_x_distance)
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
    def map_bool(s): return pd.Series(s).map({True:1, False:0, "true":1, "false":0, 1:1, 0:0}).fillna(0).astype(int)
    for b in ["is_man_up","empty_net","is_heave","is_long_range"]:
        if b in df.columns:
            df[b] = map_bool(df.get(b, 0))
        else:
            df[b] = 0

    # Cats
    for c in ["goalie_lateral","attack_type","shot_type","shooter_handedness",
              "opponent_team_level","our_team_level","distance_bin","angle_bin",
              "defender_count_bin","dist_x_manup","dist_x_defbin","clock_bin",
              "opponent_team_name","our_team_name","game_id"]:
        df[c] = df.get(c, "Unknown").fillna("Unknown").replace("", "Unknown")

    # Use distance_bin directly (already collapsed to 10+ by exporter)
    df["distance_bin_model"] = df["distance_bin"]
    df["dist_x_manup_model"] = df["distance_bin_model"].astype(str) + "|" + df["is_man_up"].astype(str).radd("MU")
    df["dist_x_defbin_model"] = df["distance_bin_model"].astype(str) + "|" + df["defender_count_bin"].astype(str)

    # Groups by game
    if "game_id" not in df.columns: raise SystemExit("features_shots.csv missing 'game_id'")
    groups = df["game_id"].astype(str)

    # OOF opponent mean encoding
    df["opp_oof_goal_rate"] = oof_target_encode(df["opponent_team_name"], y, groups=groups, n_splits=5, min_count=5, global_prior=0.5)

    # Pipelines
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    cat_cols = ["goalie_lateral","attack_type","shot_type","shooter_handedness",
                "opponent_team_level","distance_bin_model","angle_bin",
                "defender_count_bin","dist_x_manup_model","dist_x_defbin_model","clock_bin"]

    num_with_bools = base_num + prior_num + ["is_man_up","empty_net","is_heave","is_long_range","opp_oof_goal_rate"]

    pre = ColumnTransformer([("num", numeric_pipeline, num_with_bools),
                             ("cat", categorical_pipeline, cat_cols)])

    models = {
        "logreg": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=1.0))]),
        "hgb":   Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
                    max_depth=None, learning_rate=0.06, max_iter=400, l2_regularization=0.0, random_state=42))]),
    }

    # Holdout by game for metrics
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    tr_idx, va_idx = next(gss.split(df, y, groups))
    df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    sw_tr, sw_va = sw[tr_idx], sw[va_idx]

    metrics_rows = []
    for name, pipe in models.items():
        pipe.fit(df_tr, y_tr, **{ "clf__sample_weight": sw_tr })
        proba_val = pipe.predict_proba(df_va)[:, 1]
        roc = roc_auc_score(y_va, proba_val)
        ll = log_loss(y_va, proba_val, labels=[0,1])
        brier = brier_score_loss(y_va, proba_val)
        ece = ece_score(y_va, proba_val, n_bins=10)
        metrics_rows.append({"model": name, "roc_auc": roc, "logloss": ll, "brier": brier, "ece": ece})
        plot_calibration(y_va, proba_val, reports_dir / f"calibration_{name}.png")

        xg_va = df_va[["possession_id","game_id","video_file","video_timestamp_mmss"]].copy()
        xg_va["xg"] = proba_val
        xg_va.to_csv(reports_dir / f"xg_by_shot_{name}.csv", index=False)

        by_game = xg_va.groupby("game_id", as_index=False).agg(shots=("xg","size"), xg_sum=("xg","sum"))
        goals_by_game = df_va.assign(goal=y_va).groupby("game_id", as_index=False)["goal"].sum()
        by_game = by_game.merge(goals_by_game, on="game_id", how="left").rename(columns={"goal":"goals"})
        by_game["xg_diff"] = by_game["goals"] - by_game["xg_sum"]
        by_game.to_csv(reports_dir / f"xg_by_game_{name}.csv", index=False)

        joblib.dump(pipe, out_dir / f"xg_{name}.pkl")

        # ---------- OOF isotonic ----------
        gkf = GroupKFold(n_splits=min(5, df["game_id"].nunique() or 2))
        oof = np.zeros(len(df), dtype=float)
        for tr, va in gkf.split(df, y, groups):
            pipe_cv = clone(pipe)
            pipe_cv.fit(df.iloc[tr], y[tr], **{ "clf__sample_weight": sw[tr] })
            oof[va] = pipe_cv.predict_proba(df.iloc[va])[:, 1]

        iso_global = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
        level_col = "opponent_team_level"
        levels = df[level_col].fillna("Unknown").replace("", "Unknown").astype(str).unique().tolist()
        iso_by_level, iso_by_key = {}, {}
        for lv in sorted(levels):
            idx_lv = (df[level_col].astype(str).fillna("Unknown").replace("", "Unknown").values == lv)
            if idx_lv.sum() >= 40 and len(np.unique(y[idx_lv])) > 1:
                iso_by_level[lv] = IsotonicRegression(out_of_bounds="clip").fit(oof[idx_lv], y[idx_lv])
        lev_vals = df[level_col].fillna("Unknown").replace("", "Unknown").astype(str)
        dist_vals = df["distance_bin_model"].fillna("Unknown").astype(str)
        for key in sorted(set(zip(lev_vals, dist_vals))):
            idx_key = (lev_vals == key[0]) & (dist_vals == key[1])
            if idx_key.sum() >= 40 and len(np.unique(y[idx_key])) > 1:
                iso_by_key[key] = IsotonicRegression(out_of_bounds="clip").fit(oof[idx_key], y[idx_key])

        joblib.dump(iso_global, out_dir / f"xg_{name}_isotonic.pkl")
        joblib.dump(iso_by_level, out_dir / f"xg_{name}_isotonic_by_level.pkl")
        joblib.dump(iso_by_key, out_dir / f"xg_{name}_isotonic_by_level_and_dist.pkl")

        # Final model on ALL + calibrated predictions
        pipe_full = clone(pipe).fit(df, y, **{ "clf__sample_weight": sw })
        joblib.dump(pipe_full, out_dir / f"xg_{name}_full.pkl")
        proba_all_raw = pipe_full.predict_proba(df)[:, 1]

        def apply_cal(p, lv, db):
            lv_key = str(lv).strip() if isinstance(lv, str) and lv.strip() else "Unknown"
            db_key = str(db).strip() if isinstance(db, str) and db.strip() else "Unknown"
            iso = iso_by_key.get((lv_key, db_key)) or iso_by_level.get(lv_key) or iso_global
            return float(iso.predict([p])[0])

        proba_all_cal = np.array([apply_cal(p, lv, db)
                                  for p, lv, db in zip(proba_all_raw, df[level_col], df["distance_bin_model"])],
                                 dtype=float)

        for tag, probs in [("all", proba_all_raw), ("all_calibrated", proba_all_cal)]:
            xg_all = df[["possession_id","game_id","video_file","video_timestamp_mmss"]].copy()
            xg_all["xg"] = probs
            xg_all.to_csv(reports_dir / f"xg_by_shot_{tag}_{name}.csv", index=False)

            by_game_all = xg_all.groupby("game_id", as_index=False).agg(shots=("xg","size"), xg_sum=("xg","sum"))
            goals_by_game_all = df.assign(goal=y).groupby("game_id", as_index=False)["goal"].sum()
            by_game_all = by_game_all.merge(goals_by_game_all, on="game_id", how="left").rename(columns={"goal":"goals"})
            by_game_all["xg_diff"] = by_game_all["goals"] - by_game_all["xg_sum"]
            by_game_all.to_csv(reports_dir / f"xg_by_game_{tag}_{name}.csv", index=False)

    pd.DataFrame(metrics_rows).to_csv(reports_dir / "metrics_validation.csv", index=False)
    (reports_dir / "metrics_validation.json").write_text(json.dumps(metrics_rows, indent=2))
    print(pd.DataFrame(metrics_rows))
    print("✓ wrote models & reports to", reports_dir.resolve())

if __name__ == "__main__":
    main()
