import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
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
        grp = tmp.groupby("cat", sort=False).agg(cnt=("y", "size"), mean=("y", "mean"))
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
    tag_to_weight = {"organic": 1.0, "lightly_curated": 0.70, "highly_curated": 0.33}
    w = tag.map(tag_to_weight).fillna(1.0)
    return dict(zip(cur["possession_id"], w))

def main():
    ap = argparse.ArgumentParser(description="GroupKFold CV by game with OOF opponent encoding + sample weights")
    ap.add_argument("--input", default="app/features_shots.csv")
    ap.add_argument("--out_dir", default="app/reports")
    ap.add_argument("--curation", default=None, help="optional: app/curation.csv (possession_id, curation_tag)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str)
    if "shot_result" not in df.columns:
        raise SystemExit("features_shots.csv missing 'shot_result' column.")
    y = (df["shot_result"].str.lower() == "goal").astype(int).values

    # Sample weights
    if "sample_weight" in df.columns:
        sw = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).values.astype(float)
    else:
        wm = load_curation_weights(Path(args.curation)) if args.curation else None
        sw = df["possession_id"].map(wm).fillna(1.0).values.astype(float) if wm else np.ones(len(df), dtype=float)

    # Numerics (NOW includes abs_angle & angle_x_distance)
    base_num = [
        "distance_m","angle_deg","abs_angle","angle_x_distance",
        "defender_count","goalie_distance_m","possession_passes","shooter_x","shooter_y"
    ]
    for c in base_num:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # In-game priors
    prior_num = [
        "poss_idx_in_game","prior_poss_before","prior_shots_before","prior_goals_before",
        "prior_shot_rate_in_game","prior_goal_rate_in_game",
        "last3_shots_before","last3_goals_before","last3_goal_rate_in_game",
    ]
    for c in prior_num:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)

    # Booleans → ints
    for b in ["is_man_up","empty_net","is_heave","is_long_range"]:
        df[b] = df.get(b).map({True:1, False:0, "true":1, "false":0, 1:1, 0:0}).fillna(0).astype(int)

    # Categorical + collapsed distance bins
    for c in ["goalie_lateral","attack_type","shot_type","shooter_handedness",
              "opponent_team_level","our_team_level","distance_bin","angle_bin",
              "defender_count_bin","dist_x_manup","dist_x_defbin","clock_bin",
              "opponent_team_name","our_team_name","game_id"]:
        df[c] = df.get(c, "Unknown").fillna("Unknown").replace("", "Unknown")

    df["distance_bin_model"] = df["distance_bin"]
    df["dist_x_manup_model"] = df["distance_bin_model"].astype(str) + "|" + df["is_man_up"].astype(str).radd("MU")
    df["dist_x_defbin_model"] = df["distance_bin_model"].astype(str) + "|" + df["defender_count_bin"].astype(str)

    if "game_id" not in df.columns:
        raise SystemExit("features_shots.csv missing 'game_id'")
    groups = df["game_id"].astype(str)

    # OOF opponent name encoding
    df["opp_oof_goal_rate"] = oof_target_encode(
        df["opponent_team_name"], y, groups=groups, n_splits=args.folds, min_count=5, global_prior=0.5
    )

    # Pipelines
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    cat_cols = [
        "goalie_lateral","attack_type","shot_type","shooter_handedness",
        "opponent_team_level","distance_bin_model","angle_bin",
        "defender_count_bin","dist_x_manup_model","dist_x_defbin_model","clock_bin",
    ]
    num_with_bools = base_num + prior_num + ["is_man_up","empty_net","is_heave","is_long_range","opp_oof_goal_rate"]

    pre = ColumnTransformer([("num", numeric_pipeline, num_with_bools),
                             ("cat", categorical_pipeline, cat_cols)])

    models = {
        "logreg": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=1.0))]),
        "hgb":   Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
            max_depth=None, learning_rate=0.06, max_iter=400, l2_regularization=0.0, random_state=42))]),
    }

    gkf = GroupKFold(n_splits=min(args.folds, max(2, df["game_id"].nunique())))
    rows, preds = [], []

    for name, pipe in models.items():
        fold_metrics = []
        fold_id = 0
        for tr, va in gkf.split(df, y, groups):
            fold_id += 1
            pipe.fit(df.iloc[tr], y[tr], **{"clf__sample_weight": sw[tr]})
            p = pipe.predict_proba(df.iloc[va])[:, 1]
            roc = roc_auc_score(y[va], p)
            ll = log_loss(y[va], p, labels=[0, 1])
            br = brier_score_loss(y[va], p)
            ece = ece_score(y[va], p, n_bins=10)
            fold_metrics.append((roc, ll, br, ece))
            preds.append(pd.DataFrame({
                "model": name, "fold": fold_id,
                "possession_id": df.iloc[va]["possession_id"].values,
                "game_id": df.iloc[va]["game_id"].values,
                "xg": p, "goal": y[va],
            }))

        m = np.array(fold_metrics)
        rows.append({
            "model": name,
            "roc_auc_mean": m[:, 0].mean(), "roc_auc_std": m[:, 0].std(),
            "logloss_mean": m[:, 1].mean(), "logloss_std": m[:, 1].std(),
            "brier_mean": m[:, 2].mean(), "brier_std": m[:, 2].std(),
            "ece_mean": m[:, 3].mean(), "ece_std": m[:, 3].std(),
        })

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(out_dir / "metrics_cv.csv", index=False)
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(out_dir / "xg_oof_preds.csv", index=False)
    print("✓ Wrote:", (out_dir / "metrics_cv.csv").resolve(), (out_dir / "xg_oof_preds.csv").resolve())

if __name__ == "__main__":
    main()
