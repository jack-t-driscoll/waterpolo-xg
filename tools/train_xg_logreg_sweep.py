# tools/train_xg_logreg_sweep.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
        if m.sum() == 0:
            continue
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

def build_design(df: pd.DataFrame, y: np.ndarray, folds: int):
    # Numerics (includes new smooth geometry features)
    base_num = [
        "distance_m", "angle_deg", "abs_angle", "angle_x_distance",
        "defender_count", "goalie_distance_m", "possession_passes", "shooter_x", "shooter_y"
    ]
    for c in base_num:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # In-game priors
    prior_num = [
        "poss_idx_in_game", "prior_poss_before", "prior_shots_before", "prior_goals_before",
        "prior_shot_rate_in_game", "prior_goal_rate_in_game",
        "last3_shots_before", "last3_goals_before", "last3_goal_rate_in_game",
    ]
    for c in prior_num:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)

    # Booleans → ints
    for b in ["is_man_up", "empty_net", "is_heave", "is_long_range"]:
        df[b] = df.get(b).map({True:1, False:0, "true":1, "false":0, 1:1, 0:0}).fillna(0).astype(int)

    # Categoricals (use collapsed distance_bin_model from exporter)
    for c in ["goalie_lateral","attack_type","shot_type","shooter_handedness",
              "opponent_team_level","our_team_level","distance_bin","angle_bin",
              "defender_count_bin","dist_x_manup","dist_x_defbin","clock_bin",
              "opponent_team_name","our_team_name","game_id"]:
        df[c] = df.get(c, "Unknown").fillna("Unknown").replace("", "Unknown")

    df["distance_bin_model"] = df["distance_bin"]
    df["dist_x_manup_model"] = df["distance_bin_model"].astype(str) + "|" + df["is_man_up"].astype(str).radd("MU")
    df["dist_x_defbin_model"] = df["distance_bin_model"].astype(str) + "|" + df["defender_count_bin"].astype(str)

    # Group key (by game)
    if "game_id" not in df.columns:
        raise SystemExit("features_shots.csv missing 'game_id'")
    groups = df["game_id"].astype(str)

    # OOF encoding for opponent name (proxy for opponent quality without leakage)
    df["opp_oof_goal_rate"] = oof_target_encode(
        df["opponent_team_name"], y, groups=groups, n_splits=folds, min_count=5, global_prior=0.5
    )

    # Preprocessor
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
    return df, groups, pre

def main():
    ap = argparse.ArgumentParser(description="Sweep LogisticRegression C with GroupKFold by game")
    ap.add_argument("--input", default="app/features_shots.csv")
    ap.add_argument("--out_dir", default="app/reports")
    ap.add_argument("--curation", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--Cs", default="0.3,0.6,1.0,1.5,3.0", help="comma-separated C values")
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

    # Build design/preprocessor
    df, groups, pre = build_design(df, y, folds=args.folds)

    Cs = [float(x) for x in str(args.Cs).split(",") if x.strip() != ""]
    gkf = GroupKFold(n_splits=min(args.folds, max(2, df["game_id"].nunique())))

    metrics_rows = []
    oof_for_best = None
    best_key = None
    best_score = np.inf  # select by lowest logloss

    for C in Cs:
        fold_metrics = []
        oof_preds = np.zeros(len(df), dtype=float)
        for fold_id, (tr, va) in enumerate(gkf.split(df, y, groups), start=1):
            pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=C))])
            pipe.fit(df.iloc[tr], y[tr], **{"clf__sample_weight": sw[tr]})
            p = pipe.predict_proba(df.iloc[va])[:, 1]
            oof_preds[va] = p

            roc = roc_auc_score(y[va], p)
            ll = log_loss(y[va], p, labels=[0,1])
            br = brier_score_loss(y[va], p)
            ece = ece_score(y[va], p, n_bins=10)
            fold_metrics.append((roc, ll, br, ece))

        m = np.array(fold_metrics)
        row = {
            "model": "logreg",
            "C": C,
            "roc_auc_mean": m[:,0].mean(), "roc_auc_std": m[:,0].std(),
            "logloss_mean": m[:,1].mean(), "logloss_std": m[:,1].std(),
            "brier_mean": m[:,2].mean(), "brier_std": m[:,2].std(),
            "ece_mean": m[:,3].mean(), "ece_std": m[:,3].std(),
        }
        metrics_rows.append(row)

        # choose best by mean logloss (lower is better)
        if row["logloss_mean"] < best_score:
            best_score = row["logloss_mean"]
            best_key = {"C": C}
            oof_for_best = oof_preds.copy()

    # Save sweep metrics
    sweep_df = pd.DataFrame(metrics_rows).sort_values("logloss_mean")
    sweep_df.to_csv(out_dir / "metrics_logreg_sweep.csv", index=False)

    # Save best choice + OOF preds for it
    (out_dir / "best_logreg_sweep.json").write_text(json.dumps(best_key, indent=2))
    oof_df = pd.DataFrame({
        "possession_id": df["possession_id"],
        "game_id": df["game_id"],
        "xg": oof_for_best,
        "goal": y,
    })
    oof_df.to_csv(out_dir / "xg_oof_preds_logreg_sweep.csv", index=False)

    print("Best C:", best_key)
    print("✓ Wrote:", (out_dir / "metrics_logreg_sweep.csv").resolve(),
                     (out_dir / "best_logreg_sweep.json").resolve(),
                     (out_dir / "xg_oof_preds_logreg_sweep.csv").resolve())

if __name__ == "__main__":
    main()
