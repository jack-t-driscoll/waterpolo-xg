import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression

# -----------------------
# Helpers
# -----------------------
def map_bool(s):
    if s is None:
        return np.zeros(0, dtype=int)
    m = {True: 1, False: 0, "true": 1, "false": 0, "True": 1, "False": 0, 1: 1, 0: 0, "1": 1, "0": 0}
    return pd.Series(s).map(m).fillna(0).astype(int)

def oof_target_encode(keys: pd.Series, y: np.ndarray, groups: pd.Series, n_splits: int = 5,
                      min_count: int = 5, global_prior: float = 0.5) -> np.ndarray:
    """Leakage-safe out-of-fold target encoding for a single categorical feature."""
    keys = keys.astype(str)
    gkf = GroupKFold(n_splits=max(2, min(n_splits, groups.nunique())))
    oof = np.zeros(len(y), dtype=float)
    for tr, va in gkf.split(np.zeros(len(y)), y, groups):
        tr_keys = keys.iloc[tr]
        tr_y = y[tr]
        counts = tr_keys.value_counts()
        means = tr_keys.groupby(tr_keys).apply(lambda k: tr_y[tr_keys.values == k.iloc[0]].mean())
        means = means[counts[means.index] >= min_count]
        enc = pd.Series(global_prior, index=keys.index)
        enc.update(keys.map(means).fillna(global_prior))
        oof[va] = enc.iloc[va].values
    return oof

# -----------------------
# Design builder
# -----------------------
def build_design(df: pd.DataFrame, y: np.ndarray, folds: int):
    # Core numeric geometry
    base_num = [
        "distance_m", "angle_deg", "abs_angle", "angle_x_distance",
        "defender_count", "goalie_distance_m", "possession_passes",
        "shooter_x", "shooter_y",
    ]
    # Robust numeric coercion
    for c in base_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    # In-game prior numerics (0.0 default)
    prior_num = [
        "poss_idx_in_game", "prior_poss_before", "prior_shots_before", "prior_goals_before",
        "prior_shot_rate_in_game", "prior_goal_rate_in_game",
        "last3_shots_before", "last3_goals_before", "last3_goal_rate_in_game",
    ]
    for c in prior_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    # Booleans → ints
    for b in ["is_man_up", "empty_net", "is_heave", "is_long_range"]:
        df[b] = map_bool(df.get(b, 0))

    # Fill categorical sources (ensure series, not scalars)
    for c in [
        "goalie_lateral", "attack_type", "shot_type", "shooter_handedness",
        "opponent_team_level", "our_team_level",
        "distance_bin", "angle_bin", "defender_count_bin",
        "dist_x_manup", "dist_x_defbin", "clock_bin",
        "opponent_team_name", "our_team_name", "game_id"
    ]:
        df[c] = df[c] if c in df.columns else "Unknown"
        df[c] = df[c].fillna("Unknown").replace("", "Unknown")

    # Collapse distance bins and crosses for the model (string-safe)
    df["distance_bin_model"] = df["distance_bin"].astype(str)
    df["dist_x_manup_model"] = (df["distance_bin_model"].astype(str) + "|" +
                                df["is_man_up"].astype(str).radd("MU"))
    df["dist_x_defbin_model"] = (df["distance_bin_model"].astype(str) + "|" +
                                 df["defender_count_bin"].astype(str))

    # Group key
    if "game_id" not in df.columns:
        raise SystemExit("features_shots.csv missing 'game_id'")
    groups = df["game_id"].astype(str)

    # OOF opponent mean encoding (leakage-safe)
    df["opp_oof_goal_rate"] = oof_target_encode(
        df["opponent_team_name"], y, groups=groups, n_splits=folds, min_count=5, global_prior=0.5
    )

    # -----------------------
    # TRIMMED cat cols (lighter, less fragmentation)
    # Keep: core categories & distance/angle bins
    # Drop: clock_bin, dist_x_manup_model, dist_x_defbin_model (noisy/fragmented)
    # -----------------------
    cat_cols = [
        "goalie_lateral", "attack_type", "shot_type", "shooter_handedness",
        "opponent_team_level", "distance_bin_model", "angle_bin", "defender_count_bin",
    ]
    num_with_bools = base_num + prior_num + ["is_man_up", "empty_net", "is_heave", "is_long_range", "opp_oof_goal_rate"]

    # Preprocessor
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer([
        ("num", numeric_pipeline, num_with_bools),
        ("cat", categorical_pipeline, cat_cols),
    ])

    return df, groups, pre

# -----------------------
# Main
# -----------------------
def load_curation_weights(p: Path):
    if p is None or not p.exists():
        return None
    w = pd.read_csv(p, dtype=str)
    if not {"possession_id", "weight"}.issubset(w.columns):
        return None
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(1.0)
    return dict(zip(w["possession_id"], w["weight"]))

def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

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
    out_path = out_dir / "xg_logreg_final.pkl"
    import joblib
    joblib.dump({
        "model": model_full,
        "iso_global": iso_global,
        "iso_by_level": iso_by_level,
        "iso_by_key": iso_by_key,
    }, out_path)

    # Export calibrated per-shot and per-game reports
    p_all = model_full.predict_proba(df)[:, 1]
    p_cal = iso_global.predict(p_all)
    def apply_cal(row):
        lv = str(row[level_col])
        db = str(row["distance_bin_model"])
        key = (lv, db)
        if key in iso_by_key:
            return float(iso_by_key[key].predict([row["xg_raw"]])[0])
        if lv in iso_by_level:
            return float(iso_by_level[lv].predict([row["xg_raw"]])[0])
        return float(p_cal[row.name])

    by_shot = pd.DataFrame({
        "possession_id": df["possession_id"],
        "game_id": df["game_id"],
        "distance_bin_model": df["distance_bin_model"],
        "opponent_team_level": df[level_col],
        "xg_raw": p_all,
    })
    by_shot["xg_cal"] = by_shot.apply(apply_cal, axis=1)
    by_shot.to_csv(reports_dir / "xg_by_shot_all_calibrated_logreg_final.csv", index=False)

    by_game = by_shot.groupby("game_id", as_index=False).agg(xg_cal=("xg_cal", "sum"), goals=("possession_id", "size"))
    by_game.to_csv(reports_dir / "xg_by_game_all_calibrated_logreg_final.csv", index=False)

    final_metrics = {
        "model": "logreg_final_C0.3",
        "roc_auc_holdout": float(metrics_row["roc_auc"]),
        "logloss_holdout": float(metrics_row["logloss"]),
        "brier_holdout": float(metrics_row["brier"]),
        "ece_holdout": float(metrics_row["ece"]),
        "n_rows": int(len(df)),
    }
    (reports_dir / "metrics_validation_final.json").write_text(json.dumps(final_metrics, indent=2))

    print("Validation (holdout by game):", final_metrics)
    print("✓ wrote models →", out_path.resolve())
    print("✓ wrote reports →", (reports_dir / "xg_by_shot_all_calibrated_logreg_final.csv").resolve())
    print("✓ wrote OOF →", (reports_dir / "xg_oof_preds_logreg_final.csv").resolve())

if __name__ == "__main__":
    main()
