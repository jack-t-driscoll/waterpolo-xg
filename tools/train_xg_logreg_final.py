# tools/train_xg_logreg_final.py
# FINAL: Logistic Regression (C=0.3) with OOF isotonic calibration and final exports
# Outputs:
#   app/reports/metrics_validation_final.csv
#   app/reports/metrics_validation_final.json
#   app/reports/xg_by_shot_all_calibrated_logreg_final.csv
#   app/reports/xg_by_game_all_calibrated_logreg_final.csv
#
# Robust to missing columns, derives geometry if absent,
# and NOW imputes NaNs so LogisticRegression never sees missing values.

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ----------------------------- Helpers --------------------------------

def map_bool(x):
    """Map common truthy/falsey values to 0/1 (ints)."""
    if isinstance(x, pd.Series):
        s = x.astype(str).str.strip().str.lower()
    else:
        s = pd.Series(x).astype(str).str.strip().str.lower()
    true_set = {"1","true","yes","y","t"}
    false_set = {"0","false","no","n","f",""}
    out = []
    for v in s:
        if v in true_set: out.append(1)
        elif v in false_set: out.append(0)
        else:
            try:
                out.append(1 if float(v) != 0 else 0)
            except Exception:
                out.append(0)
    return pd.Series(out, index=s.index).astype(int)

def safe_numeric_column(df: pd.DataFrame, name: str, fill=None) -> pd.Series:
    """Return a numeric Series aligned to df.index; create if absent."""
    if name in df.columns:
        s = pd.to_numeric(df[name], errors="coerce")
    else:
        s = pd.Series(np.nan, index=df.index, name=name)
    if fill is not None:
        s = s.fillna(fill)
    return s

def safe_categorical_column(df: pd.DataFrame, name: str, default="Unknown") -> pd.Series:
    """Return a string Series aligned to df.index; create if absent."""
    if name in df.columns:
        s = df[name].astype(str).fillna(default).replace("", default)
    else:
        s = pd.Series(default, index=df.index, name=name)
    return s

def ensure_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived geometry if missing; alias angle if needed."""
    d = df.copy()
    # Canonical angle: prefer angle_deg; alias angle_deg_signed -> angle_deg
    if "angle_deg" not in d.columns and "angle_deg_signed" in d.columns:
        d["angle_deg"] = d["angle_deg_signed"]

    # abs_angle
    if "abs_angle" not in d.columns and "angle_deg" in d.columns:
        d["abs_angle"] = pd.to_numeric(d["angle_deg"], errors="coerce").abs()

    # angle_x_distance
    if "angle_x_distance" not in d.columns and {"angle_deg","distance_m"}.issubset(d.columns):
        ang = pd.to_numeric(d["angle_deg"], errors="coerce")
        dist = pd.to_numeric(d["distance_m"], errors="coerce")
        d["angle_x_distance"] = ang * dist

    return d

def build_design(df_in: pd.DataFrame, y: np.ndarray, folds=5):
    """Return (df_prepared, groups, ColumnTransformer, num_cols, cat_cols) for the pipeline."""
    df = ensure_geometry(df_in)

    # -------- Feature sets (Phase I final) --------
    base_num = [
        "distance_m","angle_deg","abs_angle","angle_x_distance",
        "defender_count","goalie_distance_m","possession_passes",
        "shooter_x","shooter_y"
    ]
    # In-game priors (may be absent; we’ll default to 0.0)
    prior_num = [
        "poss_idx_in_game","prior_poss_before","prior_shots_before","prior_goals_before",
        "prior_shot_rate_in_game","prior_goal_rate_in_game",
        "last3_shots_before","last3_goals_before","last3_goal_rate_in_game"
    ]

    # Numeric coercion (robust series creation)
    for c in base_num:
        df[c] = safe_numeric_column(df, c, fill=None)  # keep NaN; imputer handles it
    for c in prior_num:
        df[c] = safe_numeric_column(df, c, fill=0.0)   # priors default to 0

    # Booleans → ints (with safe default)
    for b in ["is_man_up","empty_net","is_heave","is_long_range"]:
        df[b] = map_bool(df.get(b, 0))

    # Categorical features (robust defaults)
    cat_all = ["goalie_lateral","attack_type","shot_type","shooter_handedness",
               "opponent_team_level","our_team_level","distance_bin","angle_bin",
               "defender_count_bin","dist_x_manup","dist_x_defbin","clock_bin",
               "opponent_team_name","our_team_name","game_id"]
    for c in cat_all:
        df[c] = safe_categorical_column(df, c, default="Unknown")

    # Final column groups for transformer
    cat_cols = ["goalie_lateral","attack_type","shot_type","shooter_handedness",
                "opponent_team_level","distance_bin","angle_bin",
                "defender_count_bin","dist_x_manup_model","dist_x_defbin_model","clock_bin"]
    # If *_model versions missing, alias from non-model versions
    if "dist_x_manup_model" not in df.columns and "dist_x_manup" in df.columns:
        df["dist_x_manup_model"] = df["dist_x_manup"]
    if "dist_x_defbin_model" not in df.columns and "dist_x_defbin" in df.columns:
        df["dist_x_defbin_model"] = df["dist_x_defbin"]
    # Ensure they exist even if both were absent
    for c in ["dist_x_manup_model","dist_x_defbin_model"]:
        if c not in df.columns:
            df[c] = "Unknown"

    num_with_bools = base_num + prior_num + ["is_man_up","empty_net","is_heave","is_long_range","opp_oof_goal_rate"]
    if "opp_oof_goal_rate" not in df.columns:
        df["opp_oof_goal_rate"] = 0.0

    # ===== Pipelines with IMPUTATION =====
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_with_bools),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Groups for CV (prefer game_id; else stratified folds)
    groups = df["game_id"] if "game_id" in df.columns else pd.Series("all", index=df.index)

    return df, groups, pre, num_with_bools, cat_cols

def metrics_dict(y_true, p_raw, p_cal):
    auc_raw = roc_auc_score(y_true, p_raw) if len(np.unique(y_true)) > 1 else float("nan")
    auc_cal = roc_auc_score(y_true, p_cal) if len(np.unique(y_true)) > 1 else float("nan")
    brier   = brier_score_loss(y_true, p_cal)
    try:
        ll = log_loss(y_true, np.clip(p_cal, 1e-6, 1 - 1e-6))
    except Exception:
        ll = float("nan")
    return {"auc_raw": round(float(auc_raw), 4),
            "auc_cal": round(float(auc_cal), 4),
            "brier": round(float(brier), 4),
            "logloss": round(float(ll), 4)}

# ----------------------------- Main -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="FINAL: LogReg (C=0.3) with OOF isotonic calibration and final exports")
    ap.add_argument("--input", default="app/features_shots.csv")
    ap.add_argument("--out_dir", default="app/models")
    ap.add_argument("--reports_dir", default="app/reports")
    ap.add_argument("--curation", default=None, help="Optional curation CSV with weights or flags")
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

    # Optional sample weights (curation / downweighting)
    sample_weight = np.ones(len(df), dtype=float)
    if args.curation and Path(args.curation).exists():
        cur = pd.read_csv(args.curation, dtype=str)
        # Example: if cur has columns ["possession_id","weight"]
        if "possession_id" in cur.columns and "weight" in cur.columns and "possession_id" in df.columns:
            w_map = dict(zip(cur["possession_id"], pd.to_numeric(cur["weight"], errors="coerce").fillna(1.0)))
            sample_weight = np.array([w_map.get(pid, 1.0) for pid in df["possession_id"]], dtype=float)

    # Optional: extra downweight long-range
    if args.downweight_long_range is not None and "distance_m" in df.columns:
        d = pd.to_numeric(df["distance_m"], errors="coerce").fillna(0.0).values
        mask = d >= 10.0
        sample_weight[mask] = sample_weight[mask] * float(args.downweight_long_range)

    # Build design / transformer
    df_prep, groups, pre, num_cols, cat_cols = build_design(df, y, folds=args.folds)

    # Splitter
    if "game_id" in df_prep.columns:
        gkf = GroupKFold(n_splits=max(2, min(args.folds, len(pd.unique(groups)))))
        split_iter = gkf.split(df_prep, y, groups=groups)
    else:
        skf = StratifiedKFold(n_splits=max(2, args.folds), shuffle=True, random_state=42)
        split_iter = skf.split(df_prep, y)

    # Model
    base_clf = LogisticRegression(C=0.3, solver="liblinear", max_iter=200)

    # OOF preds (raw)
    oof_raw = np.zeros(len(df_prep), dtype=float)
    fold_ids = np.zeros(len(df_prep), dtype=int)
    models = []

    X_all = df_prep  # ColumnTransformer handles selection + preprocessing

    for fidx, (tr, va) in enumerate(split_iter, start=1):
        Xtr = X_all.iloc[tr].copy()
        Xva = X_all.iloc[va].copy()
        ytr = y[tr]
        sw  = sample_weight[tr] if sample_weight is not None else None

        # Fit preprocessing on train only
        pre.fit(Xtr)
        Xtr_enc = pre.transform(Xtr)
        Xva_enc = pre.transform(Xva)

        clf = LogisticRegression(C=0.3, solver="liblinear", max_iter=200)
        clf.fit(Xtr_enc, ytr, sample_weight=sw)

        pva = clf.predict_proba(Xva_enc)[:, 1]
        oof_raw[va] = pva
        fold_ids[va] = fidx
        models.append((pre, clf))

    # Global isotonic on OOF
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_raw, y)

    # Fit FINAL model on all data
    pre.fit(X_all)
    X_all_enc = pre.transform(X_all)
    final_clf = LogisticRegression(C=0.3, solver="liblinear", max_iter=200)
    final_clf.fit(X_all_enc, y, sample_weight=sample_weight)

    # Final predictions (raw + calibrated)
    p_raw_all = final_clf.predict_proba(X_all_enc)[:, 1]
    p_cal_all = iso.transform(p_raw_all)

    # ---------------- Exports ----------------
    # 1) Metrics
    met = metrics_dict(y, p_raw_all, p_cal_all)
    metrics_row = {
        "auc_raw": met["auc_raw"],
        "auc_cal": met["auc_cal"],
        "brier": met["brier"],
        "logloss": met["logloss"],
        "n_rows": len(df_prep),
        "model": "logreg_C0.3_iso_oof",
        "folds": args.folds,
    }
    pd.DataFrame([metrics_row]).to_csv(reports_dir / "metrics_validation_final.csv", index=False)
    (reports_dir / "metrics_validation_final.json").write_text(json.dumps(metrics_row, indent=2))

    # 2) By-shot table (calibrated)
    by_shot_cols = ["possession_id","game_id","period","time_remaining","player_number",
                    "distance_m","angle_deg","defender_count","is_man_up","empty_net",
                    "shot_type","attack_type","shooter_handedness"]
    present_cols = [c for c in by_shot_cols if c in df_prep.columns]
    shot_tbl = df_prep[present_cols].copy()
    shot_tbl["y"] = y
    shot_tbl["xg_raw"] = p_raw_all
    shot_tbl["xg"] = p_cal_all
    shot_tbl.to_csv(reports_dir / "xg_by_shot_all_calibrated_logreg_final.csv", index=False)

    # 3) By-game table (calibrated)
    if "game_id" in df_prep.columns:
        g = df_prep.copy()
        g["xg"] = p_cal_all
        g["y"] = y
        by_game = g.groupby("game_id", dropna=False).agg(
            shots=("y","size"),
            goals=("y","sum"),
            xg=("xg","sum"),
            xg_per_shot=("xg","mean"),
        ).reset_index()
        by_game.to_csv(reports_dir / "xg_by_game_all_calibrated_logreg_final.csv", index=False)
    else:
        pd.DataFrame(columns=["game_id","shots","goals","xg","xg_per_shot"]).to_csv(
            reports_dir / "xg_by_game_all_calibrated_logreg_final.csv", index=False
        )

    # 4) Model coefficients (best-effort)
    try:
        # Extract numeric col names (as passed to transformer)
        num_feat_names = list(pre.transformers_[0][2])  # num_with_bools
        # Extract OHE names
        ohe = pre.transformers_[1][1].named_steps["ohe"]
        cat_base = pre.transformers_[1][2]
        cat_feat_names = list(ohe.get_feature_names_out(cat_base))
        feat_names = num_feat_names + cat_feat_names
        coefs = pd.DataFrame({"feature": feat_names, "coef": final_clf.coef_.ravel()[:len(feat_names)]})
        coefs.to_csv(out_dir / "logreg_final_coefficients.csv", index=False)
    except Exception:
        pass

    print("✓ wrote reports →", (reports_dir / "xg_by_shot_all_calibrated_logreg_final.csv").resolve())

if __name__ == "__main__":
    main()
