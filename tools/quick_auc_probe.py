# tools/quick_auc_probe.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def engineer_geometry(df: pd.DataFrame):
    # abs_angle
    if "angle_deg" in df.columns and "abs_angle" not in df.columns:
        try:
            ang = pd.to_numeric(df["angle_deg"], errors="coerce")
            df["abs_angle"] = ang.abs()
        except Exception:
            df["abs_angle"] = np.nan

    # angle_x_distance
    if "angle_x_distance" not in df.columns:
        if "abs_angle" in df.columns and "distance_m" in df.columns:
            try:
                a = pd.to_numeric(df["abs_angle"], errors="coerce")
                d = pd.to_numeric(df["distance_m"], errors="coerce")
                df["angle_x_distance"] = a * d
            except Exception:
                df["angle_x_distance"] = np.nan
        else:
            df["angle_x_distance"] = np.nan


def present(df: pd.DataFrame, candidates):
    """Return the subset of candidate columns that actually exist in df."""
    return [c for c in candidates if c in df.columns]


def make_pipe(df: pd.DataFrame, use_geom=True, use_cat=True, C=0.3):
    # Candidate feature sets
    geom_candidates = [
        "distance_m", "angle_deg", "abs_angle", "angle_x_distance",
        "defender_count", "goalie_distance_m", "possession_passes",
        "shooter_x", "shooter_y",
    ]
    cat_candidates = [
        "goalie_lateral", "attack_type", "shot_type", "shooter_handedness",
        "opponent_team_level", "distance_bin", "angle_bin",
        "defender_count_bin", "clock_bin",
    ]

    # Filter by presence
    num_cols = present(df, geom_candidates) if use_geom else []
    cat_cols = present(df, cat_candidates) if use_cat else []

    # If nothing selected, raise a friendly error
    if not num_cols and not cat_cols:
        raise SystemExit("No usable columns found for this probe (all missing).")

    # Pipelines
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    # Column transformer using only present columns
    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipeline, cat_cols))

    pre = ColumnTransformer(transformers)

    # Model
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=C))])
    return model


def cv_auc(model, X: pd.DataFrame, y: np.ndarray, groups: pd.Series, n_splits=5):
    # Ensure at least 2 splits (in case of few games)
    n_splits = max(2, min(n_splits, max(2, int(pd.Series(groups).nunique()))))
    gkf = GroupKFold(n_splits=n_splits)

    aucs = []
    for tr, va in gkf.split(X, y, groups):
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
    aucs = np.array(aucs, dtype=float)
    return float(aucs.mean()), float(aucs.std())


def main():
    ap = argparse.ArgumentParser(description="Quick AUC probe: geom vs cats")
    ap.add_argument("--input", default="app/features_shots.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str)

    # Label
    if "shot_result" not in df.columns:
        raise SystemExit("features_shots.csv missing 'shot_result'")
    y = (df["shot_result"].str.lower() == "goal").astype(int).values

    # Groups
    if "game_id" not in df.columns:
        raise SystemExit("features_shots.csv missing 'game_id'")
    g = df["game_id"].astype(str)

    # Prep numerics + engineered geom
    ensure_numeric(df, ["angle_deg", "distance_m", "defender_count",
                        "goalie_distance_m", "possession_passes",
                        "shooter_x", "shooter_y"])
    engineer_geometry(df)

    # Build probes
    mA = make_pipe(df.copy(), use_geom=True,  use_cat=True,  C=0.3)  # geom+cats
    mB = make_pipe(df.copy(), use_geom=True,  use_cat=False, C=0.3)  # geom-only
    mC = make_pipe(df.copy(), use_geom=False, use_cat=True,  C=0.3)  # cats-only
    mD = make_pipe(df.copy(), use_geom=True,  use_cat=True,  C=1.0)  # C sensitivity

    a_mu, a_sd = cv_auc(mA, df, y, g)
    b_mu, b_sd = cv_auc(mB, df, y, g)
    c_mu, c_sd = cv_auc(mC, df, y, g)
    d_mu, d_sd = cv_auc(mD, df, y, g)

    print(f"A) geom+cats (C=0.3): AUC={a_mu:.3f} ± {a_sd:.3f}")
    print(f"B) geom-only (C=0.3): AUC={b_mu:.3f} ± {b_sd:.3f}")
    print(f"C) cats-only (C=0.3): AUC={c_mu:.3f} ± {c_sd:.3f}")
    print(f"D) geom+cats (C=1.0): AUC={d_mu:.3f} ± {d_sd:.3f}")


if __name__ == "__main__":
    main()
