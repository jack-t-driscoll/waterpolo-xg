import argparse
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

def map_bool(s):
    if isinstance(s, pd.Series):
        return s.map({True:1, False:0, "true":1, "false":0, 1:1, 0:0}).fillna(0).astype(int)
    return 0

def main():
    ap = argparse.ArgumentParser(description="Cross-validated logistic regression (diagnostic)")
    ap.add_argument("--input", default="app/features_shots.csv")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str)
    if "shot_result" not in df.columns:
        raise SystemExit("features_shots.csv missing 'shot_result'")
    y = (df["shot_result"].str.lower() == "goal").astype(int).values

    # numerics
    num = ["distance_m","angle_deg","defender_count","goalie_distance_m","possession_passes","shooter_x","shooter_y"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    # derived
    df["abs_angle"] = df["angle_deg"].abs()
    df["angle_x_distance"] = df["abs_angle"] * df["distance_m"]
    base_num = num + ["abs_angle","angle_x_distance"]

    # cats
    for c in ["goalie_lateral","attack_type","shot_type","shooter_handedness",
              "opponent_team_level","distance_bin","angle_bin",
              "defender_count_bin","dist_x_manup","dist_x_defbin","clock_bin","game_id"]:
        df[c] = df.get(c, "Unknown")
        if not isinstance(df[c], pd.Series):
            df[c] = pd.Series(["Unknown"] * len(df), index=df.index)
        df[c] = df[c].fillna("Unknown").replace("", "Unknown").astype(str)

    # groups
    groups = df["game_id"].astype(str) if "game_id" in df.columns else pd.Series(["g0"]*len(df))

    # preprocessors
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([
        ("num", numeric_pipeline, base_num),
        ("cat", categorical_pipeline, ["goalie_lateral","attack_type","shot_type","shooter_handedness",
                                       "opponent_team_level","distance_bin","angle_bin",
                                       "defender_count_bin","dist_x_manup","dist_x_defbin","clock_bin"]),
    ])

    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=0.3))])

    gkf = GroupKFold(n_splits=min(args.folds, max(2, groups.nunique())))
    aucs, loglosses, briers = [], [], []
    for tr, va in gkf.split(df, y, groups):
        pipe = Pipeline(model.steps)
        pipe.fit(df.iloc[tr], y[tr])
        p = pipe.predict_proba(df.iloc[va])[:,1]
        aucs.append(roc_auc_score(y[va], p))
        loglosses.append(log_loss(y[va], p, labels=[0,1]))
        briers.append(brier_score_loss(y[va], p))
    print("CV AUC:", np.mean(aucs), "±", np.std(aucs))
    print("CV LogLoss:", np.mean(loglosses))
    print("CV Brier:", np.mean(briers))

if __name__ == "__main__":
    main()
