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
    if s is None:
        return np.zeros(0, dtype=int)
    m = {True: 1, False: 0, "true": 1, "false": 0, "True": 1, "False": 0, 1: 1, 0: 0, "1": 1, "0": 0}
    return pd.Series(s).map(m).fillna(0).astype(int)

def build_preprocessor(df: pd.DataFrame):
    base_num = [
        "distance_m", "angle_deg", "abs_angle", "angle_x_distance",
        "defender_count", "goalie_distance_m", "possession_passes",
        "shooter_x", "shooter_y",
    ]
    for c in base_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

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

    for b in ["is_man_up", "empty_net", "is_heave", "is_long_range"]:
        df[b] = map_bool(df.get(b, 0))

    for c in ["goalie_lateral","attack_type","shot_type","shooter_handedness",
              "opponent_team_level","distance_bin","angle_bin","defender_count_bin"]:
        df[c] = df[c] if c in df.columns else "Unknown"
        df[c] = df[c].fillna("Unknown").replace("", "Unknown")

    df["distance_bin_model"] = df["distance_bin"].astype(str)

    cat_cols = [
        "goalie_lateral", "attack_type", "shot_type", "shooter_handedness",
        "opponent_team_level", "distance_bin_model", "angle_bin", "defender_count_bin",
    ]
    num_with_bools = base_num + prior_num + ["is_man_up","empty_net","is_heave","is_long_range"]

    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer([
        ("num", numeric_pipeline, num_with_bools),
        ("cat", categorical_pipeline, cat_cols)
    ])
    return pre

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="app/features_shots.csv")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype=str)
    y = (df["shot_result"].str.lower() == "goal").astype(int).values
    groups = df["game_id"].astype(str) if "game_id" in df.columns else pd.Series(["1"]*len(df))

    pre = build_preprocessor(df)
    clf = LogisticRegression(max_iter=1000, C=0.3)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    gkf = GroupKFold(n_splits=max(2, min(args.folds, df["game_id"].nunique())))
    aucs, losses, briers = [], [], []
    for tr, va in gkf.split(df, y, groups):
        pipe.fit(df.iloc[tr], y[tr])
        p = pipe.predict_proba(df.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p))
        losses.append(log_loss(y[va], p, labels=[0,1]))
        briers.append(brier_score_loss(y[va], p))

    print(f"CV AUC: {np.mean(aucs)} ± {np.std(aucs)}")
    print(f"CV LogLoss: {np.mean(losses)}")
    print(f"CV Brier: {np.mean(briers)}")

if __name__ == "__main__":
    main()
