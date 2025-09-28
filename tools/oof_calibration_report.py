# tools/oof_calibration_report.py
# Build OOF reliability curve, pooling small bins for stability.
# Robust to different column names in OOF (e.g., xg/goal vs pred_prob/is_goal).

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def coerce_cols(df, pred_candidates=("pred_prob","xg","prob","p_goal"),
                     goal_candidates=("is_goal","goal","y","label")):
    pred_col = next((c for c in pred_candidates if c in df.columns), None)
    goal_col = next((c for c in goal_candidates if c in df.columns), None)
    if pred_col is None or goal_col is None:
        raise ValueError(f"Could not find prediction/goal columns in OOF. "
                         f"Have columns: {list(df.columns)}")
    out = df.copy()
    out["pred_prob"] = pd.to_numeric(out[pred_col], errors="coerce")
    out["is_goal"] = pd.to_numeric(out[goal_col], errors="coerce").fillna(0).astype(int)
    return out


def pooled_reliability(df, min_bin_size=10):
    # Bin by predicted probability
    df = df.dropna(subset=["pred_prob","is_goal"]).copy()
    df["prob_bin"] = pd.cut(df["pred_prob"], bins=np.linspace(0,1,11), include_lowest=True)

    # Raw table
    raw = df.groupby("prob_bin").agg(
        n=("is_goal","size"),
        pred_sum=("pred_prob","sum"),
        goal_sum=("is_goal","sum"),
    ).reset_index()

    # Pool neighboring small bins
    pooled_rows = []
    accum_n = 0
    accum_pred_sum = 0.0
    accum_goal_sum = 0.0
    accum_bins = []

    for _, r in raw.iterrows():
        accum_n += int(r["n"])
        accum_pred_sum += float(r["pred_sum"])
        accum_goal_sum += float(r["goal_sum"])
        accum_bins.append(str(r["prob_bin"]))

        if accum_n >= min_bin_size:
            pooled_rows.append({
                "bins": " + ".join(accum_bins),
                "n": accum_n,
                "pred_mean": accum_pred_sum / accum_n if accum_n > 0 else np.nan,
                "obs_rate": accum_goal_sum / accum_n if accum_n > 0 else np.nan,
            })
            accum_n = 0
            accum_pred_sum = 0.0
            accum_goal_sum = 0.0
            accum_bins = []

    # Tail (if any)
    if accum_n > 0:
        pooled_rows.append({
            "bins": " + ".join(accum_bins),
            "n": accum_n,
            "pred_mean": accum_pred_sum / accum_n if accum_n > 0 else np.nan,
            "obs_rate": accum_goal_sum / accum_n if accum_n > 0 else np.nan,
        })

    rel = pd.DataFrame(pooled_rows)
    rel["gap"] = rel["obs_rate"] - rel["pred_mean"]
    rel["abs_gap"] = rel["gap"].abs()
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)  # kept for future stratified views
    ap.add_argument("--oof", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_bin_size", type=int, default=10)
    args = ap.parse_args()

    # load
    feats = pd.read_csv(args.features)  # not used in this basic reliability, but kept available
    oof = pd.read_csv(args.oof)

    oof_std = coerce_cols(oof)
    rel = pooled_reliability(oof_std, min_bin_size=args.min_bin_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel.to_csv(out_dir / "oof_reliability_curve.csv", index=False)
    print(f"✓ Wrote pooled reliability curve → {out_dir/'oof_reliability_curve.csv'}")


if __name__ == "__main__":
    main()
