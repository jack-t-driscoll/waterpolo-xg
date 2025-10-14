import pandas as pd, numpy as np

by = pd.read_csv("app/reports/xg_by_shot_all_calibrated_logreg_final.csv")
fe = pd.read_csv("app/features_shots.csv", dtype=str)

x = by.get("xg_cal", by.get("xg")).rename("xg")
df = pd.concat([by["possession_id"], x], axis=1).merge(
    fe[["possession_id","shot_result"]], on="possession_id", how="left"
)

df["goal"] = (df["shot_result"].str.lower()=="goal").astype(int)
df["xg"] = pd.to_numeric(df["xg"], errors="coerce")
df = df.dropna(subset=["xg","goal"])

# Equal-width bins
bins = np.linspace(0.0, 1.0, 11)
df["bin"] = pd.cut(df["xg"], bins=bins, include_lowest=True)

g = df.groupby("bin", dropna=False).agg(
    n=("goal","size"), pred=("xg","mean"), obs=("goal","mean")
)
g["pred"] = g["pred"].round(3); g["obs"] = g["obs"].round(3)

print("Equal-width calibration (0.1 bins):")
print(g.to_string())

# Also show counts of each unique calibrated value (discreteness check)
vc = df["xg"].value_counts().sort_index()
print("\nUnique calibrated xg values and counts:")
print(vc.to_string())
