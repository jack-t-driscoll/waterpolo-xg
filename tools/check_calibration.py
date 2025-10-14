import pandas as pd

# Load
by_shot = pd.read_csv("app/reports/xg_by_shot_all_calibrated_logreg_final.csv")
feat = pd.read_csv("app/features_shots.csv", dtype=str)

# One-and-only xg column (prefer calibrated)
if "xg_cal" in by_shot.columns:
    bx = by_shot[["possession_id","xg_cal"]].rename(columns={"xg_cal":"xg"})
elif "xg" in by_shot.columns:
    bx = by_shot[["possession_id","xg"]].copy()
else:
    raise SystemExit(f"by_shot has no xg/xg_cal. Columns: {list(by_shot.columns)}")

# Binary goal
feat["goal"] = (feat["shot_result"].str.lower() == "goal").astype(int)

# Merge (ensures only a single 'xg' column exists)
df = bx.merge(feat[["possession_id","goal"]], on="possession_id", how="left")

print(f"Rows: {len(df)} | missing xg: {df['xg'].isna().sum()} | missing goal: {df['goal'].isna().sum()}")

# Coerce and drop missing
df["xg"] = pd.to_numeric(df["xg"], errors="coerce")
df = df.dropna(subset=["xg","goal"])

# Deciles
df["decile"] = pd.qcut(df["xg"], 10, labels=False, duplicates="drop")
g = df.groupby("decile", dropna=False).agg(
    n=("goal","size"),
    pred=("xg","mean"),
    obs=("goal","mean")
).sort_index()

g["pred"] = g["pred"].round(3)
g["obs"] = g["obs"].round(3)
print("\nCalibration by decile (xg vs. goal rate):")
print(g.to_string())
