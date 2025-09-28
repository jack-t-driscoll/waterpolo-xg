# tools/backfill_levels.py
import pandas as pd
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="app/shots.csv")
ap.add_argument("--output", default="app/shots.csv", help="Overwrite by default")
args = ap.parse_args()

p = Path(args.input)
df = pd.read_csv(p, dtype=str)

for col in ["our_team_level","opponent_team_level","game_id"]:
    if col not in df.columns:
        df[col] = ""

# Per-game mode fill for missing levels
for lvl_col in ["our_team_level","opponent_team_level"]:
    def fill_group(g):
        mode = g[lvl_col][g[lvl_col].notna() & (g[lvl_col].str.strip()!="")].mode()
        if mode.empty: 
            return g[lvl_col]  # nothing to fill
        m = mode.iloc[0]
        return g[lvl_col].where(g[lvl_col].astype(str).str.strip()!="", m)
    df[lvl_col] = df.groupby("game_id", group_keys=False).apply(fill_group)

df.to_csv(args.output, index=False)
print("Backfilled levels to", Path(args.output).resolve())
