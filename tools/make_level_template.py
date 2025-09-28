# tools/make_level_template.py
import pandas as pd
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="app/shots.csv")
ap.add_argument("--output", default="app/team_levels_template.csv")
args = ap.parse_args()

df = pd.read_csv(args.input, dtype=str)
for c in ["game_id","our_team_name","opponent_team_name"]:
    if c not in df.columns:
        df[c] = ""

tpl = (
    df.groupby("game_id", as_index=False)[["our_team_name","opponent_team_name"]]
      .agg(lambda s: s.dropna().astype(str).str.strip().replace("", pd.NA).mode().iloc[0]
           if not s.dropna().astype(str).str.strip().replace("", pd.NA).mode().empty else "")
      .sort_values("game_id")
)
tpl["our_team_level"] = ""        # fill me
tpl["opponent_team_level"] = ""   # fill me

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
tpl.to_csv(args.output, index=False)
print("Wrote template:", Path(args.output).resolve())
