# tools/compare_weighted_unweighted.py
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Compare unweighted vs weighted offense reports per game.")
    ap.add_argument("--dir", default="app", help="Directory where reports live")
    args = ap.parse_args()

    d = Path(args.dir)
    un = pd.read_csv(d / "offense_report_by_game.csv", dtype=str)
    wt = pd.read_csv(d / "offense_report_by_game_weighted.csv", dtype=str)

    for df in (un, wt):
        for c in df.columns:
            if c == "game_id":
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")

    merged = un.merge(wt, on="game_id", suffixes=("_un", "_wt"))
    merged["shot_rate_delta"] = merged["shot_rate_wt"] - merged["shot_rate_un"]
    merged["goal_rate_delta"] = merged["goal_rate_wt"] - merged["goal_rate_un"]
    merged["turnover_rate_delta"] = merged["turnover_rate_wt"] - merged["turnover_rate_un"]

    out = d / "offense_report_by_game_compare.csv"
    merged.to_csv(out, index=False)
    print("✓ Wrote comparison →", out.resolve())

if __name__ == "__main__":
    main()
