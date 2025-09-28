# tools/summarize_calibration.py
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Summarize per-game xG vs goals and flag outliers.")
    ap.add_argument("--by_game", default="app/reports/xg_by_game_all_calibrated_hgb.csv")
    ap.add_argument("--out", default="app/reports/xg_by_game_summary.csv")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.by_game)
    if not {"game_id","shots","xg_sum","goals"}.issubset(df.columns):
        raise SystemExit("by_game file missing required columns")

    df["residual"] = df["goals"] - df["xg_sum"]
    df["xg_rate"] = df["xg_sum"] / df["shots"]
    df["goals_rate"] = df["goals"] / df["shots"]

    overall = pd.DataFrame([{
        "games": len(df),
        "shots_total": int(df["shots"].sum()),
        "goals_total": int(df["goals"].sum()),
        "xg_total": float(df["xg_sum"].sum()),
        "goals_rate_overall": float(df["goals"].sum() / max(1, df["shots"].sum())),
        "xg_rate_overall": float(df["xg_sum"].sum() / max(1, df["shots"].sum())),
        "mae_goals": float((df["residual"].abs()).mean()),
        "rmse_goals": float((df["residual"]**2).mean() ** 0.5),
    }])

    top_under = df.nsmallest(args.topk, "residual")
    top_over  = df.nlargest(args.topk, "residual")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    top_under.to_csv(out_dir / "xg_by_game_top_under.csv", index=False)
    top_over.to_csv(out_dir / "xg_by_game_top_over.csv", index=False)
    overall.to_csv(out_dir / "xg_by_game_overall.csv", index=False)

    print("✓ Wrote:")
    print(" ", Path(args.out).resolve())
    print(" ", (out_dir / 'xg_by_game_top_under.csv').resolve())
    print(" ", (out_dir / 'xg_by_game_top_over.csv').resolve())
    print(" ", (out_dir / 'xg_by_game_overall.csv').resolve())

if __name__ == "__main__":
    main()
