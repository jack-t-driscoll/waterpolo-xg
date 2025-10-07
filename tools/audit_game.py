import argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game_id", required=True, type=str)
    ap.add_argument("--audit", default="app/reports/homography/propagation_audit_multi.csv")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.audit, dtype=str)
    except FileNotFoundError:
        raise SystemExit(f"Audit file not found: {args.audit}")

    g = df[df["game_id"].astype(str) == args.game_id]
    print(f"Game {args.game_id} rows: {len(g)}")
    if g.empty:
        return

    cols = [c for c in ["video_file","status","inliers","good","from_anchor"] if c in g.columns]
    print(g[cols].to_string(index=False))

    fails = g[g["status"].str.lower() == "fail"]["video_file"].tolist()
    print("\nFailures:", fails if fails else "None")

if __name__ == "__main__":
    main()
