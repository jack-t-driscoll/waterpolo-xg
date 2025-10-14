import argparse, pandas as pd, numpy as np
from pathlib import Path

def map_bool(x):
    s = str(x).strip().lower()
    return 1 if s in {"1", "true", "t", "yes", "y"} else 0

def derive_distance_bin(distance_m: pd.Series) -> pd.Series:
    # Fallback bins if distance_bin is missing. Matches typical coarse buckets.
    x = pd.to_numeric(distance_m, errors="coerce")
    bins = [-0.001, 2, 4, 6, 8, 10, 15, np.inf]
    labels = ["0-2","2-4","4-6","6-8","8-10","10-15","15+"]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True).astype("category")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None, help="If omitted, overwrite input")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp

    df = pd.read_csv(inp, dtype=str)

    # distance_bin_model: prefer existing distance_bin, else derive from distance_m
    if "distance_bin" in df.columns:
        df["distance_bin_model"] = df["distance_bin"].fillna("Unknown").replace("", "Unknown")
    else:
        db = derive_distance_bin(df.get("distance_m"))
        df["distance_bin_model"] = db.astype(str).fillna("Unknown").replace("", "Unknown")

    # crosses used by downstream reports (trainer also recreates, but this keeps CSV consistent)
    mu = df.get("is_man_up", "False").map(map_bool)
    df["dist_x_manup_model"] = df["distance_bin_model"].astype(str) + "|MU" + mu.astype(str)

    dcb = df.get("defender_count_bin")
    if dcb is None:
        # fallback: coarse defender count bins from defender_count if needed
        dc = pd.to_numeric(df.get("defender_count"), errors="coerce")
        cats = pd.cut(dc, bins=[-1,0,1,2,3,99], labels=["0","1","2","3","4+"], include_lowest=True)
        dcb = cats.astype(str).fillna("Unknown").replace("", "Unknown")
    df["dist_x_defbin_model"] = df["distance_bin_model"].astype(str) + "|" + dcb.astype(str)

    df.to_csv(out, index=False)
    print(f"✓ Wrote {out}")

if __name__ == "__main__":
    main()
