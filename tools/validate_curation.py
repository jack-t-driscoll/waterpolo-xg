# tools/validate_curation.py
import argparse
from pathlib import Path
import pandas as pd

VALID_TAGS = {"organic", "lightly_curated", "highly_curated"}
TAG_TO_WEIGHT = {
    "organic": 1.00,
    "lightly_curated": 0.70,
    "highly_curated": 0.33,
}

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_shots(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = normalize_headers(df)
    if "possession_id" not in df.columns:
        raise SystemExit("shots.csv missing 'possession_id'")
    return df

def load_curation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df = normalize_headers(df)
    if "possession_id" not in df.columns:
        raise SystemExit("curation file missing 'possession_id'")
    if "curation_tag" not in df.columns:
        raise SystemExit("curation file must have 'curation_tag'")
    return df

def map_weights(cur: pd.DataFrame) -> pd.DataFrame:
    cur = cur.copy()
    cur["possession_id"] = cur["possession_id"].astype(str)
    tag = cur["curation_tag"].astype(str).str.lower().str.strip()
    unknown = tag[~tag.isin(VALID_TAGS) & (tag != "")]
    if not unknown.empty:
        print("! Warning: unknown curation_tag values found:", sorted(unknown.unique().tolist()))
    # Map tags → weights; blank/NaN → organic (1.0)
    weight = tag.map(TAG_TO_WEIGHT).fillna(TAG_TO_WEIGHT["organic"])
    out = pd.DataFrame({"possession_id": cur["possession_id"], "sample_weight": weight})
    # Deduplicate, keep last
    out = out.drop_duplicates(subset=["possession_id"], keep="last")
    return out

def main():
    ap = argparse.ArgumentParser(description="Validate curation file and map three-level tags → weights.")
    ap.add_argument("--shots", default="app/shots.csv")
    ap.add_argument("--curation", default="app/curation.csv")
    ap.add_argument("--output", default="app/curation_resolved.csv", help="Write (possession_id, sample_weight)")
    args = ap.parse_args()

    shots = load_shots(Path(args.shots))
    cur = load_curation(Path(args.curation))
    resolved = map_weights(cur)

    # IDs not in shots
    shots_ids = set(shots["possession_id"].astype(str))
    cur_ids = set(resolved["possession_id"].astype(str))
    missing = sorted(cur_ids - shots_ids)
    if missing:
        print(f"! {len(missing)} curation possession_id(s) not found in shots.csv (showing up to 10):", missing[:10])

    # Summaries
    total_poss = shots["possession_id"].nunique()
    cur_poss = resolved["possession_id"].nunique()
    pct_covered = 100.0 * cur_poss / total_poss if total_poss else 0.0
    sw = pd.to_numeric(resolved["sample_weight"], errors="coerce").fillna(1.0)
    print(f"Shots possessions total: {total_poss}")
    print(f"Curation rows (unique possessions): {cur_poss} ({pct_covered:.1f}%)")
    print(f"Sample weight stats: min={sw.min():.2f}  max={sw.max():.2f}  mean={sw.mean():.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(out_path, index=False)
    print(f"✓ Wrote resolved weights → {out_path.resolve()}")
    print("Info: possessions not in this file will default to sample_weight=1.0 in the exporter.")

if __name__ == "__main__":
    main()
