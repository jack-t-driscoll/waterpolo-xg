# tools/make_curation_template.py
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Create a curation template (three-level curation_tag).")
    ap.add_argument("--shots", default="app/shots.csv", help="Path to shots.csv")
    ap.add_argument("--output", default="app/curation_template.csv", help="Where to write the template CSV")
    args = ap.parse_args()

    shots_path = Path(args.shots)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(shots_path, dtype=str)
    for col in ["possession_id", "game_id"]:
        if col not in df.columns:
            raise SystemExit(f"shots.csv missing required column: {col}")

    cur = (
        df[["game_id", "possession_id"]]
        .drop_duplicates()
        .sort_values(["game_id", "possession_id"], kind="stable")
        .reset_index(drop=True)
    )

    # Three-level Option A: curation_tag in {organic, lightly_curated, highly_curated}
    # Leave blank → defaults to organic.
    cur["curation_tag"] = ""

    cur.to_csv(out_path, index=False)
    print(f"✓ Wrote template → {out_path.resolve()}")
    print("Fill 'curation_tag' with one of: organic | lightly_curated | highly_curated (blank → organic).")

if __name__ == "__main__":
    main()
