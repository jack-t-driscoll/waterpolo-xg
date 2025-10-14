from pathlib import Path
import pandas as pd

BASE = Path("app/reports/corrections")
SRC  = BASE / "shot_xy_overrides.csv"          # your manual tags live here
OUT  = BASE / "corrections_shot_xy.csv"        # exporter-friendly, shot-XY only

BASE.mkdir(parents=True, exist_ok=True)

# If there are no overrides yet, just create an empty, correctly-schemed file
if not SRC.exists():
    pd.DataFrame(columns=["possession_id","field","value"]).to_csv(OUT, index=False)
    print(f"? wrote empty file (no overrides found) ? {OUT}")
    raise SystemExit(0)

df = pd.read_csv(SRC, dtype=str)
need = {"possession_id","x_m","y_m"}
missing = need - set(df.columns)
if missing:
    raise SystemExit(f"shot_xy_overrides.csv is missing columns: {missing}. "
                     f"Expected columns: {sorted(need)}")

rows = []
for _, r in df.iterrows():
    pid = r["possession_id"]
    x   = r["x_m"]
    y   = r["y_m"]
    rows.append({"possession_id": pid, "field": "shooter_x", "value": str(x)})
    rows.append({"possession_id": pid, "field": "shooter_y", "value": str(y)})

out_df = pd.DataFrame(rows, columns=["possession_id","field","value"])

# If OUT already exists, replace same (pid, field) with latest values
if OUT.exists():
    old = pd.read_csv(OUT, dtype=str)
    key = ["possession_id","field"]
    old = old[~old.set_index(key).index.isin(out_df.set_index(key).index)]
    out_df = pd.concat([old, out_df], ignore_index=True)

out_df.to_csv(OUT, index=False)
print(f"? wrote shot-XY corrections ? {OUT}  (rows={len(out_df)})")
