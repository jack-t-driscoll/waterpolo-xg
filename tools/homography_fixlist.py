from pathlib import Path
import pandas as pd

AUD = Path("app/reports/homography/propagation_audit.csv")
OUT = Path("app/reports/homography/fixlist.csv")

if not AUD.exists():
    raise SystemExit(f"Missing audit: {AUD}")

a = pd.read_csv(AUD, dtype=str)
fails = a[a["status"].str.contains("match FAIL", na=False)].copy()

def to_int(x):
    try: return int(x)
    except: return -1

fails["good_int"] = fails["good"].map(to_int)
fails = fails.sort_values(["game_id","good_int"])

# Save full failure list
OUT.parent.mkdir(parents=True, exist_ok=True)
fails.to_csv(OUT, index=False)
print(f"✓ Wrote {OUT.resolve()}")

# Also print a small, actionable view (top 5 per game)
print("\n=== Failures to consider calibrating (top 5 per game) ===")
for gid, g in fails.groupby("game_id"):
    g5 = g.head(5)
    print(f"[game {gid}] {len(g)} failures, top 5:")
    for _, r in g5.iterrows():
        print(f"  - {r['video_file']}: {r['status']} (good={r['good']}, inliers={r['inliers']})")
