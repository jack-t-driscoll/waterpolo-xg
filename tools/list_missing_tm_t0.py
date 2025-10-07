# tools/list_missing_tm_t0.py
from pathlib import Path
import pandas as pd

APP = Path("app")
MAN = APP / "reports" / "frames" / "frames_manifest.csv"
SHOTS = APP / "shots.csv"

shots = pd.read_csv(SHOTS, dtype=str)
shots = shots[shots.get("event_type","").astype(str).str.lower()=="shot"].copy()
shot_ids = shots["possession_id"].astype(str)

man = pd.read_csv(MAN, dtype=str)
pivot = man.pivot_table(index="possession_id", columns="context", values="frame_path", aggfunc="first")

# 1) Shots that have NO manifest row at all
absent = shot_ids[~shot_ids.isin(pivot.index.astype(str))].tolist()

# 2) Shots present in manifest but missing tm/t0 cells
need_tm = []
need_t0 = []
for pid in shot_ids:
    if pid in pivot.index:
        row = pivot.loc[pid]
        if ("tm" not in row.index) or pd.isna(row.get("tm")):
            need_tm.append(pid)
        if ("t0" not in row.index) or pd.isna(row.get("t0")):
            need_t0.append(pid)

print("Absent from manifest:", absent)
print("Present but missing tm:", need_tm)
print("Present but missing t0:", need_t0)
