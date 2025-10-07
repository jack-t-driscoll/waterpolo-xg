import pandas as pd
shots = pd.read_csv("app/shots.csv", dtype=str)
shots = shots[shots["video_timestamp_mmss"].notna() & (shots["video_timestamp_mmss"].str.strip()!="")]
print("Rows with valid timestamps:", len(shots))

man = pd.read_csv("app/reports/frames/frames_manifest.csv", dtype=str)
print("Manifest rows:", len(man))
print(man["context"].value_counts(dropna=False).to_string())
