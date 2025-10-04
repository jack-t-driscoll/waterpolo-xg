import pandas as pd
f = pd.read_csv("app/features_shots.csv", dtype=str)
print("features_shots.csv columns:\n", list(f.columns))
keys = ["distance_m","angle_deg","defender_count","goalie_distance_m","possession_passes",
        "is_man_up","empty_net","shot_type","attack_type","shooter_handedness","man_state"]
for k in keys:
    if k in f.columns:
        print(f"{k:20s} present, nulls={f[k].isna().sum()}")
    else:
        print(f"{k:20s} MISSING")
