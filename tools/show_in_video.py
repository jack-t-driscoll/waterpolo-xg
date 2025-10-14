import pandas as pd
df = pd.read_csv("app/shots.csv", dtype=str)
print(df[df["video_file"]=="IMG_5045.MOV"][["possession_id","video_timestamp_mmss","player_number","shot_result"]].to_string(index=False))
