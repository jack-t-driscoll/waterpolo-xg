import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression

oof = pd.read_csv("app/reports/xg_oof_preds_logreg_final.csv")  # columns: possession_id, game_id, xg, goal
iso = IsotonicRegression(out_of_bounds="clip").fit(oof["xg"].values, oof["goal"].values)

by = pd.read_csv("app/reports/xg_by_shot_all_calibrated_logreg_final.csv")
raw = by.get("xg_raw", by.get("xg"))  # fallback if xg_raw missing
by["xg"] = iso.predict(raw.values)    # overwrite with smoother global-calibrated xg
by.to_csv("app/reports/xg_by_shot_CAL_global.csv", index=False)
print("? wrote app/reports/xg_by_shot_CAL_global.csv")
