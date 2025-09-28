# tools/find_review_candidates.py
import argparse
from pathlib import Path
import pandas as pd

# --- Constants / sets ---
MAN_UP_STATES_OUR = {"6v5", "6v4"}  # our numerical advantage (exclusions)
PULLED_GOALIE_STATES_OUR = {"7v6 (pulled goalie/empty net)"}
PULLED_GOALIE_STATES_THEIRS = {"6v7 (opponent pulled goalie/empty net)"}

def mmss_to_seconds(s):
    try:
        mm, ss = str(s).split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return None

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    # Ensure required columns exist
    need = [
        "possession_id","game_id","event_type","attack_type","man_state",
        "shot_type","shot_result","turnover_type","turnover_player_number",
        "video_file","video_timestamp_mmss","goalie_present",
    ]
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA

    # Pre-compute seconds; normalize a few strings
    df["t_seconds"] = df["video_timestamp_mmss"].apply(mmss_to_seconds)
    for c in ["event_type","attack_type","man_state","shot_type","turnover_type","goalie_present"]:
        df[c] = df[c].astype(str).str.strip()

    return df

def flag_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    reasons = []

    # A) Tactic says man-up but counts don't (and vice versa)
    mask_tactic_manup = df["attack_type"].str.lower() == "man-up"
    mask_counts_manup = df["man_state"].isin(MAN_UP_STATES_OUR)
    reasons.append(("tactic_without_counts", mask_tactic_manup & ~mask_counts_manup))
    reasons.append(("counts_without_tactic", mask_counts_manup & ~mask_tactic_manup))

    # B) Penalty shot but no 5m_drawn nearby (≤45s earlier, same game+file)
    df_sorted = df.sort_values(["game_id","video_file","t_seconds","possession_id"], kind="stable").reset_index()
    last_draw_time = {}
    penalty_without_drawn = pd.Series(False, index=df.index)
    for _, row in df_sorted.iterrows():
        key = (row["game_id"], row["video_file"])
        ts = row["t_seconds"]
        et = (row["event_type"] or "").lower()
        if et == "5m_drawn":
            last_draw_time[key] = ts
            continue
        if et == "shot" and (row["shot_type"] or "").lower() == "5m penalty":
            ok = False
            if key in last_draw_time and ts is not None and last_draw_time[key] is not None:
                dt = ts - last_draw_time[key]
                ok = (0 <= dt <= 45)
            penalty_without_drawn.at[row["index"]] = not ok
    reasons.append(("penalty_without_drawn", penalty_without_drawn))

    # C) Ejection drawn but no man-up possession soon after (≤45s ahead, same game+file)
    eject_no_follow = pd.Series(False, index=df.index)
    for i, row in df_sorted.iterrows():
        if (row["event_type"] or "").lower() != "ejection_drawn":
            continue
        g, vf, ts0 = row["game_id"], row["video_file"], row["t_seconds"]
        ok = False
        # Look ahead a handful of possessions within 45s
        for j in range(i+1, min(i+11, len(df_sorted))):
            r2 = df_sorted.iloc[j]
            if r2["game_id"] != g or r2["video_file"] != vf:
                continue
            if ts0 is not None and r2["t_seconds"] is not None and (r2["t_seconds"] - ts0) > 45:
                break
            if r2["man_state"] in MAN_UP_STATES_OUR:
                ok = True
                break
        if not ok:
            eject_no_follow.at[row["index"]] = True
    reasons.append(("ejection_without_followup_manup", eject_no_follow))

    # D) Turnover: shot clock violation has a player number (should be blank)  <-- FIXED
    if "turnover_player_number" in df.columns:
        ser = df["turnover_player_number"]
        has_player = ser.fillna("").astype(str).str.strip().ne("")
        scv = df["turnover_type"].str.lower() == "shot clock violation"
        reasons.append(("shot_clock_has_player_number", scv & has_player))

    # E) Duplicate timestamps within same clip (might be fine, but worth a look)
    dup = df.duplicated(subset=["game_id","video_file","video_timestamp_mmss"], keep=False)
    reasons.append(("duplicate_timestamp_same_clip", dup))

    # F) Empty-net sanity: goalie_present=false but counts not 7v6 or 6v7 (shots only)  <-- ROBUST + QUIET
    empty_net_flag = pd.Series(False, index=df.index)
    if "goalie_present" in df.columns:
        empty_net = df["goalie_present"].astype(str).str.lower() == "false"
        # Accept both 7v6 and 6v7, ignore any parenthetical text (non-capturing group silences warning)
        counts_allow_empty = df["man_state"].astype(str).str.contains(
            r"\b(?:7v6|6v7)\b", case=False, na=False, regex=True
        )
        shot_rows = (df["event_type"].str.lower() == "shot")
        empty_net_flag = empty_net & ~counts_allow_empty & shot_rows
    reasons.append(("empty_net_but_counts_not_7v6_or_6v7", empty_net_flag))

    # Build review queue
    any_flag = None
    for name, mask in reasons:
        any_flag = mask.copy() if any_flag is None else (any_flag | mask)
    review = df[any_flag].copy() if any_flag is not None else df.iloc[0:0].copy()

    # Attach reason tags
    def row_reasons(i):
        return "|".join([name for name, mask in reasons if bool(mask.loc[i])])
    review["reasons"] = [row_reasons(i) for i in review.index]

    # Columns to show
    cols = [
        "possession_id","game_id","event_type","attack_type","man_state",
        "shot_type","shot_result","turnover_type","video_file","video_timestamp_mmss"
    ]
    for c in cols:
        if c not in review.columns:
            review[c] = ""
    return review[["reasons"] + cols].sort_values(
        ["game_id","video_file","video_timestamp_mmss","possession_id"], kind="stable"
    )

def main():
    ap = argparse.ArgumentParser(description="Find likely review candidates in shots.csv")
    ap.add_argument("--input", default="shots.csv", help="Path to shots.csv")
    ap.add_argument("--output", default="review_queue.csv", help="Output CSV path")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")  # force .csv extension

    df = load_csv(in_path)
    review = flag_inconsistencies(df)
    review.to_csv(out_path, index=False)
    print(f"✓ Wrote {len(review)} candidates to {out_path.resolve()}")

if __name__ == "__main__":
    main()
