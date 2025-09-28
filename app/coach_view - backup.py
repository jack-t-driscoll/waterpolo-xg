# app/coach_view.py
# Coach summaries viewer with "Low data" badges, per-tab filters (unique keys), and a Model Card tab.

from pathlib import Path
import pandas as pd
import streamlit as st

REPORT_DIR = Path(__file__).resolve().parent / "reports" / "coach"
MODEL_CARD = Path(__file__).resolve().parent / "reports" / "model_card.md"

PAGES = {
    "Overall": "coach_overall.csv",
    "By Game": "coach_by_game.csv",
    "By Distance": "coach_by_distance.csv",
    "By Angle": "coach_by_angle.csv",
    "By Man State": "coach_by_manup.csv",
    "By Attack Type": "coach_by_attack_type.csv",
    "By Defender Count": "coach_by_defenders.csv",
    "By Opponent Level": "coach_by_opponent_level.csv",
    "Shooters": "coach_shooters.csv",
}

def load_csv(name: str) -> pd.DataFrame | None:
    fp = REPORT_DIR / name
    if not fp.exists() or fp.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(fp)
    except Exception:
        return None

def low_data_badge(row) -> str:
    if "low_data" in row and pd.notna(row["low_data"]):
        val = row["low_data"]
        if isinstance(val, str):
            val = val.strip().lower() in ("true", "1", "t", "yes")
        try:
            val = bool(val)
        except Exception:
            val = False
        if val:
            return "low data"
    return ""

def style_low_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "badge", [low_data_badge(r) for _, r in out.iterrows()])
    return out

def make_filters(df: pd.DataFrame, key_prefix: str):
    cols = st.columns(3)
    with cols[0]:
        show_low_only = st.checkbox(
            "Show low-data only",
            value=False,
            key=f"{key_prefix}_lowonly",
        )
    with cols[1]:
        sort_col = st.selectbox(
            "Sort by",
            options=list(df.columns),
            index=min(1, max(0, len(df.columns) - 1)),
            key=f"{key_prefix}_sortcol",
        )
    with cols[2]:
        ascending = st.checkbox(
            "Ascending sort",
            value=False,
            key=f"{key_prefix}_asc",
        )

    # Filter
    if "low_data" in df.columns and show_low_only:
        df = df[df["low_data"].astype(str).str.lower().isin(["true", "1", "t", "yes"])]

    # Sort (best-effort)
    try:
        df = df.sort_values(sort_col, ascending=ascending, kind="mergesort")
    except Exception:
        pass

    return df

def render_model_card():
    st.subheader("Model Card")
    if MODEL_CARD.exists():
        try:
            text = MODEL_CARD.read_text(encoding="utf-8", errors="ignore")
            st.markdown(text)
        except Exception as e:
            st.error(f"Could not read model card: {e}")
    else:
        st.info("No model card found. Create one at `app/reports/model_card.md`.")

def main():
    st.set_page_config(page_title="Coach View — Water Polo xG", layout="wide")
    st.title("Coach View — Water Polo xG (Phase I)")

    st.caption(f"Reading coach reports from: `{REPORT_DIR}`")
    if not REPORT_DIR.exists():
        st.error("Coach report folder not found. Generate it with coach_report.py.")
        st.stop()

    tab_names = list(PAGES.keys()) + ["Model Card"]
    tabs = st.tabs(tab_names)

    # Data tabs
    for tab, name in zip(tabs[:-1], list(PAGES.keys())):
        with tab:
            file = PAGES[name]
            df = load_csv(file)
            st.subheader(name)
            st.caption(file)
            if df is None or df.empty:
                st.warning("No data available for this table. Did you run coach_report.py?")
                continue

            # Convert common numeric columns for nicer sort
            for c in ["n", "shots", "goals", "xg_sum", "xg_ps", "xg_per_shot", "goal_rate", "xg_minus_goals", "xg_diff"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            df_disp = style_low_data(df)
            df_disp = make_filters(df_disp, key_prefix=name.replace(" ", "_").lower())

            st.dataframe(df_disp, use_container_width=True)

            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=file,
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{name.replace(' ', '_').lower()}",
            )

    # Model Card tab
    with tabs[-1]:
        render_model_card()

    st.markdown("---")
    st.caption(
        "Low-data badge = bucket has fewer than your configured minimum samples (default 10). "
        "Use for directional insight; treat with caution."
    )

if __name__ == "__main__":
    main()
