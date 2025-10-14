import pathlib, pandas as pd

p = pathlib.Path("app/reports/corrections/corrections.csv")
print("exists:", p.exists(), "| path:", p)

if p.exists():
    try:
        df = pd.read_csv(p, dtype=str)
        print("rows:", len(df))
        print("columns:", list(df.columns))
        print(df.head(20).to_string(index=False))
    except Exception as e:
        print("ERROR reading corrections.csv:", e)