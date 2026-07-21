import sqlite3
import os
import pandas as pd

db = "data/database/finops.db"
print("=== Tables in finops.db ===")
with sqlite3.connect(db) as conn:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )
    for t in tables["name"]:
        n = pd.read_sql_query(f"SELECT COUNT(*) as n FROM [{t}]", conn).iloc[0, 0]
        try:
            cols = pd.read_sql_query(f"SELECT * FROM [{t}] LIMIT 1", conn).columns.tolist()
        except Exception:
            cols = []
        print(f"  {t}: {n:,} rows | cols: {cols}")

print()
print("=== Processed files ===")
proc = "data/processed"
if os.path.exists(proc):
    for fname in os.listdir(proc):
        fp = os.path.join(proc, fname)
        sz = os.path.getsize(fp) / 1024 / 1024
        print(f"  {fname}  ({sz:.1f} MB)")
        if fname.endswith(".csv"):
            df_head = pd.read_csv(fp, nrows=3)
            total = sum(1 for _ in open(fp, encoding="utf-8")) - 1
            print(f"    rows   : {total:,}")
            print(f"    columns: {list(df_head.columns)}")
            if "date" in df_head.columns:
                df_full = pd.read_csv(fp, usecols=["date"], parse_dates=["date"])
                print(f"    date range: {df_full['date'].min().date()} -> {df_full['date'].max().date()}")
else:
    print("  data/processed directory not found")

print()
print("=== Raw files ===")
raw = "data/raw"
if os.path.exists(raw):
    for fname in os.listdir(raw):
        fp = os.path.join(raw, fname)
        sz = os.path.getsize(fp) / 1024 / 1024
        print(f"  {fname}  ({sz:.1f} MB)")
else:
    print("  data/raw directory not found")
